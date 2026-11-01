module sample

// A port of c/sample.h: temperature + nucleus sampling, the stop set, and the log-probability helpers.
// The numeric core is bit-for-bit the C code's: the same xorshift64* generator, the same float / double
// arithmetic in the same order, expf / exp / log called from libm. tests/sample_diff compares picks and
// distributions with the C original.
// Differences: state lives in a `Sampler` / `StopSet` instead of file-scope globals; the stop-set's
// environment switches (SERVE, SERVE_BATCH, COLI_SERVE_ALL_STOPS) are read by the caller and passed in
// `StopOpts`; the COLI_LOGIT_DUMP / COLI_LOGIT_GAP stderr probes are not ported.
import math
import tokenizer

#include <math.h>

fn C.expf(x f32) f32
fn C.exp(x f64) f64
fn C.log(x f64) f64

@[inline]
fn finite(x f32) bool {
	return math.is_finite(f64(x))
}

@[inline]
fn isnan32(x f32) bool {
	return math.is_nan(f64(x))
}

pub struct Sampler {
pub mut:
	rng    u64 = u64(0x9E3779B97F4A7C15)
	temp   f32   // <= 0: greedy
	nuc    f32   // top-p; outside (0,1): no truncation
	pbuf   []f32 // distribution, indexed by token id
	pidx   []int
	warned bool
}

// rndu: xorshift64*-style generator of c/sample.h, a double in [0,1)
pub fn (mut s Sampler) rndu() f64 {
	s.rng ^= s.rng << 13
	s.rng ^= s.rng >> 7
	s.rng ^= s.rng << 17
	return f64(s.rng >> 11) * (1.0 / 9007199254740992.0)
}

// argmax_v: first largest non-NaN entry; 0 if there is none
pub fn argmax_v(lo []f32) int {
	mut b := -1
	mut bv := f32(math.inf(-1))
	for i in 0 .. lo.len {
		x := lo[i]
		if x == x && x > bv {
			bv = x
			b = i
		}
	}
	return if b < 0 { 0 } else { b }
}

fn (s &Sampler) siftdown(mut h []int, n int, i0 int) {
	mut i := i0
	iv := h[i]
	kv := s.pbuf[iv]
	for {
		l := 2 * i + 1
		if l >= n {
			break
		}
		mut b := l
		if l + 1 < n && s.pbuf[h[l + 1]] > s.pbuf[h[l]] {
			b = l + 1
		}
		if s.pbuf[h[b]] <= kv {
			break
		}
		h[i] = h[b]
		i = b
	}
	h[i] = iv
}

// dist_build: softmax(lo/temp), truncated to top-p `nuc`; pbuf stays indexed by token id
pub fn (mut s Sampler) dist_build(lo []f32) {
	v := lo.len
	if s.pbuf.len != v {
		s.pbuf = []f32{len: v}
		s.pidx = []int{len: v}
	}
	mut mxi := -1
	mut mx := f32(0)
	for i in 0 .. v {
		if finite(lo[i]) && (mxi < 0 || lo[i] > mx) {
			mx = lo[i]
			mxi = i
		}
	}
	mut sum := f64(0)
	invt := f32(1.0) / (if s.temp > f32(1e-4) { s.temp } else { f32(1e-4) })
	if mxi >= 0 {
		for i in 0 .. v {
			s.pbuf[i] = if finite(lo[i]) { C.expf((lo[i] - mx) * invt) } else { f32(0) }
			sum += f64(s.pbuf[i])
		}
	}
	if mxi < 0 || !math.is_finite(sum) || sum <= 0.0 {
		if !s.warned {
			s.warned = true
			eprintln('[SAMPLE] warning: non-finite logits (NaN/Inf) — falling back to argmax; output may be degraded. This usually means a numerical blow-up upstream.')
		}
		a := if mxi >= 0 { mxi } else { 0 }
		for i in 0 .. v {
			s.pbuf[i] = 0
		}
		s.pbuf[a] = 1
		return
	}
	for i in 0 .. v {
		s.pbuf[i] /= f32(sum)
	}
	if s.nuc > 0 && s.nuc < f32(1.0) {
		for i in 0 .. v {
			s.pidx[i] = i
		}
		mut i := v / 2 - 1
		for i >= 0 {
			s.siftdown(mut s.pidx, v, i)
			i--
		}
		mut s2 := f64(0)
		mut cum := f64(0)
		mut out := v
		for {
			root := s.pidx[0]
			out--
			s.pidx[0] = s.pidx[out]
			s.pidx[out] = root
			s2 += f64(s.pbuf[root])
			cum += f64(s.pbuf[root])
			if out > 0 {
				s.siftdown(mut s.pidx, out, 0)
			}
			if !(cum < f64(s.nuc) && out > 0) {
				break
			}
		}
		for k in 0 .. out {
			s.pbuf[s.pidx[k]] = 0
		}
		s2f := f32(s2)
		for k in out .. v {
			s.pbuf[s.pidx[k]] /= s2f
		}
	}
}

// dist_sample: draw from pbuf; ban >= 0 excludes that token (renormalizing on the fly)
pub fn (mut s Sampler) dist_sample(ban int) int {
	v := s.pbuf.len
	mut z := 1.0 - (if ban >= 0 { f64(s.pbuf[ban]) } else { 0.0 })
	if z <= 1e-12 {
		z = 1e-12
	}
	u := s.rndu() * z
	mut cum := f64(0)
	for i in 0 .. v {
		if i == ban {
			continue
		}
		cum += f64(s.pbuf[i])
		if cum >= u {
			return i
		}
	}
	for i := v - 1; i >= 0; i-- {
		if i != ban && s.pbuf[i] > 0 {
			return i
		}
	}
	return 0
}

// pick_tok: greedy if temp <= 0, sampling otherwise. ban = token rejected by speculative verification.
pub fn (mut s Sampler) pick_tok(lo []f32, ban int) int {
	if s.temp <= 0 {
		return argmax_v(lo)
	}
	s.dist_build(lo)
	return s.dist_sample(ban)
}

// ---- stop set ----

pub struct StopSet {
pub mut:
	ids []int
}

pub struct StopOpts {
pub:
	serve_batch bool // SERVE set and SERVE_BATCH nonzero
	all_stops   bool // COLI_SERVE_ALL_STOPS set
	quiet       bool // no [stop] lines on stderr
}

pub fn (st &StopSet) is_stop(t int) bool {
	return t in st.ids
}

const content_ctl = ['[THINK]', '[/THINK]', '[TOOL_CALLS]', '[ARGS]', '[CALL_ID]']

// arm: the config's stop ids, the tokenizer's eos, then (given a tokenizer) its special tokens that are
// not content-carrying control tokens; at most 64
pub fn arm(cfg_stops []int, tok_eos int, t &tokenizer.Tok, opts StopOpts) StopSet {
	mut st := StopSet{}
	for id in cfg_stops {
		if st.ids.len >= 64 {
			break
		}
		st.ids << id
	}
	if tok_eos >= 0 && !st.is_stop(tok_eos) && st.ids.len < 64 {
		st.ids << tok_eos
	}
	mut nsp := 0
	if !isnil(t) {
		for id in 0 .. t.n_ids {
			if st.ids.len >= 64 {
				break
			}
			if !t.id_special[id] || st.is_stop(id) {
				continue
			}
			mut content := false
			if t.has_str[id] && t.id2str[id] in content_ctl {
				content = true
			}
			if !content {
				st.ids << id
				nsp++
			}
		}
	}
	mut np := nsp
	if opts.serve_batch && tok_eos >= 0 && !opts.all_stops {
		mut kept := []int{}
		for id in st.ids {
			if id == tok_eos {
				kept << id
			}
		}
		if kept.len < st.ids.len && !opts.quiet {
			eprintln('[stop] batched serve mode: filtered ${st.ids.len - kept.len} non-EOS stop tokens (tool-call safety, #401)')
		}
		st.ids = kept
		np = 0
	}
	if !opts.quiet {
		mut line := '[stop] ${st.ids.len} stop tokens:'
		for id in st.ids {
			line += ' ${id}'
		}
		if np > 0 {
			line += " (${np} from the tokenizer's special set)"
		}
		eprintln(line)
	}
	return st
}

// ---- classified log-prob row reduction ----

pub enum LogprobStatus {
	finite
	nan
	pos_inf
	neg_inf
	all_nonfinite
	finite_overflow
	invalid
}

pub fn (s LogprobStatus) name() string {
	return match s {
		.finite { 'FINITE' }
		.nan { 'NAN' }
		.pos_inf { 'POS_INF' }
		.neg_inf { 'NEG_INF' }
		.all_nonfinite { 'ALL_NONFINITE' }
		.finite_overflow { 'FINITE_OVERFLOW' }
		.invalid { 'INVALID' }
	}
}

pub struct LogprobRow {
pub mut:
	max    f32
	logse  f64
	logz   f64
	argmax int
	status LogprobStatus = .invalid
}

// logprob_row_checked reduces one logit row and says why it could not
pub fn logprob_row_checked(lo []f32) LogprobRow {
	mut r := LogprobRow{}
	v := lo.len
	if v <= 0 {
		return r
	}
	mut finite_count := 0
	mut saw_nan := false
	mut saw_pos := false
	mut saw_neg := false
	for x in lo {
		if finite(x) {
			finite_count++
		} else if isnan32(x) {
			saw_nan = true
		} else if x > 0 {
			saw_pos = true
		} else {
			saw_neg = true
		}
	}
	if finite_count != v {
		r.status = if finite_count == 0 {
			.all_nonfinite
		} else if saw_nan {
			.nan
		} else if saw_pos {
			.pos_inf
		} else if saw_neg {
			.neg_inf
		} else {
			.invalid
		}
		return r
	}
	r.max = lo[0]
	r.argmax = 0
	for i in 1 .. v {
		if lo[i] > r.max {
			r.max = lo[i]
			r.argmax = i
		}
	}
	mut se := f64(0)
	for i in 0 .. v {
		term := C.exp(f64(lo[i]) - f64(r.max))
		if !math.is_finite(term) || !math.is_finite(se + term) {
			r.status = .finite_overflow
			return r
		}
		se += term
	}
	r.logse = C.log(se)
	r.logz = f64(r.max) + r.logse
	r.status = if math.is_finite(r.logse) && math.is_finite(r.logz) {
		.finite
	} else {
		.finite_overflow
	}
	return r
}

// logprob_from_row_checked reads one target out of a reduced row; the value is usable only with .finite
pub fn logprob_from_row_checked(lo []f32, target int, r &LogprobRow) (LogprobStatus, f64) {
	nan := math.nan()
	if lo.len == 0 || isnil(r) || target < 0 {
		return .invalid, nan
	}
	if r.status != .finite {
		return r.status, nan
	}
	value := f64(lo[target]) - r.logz
	if !math.is_finite(value) {
		return .finite_overflow, nan
	}
	return .finite, value
}

// logprob_target: log-probability of `target`, and whether it is the argmax (first largest, as the C loop)
pub fn logprob_target(lo []f32, target int) (f64, bool) {
	mut mx := lo[0]
	mut best := 0
	for i in 1 .. lo.len {
		if lo[i] > mx {
			mx = lo[i]
			best = i
		}
	}
	mut se := f64(0)
	for x in lo {
		se += C.exp(f64(x) - f64(mx))
	}
	return f64(lo[target] - mx) - C.log(se), best == target
}

// mt_is_glm: "glm" anywhere in model_type, case-insensitive
pub fn mt_is_glm(s string) bool {
	b := s.bytes()
	for i in 0 .. b.len {
		if i + 2 < b.len && (b[i] | 32) == `g` && (b[i + 1] | 32) == `l` && (b[i + 2] | 32) == `m` {
			return true
		}
	}
	return false
}
