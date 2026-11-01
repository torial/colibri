module olmoe

// A port of the CPU path of c/olmoe.c: OLMoE (OLMoE-1B-7B and kin) from a converted merged-int8 checkpoint. A decoder
// with full multi-head attention (qk-norm over the whole hidden vector, RoPE per head), and a mixture-of-experts MLP:
// a router over `num_experts`, the top-k experts per token, each a SwiGLU of int8 weights with one f32 scale per row,
// streamed from the safetensors shards through a small LRU cache.
//
// What is ported: config.json with the engine's range checks; the dense weights (F32 or BF16 on disk, f32 in memory);
// embedding, final norm and lm_head; the layer loop; attention with a KV cache (f32, or rounded to f16 as KV16 does);
// routing (softmax, top-k, optional renormalization); the expert FFN; prefill of S tokens and one-token decode;
// greedy generation. The numbers come from the engine's own kernels (the f32 matmul's fixed 32-lane order, and
// c/olmoe.c's int8 matmul, both through `kernels`) and from libm, the loops around them written in the C code's order,
// so the logits are meant to be the C engine's bits (tests/olmoe_diff compares every logit of every step).
// Not ported: GGUF checkpoints, sampling modes and chat/serve loops (see `sample`), the prefetch pilot, hot-expert
// pinning, the GPU chains, the half-resident dense weights (COLI_DENSE_F16), IDOT / FUSED3 (opt-in lossy paths).
import os
import cjson
import math
import safetensors
import kernels

#include <math.h>

fn C.expf(x f32) f32
fn C.sqrtf(x f32) f32
fn C.powf(x f32, y f32) f32
fn C.cosf(x f32) f32
fn C.sinf(x f32) f32

const max_ctx = 4096 // attention()'s score buffer in the C engine

pub struct Cfg {
pub mut:
	hidden     int
	n_layers   int
	n_heads    int
	n_kv_heads int
	head_dim   int
	n_experts  int
	topk       int
	inter      int
	vocab      int
	theta      f32
	eps        f32
	norm_topk  bool
}

fn req_num(r &cjson.Value, k string) !f64 {
	v := r.get(k) or { return error('config.json: missing or non-numeric "${k}"') }
	if v.kind != .num {
		return error('config.json: missing or non-numeric "${k}"')
	}
	return v.num
}

// load_cfg reads <dir>/config.json: the dimensions, range-checked so bad ones cannot drive an allocation
pub fn load_cfg(dir string) !Cfg {
	path := '${dir}/config.json'
	raw := os.read_file(path) or { return error('${path}: ${err.msg()}') }
	if raw.len > (256 << 20) {
		return error('${path}: config.json missing or larger than 256 MB')
	}
	r := cjson.parse(raw)
	mut c := Cfg{}
	c.hidden = int(req_num(&r, 'hidden_size')!)
	c.n_layers = int(req_num(&r, 'num_hidden_layers')!)
	c.n_heads = int(req_num(&r, 'num_attention_heads')!)
	c.n_kv_heads = int(req_num(&r, 'num_key_value_heads')!)
	c.n_experts = int(req_num(&r, 'num_experts')!)
	c.topk = int(req_num(&r, 'num_experts_per_tok')!)
	c.inter = int(req_num(&r, 'intermediate_size')!)
	c.vocab = int(req_num(&r, 'vocab_size')!)
	if c.hidden < 1 || c.hidden > (1 << 20) || c.n_heads < 1 || c.n_heads > (1 << 16) || c.inter < 1
		|| c.inter > (1 << 24) || c.vocab < 1 || c.vocab > (1 << 24) || c.n_layers < 1
		|| c.n_layers > 4096 || c.n_experts < 1 || c.n_experts > (1 << 20) || c.n_kv_heads < 1
		|| c.topk < 1 || c.topk > c.n_experts {
		return error('config.json: dimension out of range')
	}
	c.head_dim = c.hidden / c.n_heads
	c.theta = if th := r.get('rope_theta') { f32(th.num) } else { f32(10000.0) }
	c.eps = if ep := r.get('rms_norm_eps') { f32(ep.num) } else { f32(1e-5) }
	c.norm_topk = if nt := r.get('norm_topk_prob') {
		nt.kind == .bool && nt.boolean
	} else {
		false
	}
	return c
}

// one routed expert as the engine holds it: int8 gate [I,D], up [I,D], down [D,I] in one block, one f32 scale per row
struct Expert {
	w  []u8 // gate | up | down
	gs []f32
	us []f32
	ds []f32
	// a GGUF's expert: the three matrices as ggml blocks, with their ggml types
	g  []u8
	u  []u8
	dn []u8
	gt int
	ut int
	dt int
mut:
	used u64
}

// a dense matrix [rows, cols]: f32 in memory (the checkpoint's F32 / BF16 / F16), or a GGUF's ggml blocks as the
// file holds them (`g`, of ggml type `t`), multiplied by the ggml kernels with the activations quantized to int8
struct Mat {
	f    []f32
	g    []u8
	t    int
	rows int
	cols int
}

struct Layer {
	in_ln   []f32
	post_ln []f32
	q       Mat
	k       Mat
	v       Mat
	o       Mat
	qn      []f32
	kn      []f32
	gate    Mat
}

pub struct Model {
pub mut:
	c          Cfg
	kv16       bool // round K and V to f16 before they are cached (the engine's default in chat; off in its oracle)
	keep       bool // record the logits of every step in `logits`
	logits     [][]f32
	hits       u64
	miss       u64
	cache_cap  int // experts kept per layer
	store      safetensors.Store
	embed      Mat
	lm_head    Mat
	gx         bool // routed experts are a GGUF's ggml tensors (gate_proj / up_proj / down_proj), not merged int8
	final_norm []f32
	layers     []Layer
	kcache     [][]f32 // [layer][head][t][hd]
	vcache     [][]f32
	max_t      int
	cache      []map[int]&Expert // per layer
	clock      u64
}

// load_t reads a dense tensor as f32 and checks its shape against the config
fn (m &Model) load_t(name string, rows i64, columns i64) ![]f32 {
	t := m.store.find(name) or { return error('missing ${name}') }
	want_rank := if columns != 0 { 2 } else { 1 }
	if t.shape.len != want_rank || t.shape[0] != rows || (columns != 0 && t.shape[1] != columns) {
		got := '[' + t.shape.map(it.str()).join(', ') + ']'
		want := if columns != 0 { '[${rows}, ${columns}]' } else { '[${rows}]' }
		return error('${name}: shape ${got}, expected ${want} from config.json, refusing (untrusted container)')
	}
	return m.store.read_f32(name)!
}

// load_mat is load_t for a matrix a GGUF may hold in ggml blocks
fn (m &Model) load_mat(name string, rows i64, columns i64) !Mat {
	t := m.store.find(name) or { return error('missing ${name}') }
	if t.is_ggml() {
		rb := kernels.gq_row_bytes(t.ggml, columns)
		if t.shape.len != 2 || t.shape[0] != rows || t.shape[1] != columns || rb < 0 || t.nbytes != rb * rows {
			return error('${name}: not [${rows}, ${columns}] in ggml type ${kernels.gq_type_name(t.ggml)}, refusing (untrusted container)')
		}
		return Mat{
			g:    m.store.read_raw(name)!
			t:    t.ggml
			rows: int(rows)
			cols: int(columns)
		}
	}
	return Mat{
		f:    m.load_t(name, rows, columns)!
		rows: int(rows)
		cols: int(columns)
	}
}

// mm: y[S,O] = x[S,I] . W^T
fn (m &Model) mm(x []f32, s_n int, w &Mat) ![]f32 {
	if w.g.len > 0 {
		return kernels.gq_matmul(x, w.g, w.t, s_n, w.cols, w.rows)
	}
	return kernels.matmul_f32(x, w.f, s_n, w.cols, w.rows)
}

// open loads the dense weights of the checkpoint in `dir`; `cap` experts per layer stay in the cache
pub fn open(dir string, cap int) !Model {
	c := load_cfg(dir)!
	mut m := Model{
		c:         c
		cache_cap: if cap < 1 { 1 } else { cap }
		store:     safetensors.open(dir)!
	}
	d := i64(c.hidden)
	m.embed = m.load_mat('model.embed_tokens.weight', c.vocab, d)!
	m.lm_head = m.load_mat('lm_head.weight', c.vocab, d)!
	m.gx = m.store.has('model.layers.0.mlp.experts.0.gate_proj.weight')
	m.final_norm = m.load_t('model.norm.weight', d, 0)!
	for i in 0 .. c.n_layers {
		p := 'model.layers.${i}.'
		m.layers << Layer{
			in_ln:   m.load_t(p + 'input_layernorm.weight', d, 0)!
			post_ln: m.load_t(p + 'post_attention_layernorm.weight', d, 0)!
			q:       m.load_mat(p + 'self_attn.q_proj.weight', d, d)!
			k:       m.load_mat(p + 'self_attn.k_proj.weight', d, d)!
			v:       m.load_mat(p + 'self_attn.v_proj.weight', d, d)!
			o:       m.load_mat(p + 'self_attn.o_proj.weight', d, d)!
			qn:      m.load_t(p + 'self_attn.q_norm.weight', d, 0)!
			kn:      m.load_t(p + 'self_attn.k_norm.weight', d, 0)!
			gate:    m.load_mat(p + 'mlp.gate.weight', c.n_experts, d)!
		}
		m.cache << map[int]&Expert{}
	}
	return m
}

pub fn (mut m Model) close() {
	m.store.close()
}

// ---- the arithmetic of the layers, in the C engine's order ----

fn rmsnorm_row(mut out []f32, ooff int, x []f32, xoff int, w []f32, d int, eps f32) {
	mut ms := f64(0)
	for i in 0 .. d {
		ms += f64(x[xoff + i]) * f64(x[xoff + i])
	}
	r := f32(1.0) / C.sqrtf(f32(ms / f64(d)) + eps)
	for i in 0 .. d {
		out[ooff + i] = x[xoff + i] * r * w[i]
	}
}

fn softmax_row(mut x []f32, n int) {
	mut mx := f32(-1e30)
	for i in 0 .. n {
		if x[i] > mx {
			mx = x[i]
		}
	}
	mut s := f32(0)
	for i in 0 .. n {
		x[i] = C.expf(x[i] - mx)
		s += x[i]
	}
	for i in 0 .. n {
		x[i] /= s
	}
}

fn rope_head(mut x []f32, off int, pos int, c &Cfg) {
	h := c.head_dim / 2
	for j in 0 .. h {
		inv := C.powf(c.theta, f32(-2.0) * f32(j) / f32(c.head_dim))
		ang := f32(pos) * inv
		cs := C.cosf(ang)
		sn := C.sinf(ang)
		a := x[off + j]
		b := x[off + j + h]
		x[off + j] = a * cs - b * sn
		x[off + j + h] = b * cs + a * sn
	}
}

// kv16_round: the number a float16 cache holds for x (clamped to +-65504, ties to even), as c/kv16.h
fn kv16_round(x f32) f32 {
	mut f := x
	if f > 65504.0 {
		f = 65504.0
	} else if f < -65504.0 {
		f = -65504.0
	}
	u := math.f32_bits(f)
	s := u & 0x80000000
	a := u & 0x7fffffff
	if a > 0x7f800000 {
		return math.f32_from_bits(u | 0x400000) // a NaN stays one
	}
	if a >= 0x38800000 { // float16's normals: 10 bits of mantissa
		return math.f32_from_bits(s | ((a + 0xfff + ((a >> 13) & 1)) & 0xffffe000))
	}
	e := a >> 23
	mn := (a & 0x7fffff) | 0x800000
	if e < 102 {
		return math.f32_from_bits(s) // under 2^-25: zero
	}
	sh := 126 - e
	n := (mn + (u32(1) << (sh - 1)) - 1 + ((mn >> sh) & 1)) >> sh
	r := f32(n) * f32(5.9604644775390625e-8) // n * 2^-24, exact
	return math.f32_from_bits(s | math.f32_bits(r))
}

// expert returns routed expert `eid` of `layer`, reading it from the shards on a miss and evicting the least recently used
fn (mut m Model) expert(layer int, eid int) !&Expert {
	m.clock++
	if mut e := m.cache[layer][eid] {
		e.used = m.clock
		m.hits++
		return e
	}
	m.miss++
	c := &m.c
	if m.gx {
		return m.expert_gx(layer, eid)
	}
	ng := i64(c.inter) * c.hidden
	nd := i64(c.hidden) * c.inter
	want_w := ng + ng + nd
	want_s := i64(c.inter) + c.inter + c.hidden
	nm := 'model.layers.${layer}.mlp.experts.${eid}.merged_weight'
	qsnm := 'model.layers.${layer}.mlp.experts.${eid}.qs'
	tw := m.store.find(nm) or {
		return error('${nm} is missing; this OLMoE engine requires a converted merged-int8 checkpoint. Run python tools/convert_olmoe_merged.py --model <source-model> --out <converted-model> first, then use <converted-model> as the model.')
	}
	if tw.nbytes != want_w {
		return error('${nm}: expert weight is ${tw.nbytes} bytes — expected ${want_w} for [inter=${c.inter},hidden=${c.hidden}], refusing (untrusted container)')
	}
	ts := m.store.find(qsnm) or {
		return error('${qsnm} is missing; this OLMoE engine requires a converted merged-int8 checkpoint. Run python tools/convert_olmoe_merged.py --model <source-model> --out <converted-model> first, then use <converted-model> as the model.')
	}
	if ts.numel != want_s {
		return error('${qsnm}: scale array is ${ts.numel} elems — expected ${want_s}, refusing (untrusted container)')
	}
	w := m.store.read_raw(nm)!
	sc := m.store.read_f32(qsnm)!
	e := &Expert{
		w:    w
		gs:   unsafe { sc[..c.inter] }
		us:   unsafe { sc[c.inter..2 * c.inter] }
		ds:   unsafe { sc[2 * c.inter..] }
		used: m.clock
	}
	if m.cache[layer].len >= m.cache_cap {
		mut victim := -1
		mut oldest := u64(0xffffffffffffffff)
		for k, v in m.cache[layer] {
			if v.used < oldest {
				oldest = v.used
				victim = k
			}
		}
		m.cache[layer].delete(victim)
	}
	m.cache[layer][eid] = e
	return e
}

// ggml_type_of: the ggml type a tensor is held in: its block type, or the float type of a plain F32 / F16 / BF16
fn ggml_type_of(t &safetensors.Tensor) int {
	if t.is_ggml() {
		return t.ggml
	}
	return match t.dtype {
		.f32 { 0 }
		.f16 { 1 }
		.bf16 { 30 }
		else { -1 }
	}
}

fn (mut m Model) read_gx(layer int, eid int, kind string, rows i64, cols i64) !([]u8, int) {
	nm := 'model.layers.${layer}.mlp.experts.${eid}.${kind}.weight'
	t := m.store.find(nm) or { return error('missing ${nm}') }
	typ := ggml_type_of(t)
	rb := if typ >= 0 { kernels.gq_row_bytes(typ, cols) } else { i64(-1) }
	if typ < 0 || rb < 0 || t.nbytes != rb * rows {
		return error('${nm}: not ${rows} rows of ${cols} in a ggml type')
	}
	return m.store.read_raw(nm)!, typ
}

// expert_gx reads a GGUF's expert: three tensors of ggml blocks
fn (mut m Model) expert_gx(layer int, eid int) !&Expert {
	c := &m.c
	g, gt := m.read_gx(layer, eid, 'gate_proj', c.inter, c.hidden)!
	u, ut := m.read_gx(layer, eid, 'up_proj', c.inter, c.hidden)!
	dn, dt := m.read_gx(layer, eid, 'down_proj', c.hidden, c.inter)!
	e := &Expert{
		g:    g
		u:    u
		dn:   dn
		gt:   gt
		ut:   ut
		dt:   dt
		used: m.clock
	}
	if m.cache[layer].len >= m.cache_cap {
		mut victim := -1
		mut oldest := u64(0xffffffffffffffff)
		for k, v in m.cache[layer] {
			if v.used < oldest {
				oldest = v.used
				victim = k
			}
		}
		m.cache[layer].delete(victim)
	}
	m.cache[layer][eid] = e
	return e
}

// expert_row: out[D] = down(silu(gate(x)) * up(x)) for one row
fn (m &Model) expert_row(e &Expert, xs []f32) ![]f32 {
	c := &m.c
	d := c.hidden
	ii := c.inter
	if m.gx {
		mut g := kernels.gq_matmul(xs, e.g, e.gt, 1, d, ii)!
		u := kernels.gq_matmul(xs, e.u, e.ut, 1, d, ii)!
		for i in 0 .. ii {
			gv := g[i]
			g[i] = (gv / (f32(1.0) + C.expf(-gv))) * u[i]
		}
		return kernels.gq_matmul(g, e.dn, e.dt, 1, ii, d)
	}
	ng := ii * d
	mut g := kernels.olm_matmul_q(xs, unsafe { e.w[..ng] }, e.gs, d, ii)!
	u := kernels.olm_matmul_q(xs, unsafe { e.w[ng..2 * ng] }, e.us, d, ii)!
	for i in 0 .. ii {
		gv := g[i]
		g[i] = (gv / (f32(1.0) + C.expf(-gv))) * u[i]
	}
	return kernels.olm_matmul_q(g, unsafe { e.w[2 * ng..] }, e.ds, ii, d)
}

// route_row: softmax of the router logits, the top-k experts (first of equals), optional renormalization
fn (m &Model) route_row(mut pr []f32) ([]int, []f32) {
	c := &m.c
	e_n := c.n_experts
	k_n := c.topk
	softmax_row(mut pr, e_n)
	mut idx := []int{len: k_n}
	mut val := []f32{len: k_n}
	for kk in 0 .. k_n {
		mut best := -1
		mut bv := f32(-1e30)
		for e in 0 .. e_n {
			mut taken := false
			for j in 0 .. kk {
				if idx[j] == e {
					taken = true
					break
				}
			}
			if !taken && pr[e] > bv {
				bv = pr[e]
				best = e
			}
		}
		if best < 0 { // all-NaN probabilities: a reproducible in-range pick, as rt_router_pick
			best = if kk < e_n { kk } else { 0 }
		}
		idx[kk] = best
		val[kk] = pr[best]
	}
	if c.norm_topk {
		mut sm := f32(0)
		for kk in 0 .. k_n {
			sm += val[kk]
		}
		for kk in 0 .. k_n {
			val[kk] /= sm
		}
	}
	return idx, val
}

// moe: the routed MLP of `layer` on S rows (x is [S,D]); returns [S,D]
fn (mut m Model) moe(layer int, x []f32, s_n int) ![]f32 {
	c := &m.c
	d := c.hidden
	e_n := c.n_experts
	logits := m.mm(x, s_n, &m.layers[layer].gate)!
	mut out := []f32{len: s_n * d}
	for s in 0 .. s_n {
		mut pr := unsafe { logits[s * e_n..(s + 1) * e_n] }.clone()
		idx, val := m.route_row(mut pr)
		xs := unsafe { x[s * d..(s + 1) * d] }
		for kk in 0 .. c.topk {
			e := m.expert(layer, idx[kk])!
			hh := m.expert_row(e, xs)!
			w := val[kk]
			for j in 0 .. d {
				out[s * d + j] += w * hh[j]
			}
		}
	}
	return out
}

// attention on the new tokens x[S,D]; pos_base is the absolute position of the first
fn (mut m Model) attention(layer int, x []f32, s_n int, pos_base int) ![]f32 {
	c := &m.c
	h_n := c.n_heads
	hd := c.head_dim
	d := c.hidden
	l := &m.layers[layer]
	mut q := m.mm(x, s_n, &l.q)!
	mut k := m.mm(x, s_n, &l.k)!
	mut vv := m.mm(x, s_n, &l.v)!
	for s in 0 .. s_n {
		qsrc := q[s * d..(s + 1) * d].clone()
		rmsnorm_row(mut q, s * d, qsrc, 0, l.qn, d, c.eps)
		ksrc := k[s * d..(s + 1) * d].clone()
		rmsnorm_row(mut k, s * d, ksrc, 0, l.kn, d, c.eps)
		pos := pos_base + s
		for hh in 0 .. h_n {
			rope_head(mut q, s * d + hh * hd, pos, c)
			rope_head(mut k, s * d + hh * hd, pos, c)
		}
	}
	if m.kv16 {
		for i in 0 .. k.len {
			k[i] = kv16_round(k[i])
		}
		for i in 0 .. vv.len {
			vv[i] = kv16_round(vv[i])
		}
	}
	for s in 0 .. s_n {
		for hh in 0 .. h_n {
			t := pos_base + s
			base := (hh * m.max_t + t) * hd
			for j in 0 .. hd {
				m.kcache[layer][base + j] = k[s * d + hh * hd + j]
				m.vcache[layer][base + j] = vv[s * d + hh * hd + j]
			}
		}
	}
	scale := f32(1.0) / C.sqrtf(f32(hd))
	mut ctx := []f32{len: s_n * d}
	mut sc := []f32{len: max_ctx}
	for hh in 0 .. h_n {
		for s in 0 .. s_n {
			qpos := pos_base + s
			qv := unsafe { q[s * d + hh * hd..s * d + (hh + 1) * hd] }
			for t in 0 .. qpos + 1 { // causal: t <= qpos
				kb := (hh * m.max_t + t) * hd
				acc := kernels.dot_f32(qv, unsafe { m.kcache[layer][kb..kb + hd] })!
				sc[t] = acc * scale
			}
			softmax_row(mut sc, qpos + 1)
			o := s * d + hh * hd
			for t in 0 .. qpos + 1 {
				vb := (hh * m.max_t + t) * hd
				a := sc[t]
				for dd in 0 .. hd {
					ctx[o + dd] += a * m.vcache[layer][vb + dd]
				}
			}
		}
	}
	return m.mm(ctx, s_n, &l.o)
}

// step feeds `ids` (S tokens at positions pos_base..) through the layers and returns the logits of the last
fn (mut m Model) step(ids []int, pos_base int) ![]f32 {
	c := &m.c
	d := c.hidden
	s_n := ids.len
	mut x := []f32{len: s_n * d}
	for s in 0 .. s_n {
		id := ids[s]
		if id < 0 || id >= c.vocab {
			return error('token id ${id} outside the vocabulary of ${c.vocab}')
		}
		if m.embed.g.len > 0 {
			rb := kernels.gq_row_bytes(m.embed.t, d)
			row := kernels.gq_dequant_row(m.embed.t, unsafe { m.embed.g[i64(id) * rb..(i64(id) + 1) * rb] }, d)!
			for j in 0 .. d {
				x[s * d + j] = row[j]
			}
		} else {
			for j in 0 .. d {
				x[s * d + j] = m.embed.f[id * d + j]
			}
		}
	}
	mut nrm := []f32{len: s_n * d}
	for i in 0 .. c.n_layers {
		l := &m.layers[i]
		for s in 0 .. s_n {
			rmsnorm_row(mut nrm, s * d, x, s * d, l.in_ln, d, c.eps)
		}
		tmp := m.attention(i, nrm, s_n, pos_base)!
		for j in 0 .. s_n * d {
			x[j] += tmp[j]
		}
		for s in 0 .. s_n {
			rmsnorm_row(mut nrm, s * d, x, s * d, l.post_ln, d, c.eps)
		}
		tmp2 := m.moe(i, nrm, s_n)!
		for j in 0 .. s_n * d {
			x[j] += tmp2[j]
		}
	}
	mut last := []f32{len: d}
	rmsnorm_row(mut last, 0, x, (s_n - 1) * d, m.final_norm, d, c.eps)
	logit := m.mm(last, 1, &m.lm_head)!
	if m.keep {
		m.logits << logit.clone()
	}
	return logit
}

fn argmax(l []f32) int {
	mut best := 0
	mut bv := l[0]
	for i in 1 .. l.len {
		if l[i] > bv {
			bv = l[i]
			best = i
		}
	}
	return best
}

// generate prefills `prompt` and decodes `n_new` tokens greedily; returns prompt + new tokens
pub fn (mut m Model) generate(prompt []int, n_new int) ![]int {
	if prompt.len == 0 {
		return error('empty prompt')
	}
	m.max_t = prompt.len + n_new
	if m.max_t > max_ctx {
		return error('context of ${m.max_t} tokens exceeds the ${max_ctx} the attention buffers hold')
	}
	c := &m.c
	m.kcache = [][]f32{len: c.n_layers, init: []f32{len: c.n_heads * m.max_t * c.head_dim}}
	m.vcache = [][]f32{len: c.n_layers, init: []f32{len: c.n_heads * m.max_t * c.head_dim}}
	mut out := prompt.clone()
	mut logit := m.step(prompt, 0)! // prefill
	for s in 0 .. n_new {
		best := argmax(logit)
		out << best
		if s == n_new - 1 {
			break
		}
		logit = m.step([best], out.len - 1)! // decode
	}
	return out
}
