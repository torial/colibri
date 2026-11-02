module glm

// The exact-math (f32) path of c/colibri.c, the GLM-5.2 engine (model_type glm_moe_dsa), in V.
// What is here: config.json; embedding, final norm, lm_head; MLA attention (low-rank query, a shared low-rank KV latent
// with a decoupled RoPE key, interleaved partial RoPE, attention done in latent space as the engine does it: the query
// absorbed through kv_b's key rows, the value read through its value rows); leading dense SwiGLU layers; sparse layers
// with a sigmoid router whose selection bias picks but does not weigh, top-k, renormalization and routed scale, int-free
// f32 routed experts through an LRU cache, and a shared expert; prefill and decode; greedy generation.
// The arithmetic follows the C engine's order (the f32 matmul's 32-lane sums through `kernels`, the latent dot products and
// value accumulation as the engine writes them), and the test (tests/glm_diff) compares every logit of every step with
// `./colibri 16 16 16` (16-bit expert and dense bits: the engine's unquantized path).
// Not here: the quantized dense / expert formats (int8, int4, grouped, planar, ggml), the DSA lightning indexer (it selects
// every key up to index_topk, so with index_topk 0 or a short context it is a no-op), Mistral-Large-4 variants, the MTP head,
// speculation, sampling, the tiers and GPU paths.
import os
import cjson
import safetensors
import kernels

#include <math.h>

fn C.expf(x f32) f32
fn C.sqrtf(x f32) f32
fn C.powf(x f32, y f32) f32
fn C.cosf(x f32) f32
fn C.sinf(x f32) f32
fn C.fmaf(a f32, b f32, c f32) f32

pub struct Cfg {
pub mut:
	hidden      int
	n_layers    int
	n_heads     int
	n_experts   int
	topk        int
	moe_inter   int
	dense_inter int
	first_dense int
	q_lora      int
	kv_lora     int
	qk_nope     int
	qk_rope     int
	qk_head     int
	v_head      int
	n_shared    int
	vocab       int
	n_group     int
	norm_topk   bool
	eps         f32
	theta       f32
	attn_scale  f32
	routed      f32
	index_topk  int
}

fn gi(r &cjson.Value, k string) int {
	v := r.get(k) or { return 0 }
	return int(v.num)
}

pub fn load_cfg(dir string) !Cfg {
	path := '${dir}/config.json'
	raw := os.read_file(path) or { return error('${path}: ${err.msg()}') }
	r := cjson.parse(raw)
	mut c := Cfg{}
	c.hidden = gi(&r, 'hidden_size')
	c.n_layers = gi(&r, 'num_hidden_layers')
	c.n_heads = gi(&r, 'num_attention_heads')
	c.n_experts = gi(&r, 'n_routed_experts')
	c.topk = gi(&r, 'num_experts_per_tok')
	c.moe_inter = gi(&r, 'moe_intermediate_size')
	c.dense_inter = gi(&r, 'intermediate_size')
	c.first_dense = gi(&r, 'first_k_dense_replace')
	c.q_lora = gi(&r, 'q_lora_rank')
	c.kv_lora = gi(&r, 'kv_lora_rank')
	c.qk_nope = gi(&r, 'qk_nope_head_dim')
	c.qk_rope = gi(&r, 'qk_rope_head_dim')
	c.v_head = gi(&r, 'v_head_dim')
	c.n_shared = gi(&r, 'n_shared_experts')
	c.vocab = gi(&r, 'vocab_size')
	c.n_group = gi(&r, 'n_group')
	c.index_topk = gi(&r, 'index_topk')
	c.norm_topk = if nt := r.get('norm_topk_prob') { nt.kind == .bool && nt.boolean } else { false }
	c.eps = if ep := r.get('rms_norm_eps') { f32(ep.num) } else { f32(1e-5) }
	c.routed = if rs := r.get('routed_scaling_factor') { f32(rs.num) } else { f32(1.0) }
	c.theta = f32(10000.0)
	if rp := r.get('rope_parameters') {
		if th := rp.get('rope_theta') {
			c.theta = f32(th.num)
		}
	}
	c.qk_head = c.qk_nope + c.qk_rope
	c.attn_scale = f32(1.0) / C.sqrtf(f32(c.qk_head))
	if c.n_group != 1 {
		return error('this engine requires n_group=1 (GLM-5.2)')
	}
	if c.hidden < 1 || c.n_layers < 1 || c.n_heads < 1 || c.n_experts < 1 || c.topk < 1 || c.topk > c.n_experts
		|| c.vocab < 1 || c.qk_rope < 2 || c.qk_rope % 2 != 0 || c.qk_rope > 256 || c.kv_lora < 1
		|| c.kv_lora > 512 || c.q_lora < 1 || c.v_head < 1 || c.qk_nope < 1 {
		return error('config: dimension out of range')
	}
	return c
}

struct Layer {
	in_ln   []f32
	post_ln []f32
	q_a     []f32
	q_a_ln  []f32
	q_b     []f32
	kv_a    []f32
	kv_a_ln []f32
	kv_b    []f32
	o       []f32
	sparse  bool
	gate    []f32 // dense MLP
	up      []f32
	down    []f32
	router  []f32 // sparse
	bias    []f32
	sh_gate []f32
	sh_up   []f32
	sh_down []f32
}

struct Expert {
	g []f32
	u []f32
	d []f32
mut:
	used u64
}

pub struct Model {
pub mut:
	c          Cfg
	keep       bool
	logits     [][]f32
	hits       u64
	miss       u64
	cache_cap  int
	store      safetensors.Store
	embed      []f32
	lm_head    []f32
	final_norm []f32
	layers     []Layer
	lat        [][]f32 // [layer][t][kv_lora]: the normed KV latent
	rop        [][]f32 // [layer][t][qk_rope]: the roped shared key
	max_t      int
	cache      []map[int]&Expert
	clock      u64
	blk        int  // block width of the engine's scalar dot loops: 8 (AVX-512), 4 (AVX2), 0
	avx2       bool // the kernels were built for AVX2: the engine's score dot has its 8-lane form
}

fn (m &Model) load(name string, want i64) ![]f32 {
	t := m.store.find(name) or { return error('missing ${name}') }
	if t.numel != want {
		return error('${name}: ${t.numel} elements, config.json asks for ${want}, refusing (untrusted container)')
	}
	return m.store.read_f32(name)!
}

pub fn open(dir string, cap int) !Model {
	c := load_cfg(dir)!
	mut m := Model{
		c:         c
		avx2:      kernels.isa().contains('avx2')
		blk:       if kernels.isa().contains('avx512f') {
			8
		} else if kernels.isa().contains('avx2') {
			4
		} else {
			0
		}
		cache_cap: if cap < 1 { 1 } else { cap }
		store:     safetensors.open(dir)!
	}
	d := i64(c.hidden)
	h := i64(c.n_heads)
	m.embed = m.load('model.embed_tokens.weight', i64(c.vocab) * d)!
	m.lm_head = m.load('lm_head.weight', i64(c.vocab) * d)!
	m.final_norm = m.load('model.norm.weight', d)!
	for i in 0 .. c.n_layers {
		p := 'model.layers.${i}.'
		sparse := i >= c.first_dense
		si := i64(c.moe_inter) * c.n_shared
		mut l := Layer{
			in_ln:   m.load(p + 'input_layernorm.weight', d)!
			post_ln: m.load(p + 'post_attention_layernorm.weight', d)!
			q_a:     m.load(p + 'self_attn.q_a_proj.weight', i64(c.q_lora) * d)!
			q_a_ln:  m.load(p + 'self_attn.q_a_layernorm.weight', c.q_lora)!
			q_b:     m.load(p + 'self_attn.q_b_proj.weight', h * c.qk_head * c.q_lora)!
			kv_a:    m.load(p + 'self_attn.kv_a_proj_with_mqa.weight', i64(c.kv_lora + c.qk_rope) * d)!
			kv_a_ln: m.load(p + 'self_attn.kv_a_layernorm.weight', c.kv_lora)!
			kv_b:    m.load(p + 'self_attn.kv_b_proj.weight', h * (c.qk_nope + c.v_head) * c.kv_lora)!
			o:       m.load(p + 'self_attn.o_proj.weight', d * h * c.v_head)!
			sparse:  sparse
		}
		if sparse {
			l = Layer{
				...l
				router:  m.load(p + 'mlp.gate.weight', i64(c.n_experts) * d)!
				bias:    m.load(p + 'mlp.gate.e_score_correction_bias', c.n_experts)!
				sh_gate: m.load(p + 'mlp.shared_experts.gate_proj.weight', si * d)!
				sh_up:   m.load(p + 'mlp.shared_experts.up_proj.weight', si * d)!
				sh_down: m.load(p + 'mlp.shared_experts.down_proj.weight', d * si)!
			}
		} else {
			l = Layer{
				...l
				gate: m.load(p + 'mlp.gate_proj.weight', i64(c.dense_inter) * d)!
				up:   m.load(p + 'mlp.up_proj.weight', i64(c.dense_inter) * d)!
				down: m.load(p + 'mlp.down_proj.weight', d * c.dense_inter)!
			}
		}
		m.layers << l
		m.cache << map[int]&Expert{}
	}
	return m
}

pub fn (mut m Model) close() {
	m.store.close()
}

// ---- elementwise pieces, in the engine's form ----

fn rmsnorm(mut out []f32, ooff int, x []f32, xoff int, w []f32, d int, eps f32) {
	mut ms := f64(0)
	for i in 0 .. d {
		ms += f64(x[xoff + i]) * f64(x[xoff + i])
	}
	r := f32(1.0) / C.sqrtf(f32(ms / f64(d)) + eps)
	for i in 0 .. d {
		out[ooff + i] = x[xoff + i] * r * w[i]
	}
}

fn softmax(mut x []f32, n int) {
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

fn sigmoidf(x f32) f32 {
	return f32(1.0) / (f32(1.0) + C.expf(-x))
}

fn siluf(x f32) f32 {
	return x / (f32(1.0) + C.expf(-x))
}

// rope_interleave: the first qk_rope values of v[off..], pairs (2j, 2j+1) rotated, written back as halves (j, half+j) --
// the engine's layout for the roped part of both the query and the shared key
fn rope_interleave(mut v []f32, off int, pos int, c &Cfg) {
	half := c.qk_rope / 2
	mut inp := []f32{len: c.qk_rope}
	for i in 0 .. c.qk_rope {
		inp[i] = v[off + i]
	}
	for j in 0 .. half {
		inv := C.powf(c.theta, f32(-2.0) * f32(j) / f32(c.qk_rope))
		ang := f32(pos) * inv
		cs := C.cosf(ang)
		sn := C.sinf(ang)
		a := inp[2 * j]
		b := inp[2 * j + 1]
		v[off + j] = a * cs - b * sn
		v[off + half + j] = b * cs + a * sn
	}
}

// lat_dot: the engine's score dot qabs . latent row -- on AVX2 builds eight fused-multiply-add lanes over whole groups of 8
// (c/colibri.c, "#442"), summed as hsum256 does, then the scalar tail. The lane structure fixes the bits, so it is reproduced.
fn lat_dot(qabs []f32, lat []f32, off int, kvl int, avx2 bool, blk int) f32 {
	mut a := f32(0)
	mut i := 0
	if avx2 {
		mut acc := [8]f32{}
		for i + 8 <= kvl {
			for j in 0 .. 8 {
				acc[j] = C.fmaf(qabs[i + j], lat[off + i + j], acc[j])
			}
			i += 8
		}
		// hsum256: low half + high half, then movehl, then the last pair
		l0 := acc[0] + acc[4]
		l1 := acc[1] + acc[5]
		l2 := acc[2] + acc[6]
		l3 := acc[3] + acc[7]
		a = (l0 + l2) + (l1 + l3)
	}
	return hybrid_dot(qabs, i, lat, off + i, kvl - i, a, blk)
}

// hybrid_dot: the shape gcc gives a plain `a += x[i]*y[i]` loop on x86 (found with tools/fmaprobe): whole blocks of `w` products
// multiplied apart and added one by one, the remainder as a chain of fused multiply-adds. w is 8 with AVX-512, 4 with AVX2, and 0
// (never fused) below.
fn hybrid_dot(q []f32, qoff int, k []f32, koff int, n int, init f32, w int) f32 {
	mut a := init
	mut i := 0
	if w > 0 {
		for i + w <= n {
			for j in 0 .. w {
				t := q[qoff + i + j] * k[koff + i + j]
				a = a + t
			}
			i += w
		}
	} else {
		for i < n {
			t := q[qoff + i] * k[koff + i]
			a = a + t
			i++
		}
		return a
	}
	for i < n {
		a = C.fmaf(q[qoff + i], k[koff + i], a)
		i++
	}
	return a
}

// ---- layers ----

fn (mut m Model) attention(li int, x []f32, s_n int, pos_base int) ![]f32 {
	c := &m.c
	l := &m.layers[li]
	d := c.hidden
	h_n := c.n_heads
	nope := c.qk_nope
	rr := c.qk_rope
	qh := c.qk_head
	vh := c.v_head
	kvl := c.kv_lora
	mut qr := kernels.matmul_f32(x, l.q_a, s_n, d, c.q_lora)!
	for s in 0 .. s_n {
		rmsnorm(mut qr, s * c.q_lora, qr.clone(), s * c.q_lora, l.q_a_ln, c.q_lora, c.eps)
	}
	mut q := kernels.matmul_f32(qr, l.q_b, s_n, c.q_lora, h_n * qh)!
	comp := kernels.matmul_f32(x, l.kv_a, s_n, d, kvl + rr)!
	for s in 0 .. s_n {
		pos := pos_base + s
		for h in 0 .. h_n {
			rope_interleave(mut q, (s * h_n + h) * qh + nope, pos, c)
		}
		// the latent: normed; the shared key: roped
		mut ls := []f32{len: kvl}
		rmsnorm(mut ls, 0, comp, s * (kvl + rr), l.kv_a_ln, kvl, c.eps)
		mut rs := []f32{len: rr}
		for i in 0 .. rr {
			rs[i] = comp[s * (kvl + rr) + kvl + i]
		}
		rope_interleave(mut rs, 0, pos, c)
		for i in 0 .. kvl {
			m.lat[li][pos * kvl + i] = ls[i]
		}
		for i in 0 .. rr {
			m.rop[li][pos * rr + i] = rs[i]
		}
	}
	mut ctx := []f32{len: s_n * h_n * vh}
	mut sc := []f32{len: m.max_t}
	if s_n > 4 {
		// more than four rows (a prompt): the engine does not absorb, it rebuilds every key's per-head nope part and value
		// with one matmul over the latent cache (kv_b), then scores and mixes in that space
		tk := pos_base + s_n
		kvb_dim := h_n * (nope + vh)
		kvb := kernels.matmul_f32(unsafe { m.lat[li][..tk * kvl] }, l.kv_b, tk, kvl, kvb_dim)!
		for s in 0 .. s_n {
			pos := pos_base + s
			nt := pos + 1
			for h in 0 .. h_n {
				qp := (s * h_n + h) * qh
				for jj in 0 .. nt {
					kn := jj * kvb_dim + h * (nope + vh)
					mut a := hybrid_dot(q, qp, kvb, kn, nope, 0, m.blk)
					a = hybrid_dot(q, qp + nope, m.rop[li], jj * rr, rr, a, m.blk)
					sc[jj] = a * c.attn_scale
				}
				softmax(mut sc, nt)
				for jj in 0 .. nt {
					vv := jj * kvb_dim + h * (nope + vh) + nope
					for dd in 0 .. vh {
						ctx[(s * h_n + h) * vh + dd] += sc[jj] * kvb[vv + dd]
					}
				}
			}
		}
		return kernels.matmul_f32(ctx, l.o, s_n, h_n * vh, d)
	}
	for s in 0 .. s_n {
		pos := pos_base + s
		nt := pos + 1
		for h in 0 .. h_n {
			qp := (s * h_n + h) * qh
			rbase := h * (nope + vh)
			mut qabs := []f32{len: kvl}
			for dd in 0 .. nope { // the query through kv_b's key rows
				coef := q[qp + dd]
				for i in 0 .. kvl {
					qabs[i] += coef * l.kv_b[(rbase + dd) * kvl + i]
				}
			}
			for jj in 0 .. nt {
				mut a := lat_dot(qabs, m.lat[li], jj * kvl, kvl, m.avx2, m.blk)
				a = hybrid_dot(q, qp + nope, m.rop[li], jj * rr, rr, a, m.blk)
				sc[jj] = a * c.attn_scale
			}
			softmax(mut sc, nt)
			mut clat := []f32{len: kvl}
			for jj in 0 .. nt {
				for i in 0 .. kvl {
					clat[i] += sc[jj] * m.lat[li][jj * kvl + i]
				}
			}
			for r in 0 .. vh { // the value rows of kv_b, in double as qt_matvec_rows does
				mut a := f64(0)
				for i in 0 .. kvl {
					a += f64(l.kv_b[(rbase + nope + r) * kvl + i]) * f64(clat[i])
				}
				ctx[(s * h_n + h) * vh + r] = f32(a)
			}
		}
	}
	return kernels.matmul_f32(ctx, l.o, s_n, h_n * vh, d)
}

// swiglu: down(silu(gate x) * up x) for S rows
fn swiglu(x []f32, s_n int, g []f32, u []f32, dn []f32, d int, inter int) ![]f32 {
	mut gg := kernels.matmul_f32(x, g, s_n, d, inter)!
	uu := kernels.matmul_f32(x, u, s_n, d, inter)!
	for i in 0 .. s_n * inter {
		gg[i] = siluf(gg[i]) * uu[i]
	}
	return kernels.matmul_f32(gg, dn, s_n, inter, d)
}

fn (mut m Model) expert(layer int, eid int) !&Expert {
	m.clock++
	if mut e := m.cache[layer][eid] {
		e.used = m.clock
		m.hits++
		return e
	}
	m.miss++
	c := &m.c
	p := 'model.layers.${layer}.mlp.experts.${eid}.'
	n := i64(c.moe_inter) * c.hidden
	e := &Expert{
		g:    m.load(p + 'gate_proj.weight', n)!
		u:    m.load(p + 'up_proj.weight', n)!
		d:    m.load(p + 'down_proj.weight', n)!
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

fn (mut m Model) moe(li int, x []f32, s_n int) ![]f32 {
	c := &m.c
	l := &m.layers[li]
	d := c.hidden
	e_n := c.n_experts
	logits := kernels.matmul_f32(x, l.router, s_n, d, e_n)!
	mut out := []f32{len: s_n * d}
	mut all_idx := [][]int{len: s_n}
	mut all_w := [][]f32{len: s_n}
	for s in 0 .. s_n {
		mut logit := logits[s * e_n..(s + 1) * e_n].clone()
		mut choice := []f32{len: e_n}
		for e in 0 .. e_n {
			logit[e] = sigmoidf(logit[e])
			choice[e] = logit[e] + l.bias[e]
		}
		mut idx := []int{len: c.topk}
		mut w := []f32{len: c.topk}
		for kk in 0 .. c.topk { // top-k of the biased scores, first of equals; the weight is the unbiased score
			mut best := -1
			mut bv := f32(-1e30)
			for e in 0 .. e_n {
				if choice[e] > bv {
					bv = choice[e]
					best = e
				}
			}
			if best < 0 {
				best = if kk < e_n { kk } else { 0 }
			}
			idx[kk] = best
			choice[best] = f32(-1e30)
			w[kk] = logit[best]
		}
		if c.norm_topk {
			mut sm := f32(0)
			for kk in 0 .. c.topk {
				sm += w[kk]
			}
			sm += f32(1e-20)
			for kk in 0 .. c.topk {
				w[kk] /= sm
			}
		}
		for kk in 0 .. c.topk {
			w[kk] *= c.routed
		}
		all_idx[s] = idx
		all_w[s] = w
	}
	// the batch's experts in order of first appearance (position by position, rank by rank): each is read once, computed on
	// every row that routed it, and joins that row's sum in this order -- the engine's "union order". For one row it is the
	// rank order; for a prompt it differs, and the order of the float sum is what the bits are.
	mut uniq := []int{}
	mut seen := []bool{len: e_n}
	for s in 0 .. s_n {
		for kk in 0 .. c.topk {
			e := all_idx[s][kk]
			if !seen[e] {
				seen[e] = true
				uniq << e
			}
		}
	}
	for e in uniq {
		ex := m.expert(li, e)!
		for s in 0 .. s_n {
			for kk in 0 .. c.topk {
				if all_idx[s][kk] == e {
					hh := swiglu(unsafe { x[s * d..(s + 1) * d] }, 1, ex.g, ex.u, ex.d, d, c.moe_inter)!
					for j in 0 .. d {
						out[s * d + j] += all_w[s][kk] * hh[j]
					}
				}
			}
		}
	}
	si := c.moe_inter * c.n_shared
	sh := swiglu(x, s_n, l.sh_gate, l.sh_up, l.sh_down, d, si)!
	for j in 0 .. s_n * d {
		out[j] += sh[j]
	}
	return out
}

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
		for j in 0 .. d {
			x[s * d + j] = m.embed[id * d + j]
		}
	}
	mut nrm := []f32{len: s_n * d}
	for li in 0 .. c.n_layers {
		l := &m.layers[li]
		for s in 0 .. s_n {
			rmsnorm(mut nrm, s * d, x, s * d, l.in_ln, d, c.eps)
		}
		tmp := m.attention(li, nrm, s_n, pos_base)!
		for j in 0 .. s_n * d {
			x[j] += tmp[j]
		}
		for s in 0 .. s_n {
			rmsnorm(mut nrm, s * d, x, s * d, l.post_ln, d, c.eps)
		}
		tmp2 := if l.sparse {
			m.moe(li, nrm, s_n)!
		} else {
			swiglu(nrm, s_n, l.gate, l.up, l.down, d, c.dense_inter)!
		}
		for j in 0 .. s_n * d {
			x[j] += tmp2[j]
		}
	}
	mut last := []f32{len: d}
	rmsnorm(mut last, 0, x, (s_n - 1) * d, m.final_norm, d, c.eps)
	logit := kernels.matmul_f32(last, m.lm_head, 1, d, c.vocab)!
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
	c := &m.c
	m.lat = [][]f32{len: c.n_layers, init: []f32{len: m.max_t * c.kv_lora}}
	m.rop = [][]f32{len: c.n_layers, init: []f32{len: m.max_t * c.qk_rope}}
	mut out := prompt.clone()
	mut logit := m.step(prompt, 0)!
	for s in 0 .. n_new {
		best := argmax(logit)
		out << best
		if s == n_new - 1 {
			break
		}
		logit = m.step([best], out.len - 1)!
	}
	return out
}
