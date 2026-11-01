module safetensors

// A port of c/st_gguf.h: a GGUF model as a safetensors index. For a folder tools/gguf_import.py wrote
// (gguf.json names the GGUF), `open_multi` opens the GGUF in place and lists its tensors under the names the
// original checkpoint gives them (the tables of st_gguf_maps.v, generated from the C header). A fused expert
// tensor [experts][rows][cols] becomes one entry per expert, each its contiguous slice of the file. Shapes are
// row-major as safetensors gives them (ggml's ne reversed). F32, F16, BF16, I32, I8 and I64 keep their
// dtypes; a block-quantized tensor (Q8_0, Q4_K, IQ2_XS, MXFP4, ...) has `ggml` set to its ggml type, which
// the float readers refuse and the C kernels of ggml_quants.h compute with. No byte is copied or converted.
// The colibri container (`colibri.layout` expert-packed-v1: each layer's routed experts in one tensor, an
// expert's gate | up | down in a 4 KiB-aligned slot) reads the same way.
import os
import cjson
import gguf

pub const ggml_base = 32

// dtype_code: st.h's integer for the dtype: the Dtype value, or ggml_base + the ggml type
pub fn (t &Tensor) dtype_code() int {
	return if t.ggml >= 0 { ggml_base + t.ggml } else { int(t.dtype) }
}

// is_ggml: a block-quantized tensor the float readers refuse
pub fn (t &Tensor) is_ggml() bool {
	return t.ggml >= 0
}

fn (t &Tensor) dtype_label() string {
	return if t.ggml >= 0 { '?' } else { t.dtype.name() }
}

fn gg_map(arch string) ?GgMap {
	for m in gg_maps() {
		if m.arch == arch {
			return m
		}
	}
	return none
}

// gguf_dtype: st_gguf_dtype, as the dtype and the ggml type (-1 when the dtype is one of st.h's own)
fn gguf_dtype(typ int) (Dtype, int) {
	match typ {
		gguf.gq_f32 { return Dtype.f32, -1 }
		gguf.gq_f16 { return Dtype.f16, -1 }
		gguf.gq_bf16 { return Dtype.bf16, -1 }
		26 { return Dtype.i32, -1 }
		24 { return Dtype.u8, -1 }
		27 { return Dtype.i64, -1 }
		else { return Dtype.u8, typ }
	}
}

// sprintf of a name with at most one %d
fn fmt_d(f string, n int) string {
	i := f.index('%d') or { return f }
	return f[..i] + n.str() + f[i + 2..]
}

struct Packed {
	types  []u32
	bytes  []u64
	stride []u64
	dims   []u64
	layers int
}

fn u32s(b []u8) []u32 {
	mut o := []u32{cap: b.len / 4}
	for i in 0 .. b.len / 4 {
		o << u32(b[4 * i]) | u32(b[4 * i + 1]) << 8 | u32(b[4 * i + 2]) << 16 | u32(b[4 * i + 3]) << 24
	}
	return o
}

fn u64s(b []u8) []u64 {
	mut o := []u64{cap: b.len / 8}
	for i in 0 .. b.len / 8 {
		mut v := u64(0)
		for k in 0 .. 8 {
			v |= u64(b[8 * i + k]) << (8 * u64(k))
		}
		o << v
	}
	return o
}

// the colibri container's layout keys; none when the GGUF is not one
fn gguf_packed(g &gguf.Ctx) !Packed {
	layout := g.get_str('colibri.layout') or { return error('') }
	if layout != 'expert-packed-v1' {
		return error("GGUF: colibri layout '${layout}' is not one this build reads")
	}
	layout_keys := error("GGUF: the colibri container's expert layout keys disagree")
	di_o := g.get('colibri.expert_dims')
	if di_o == none {
		return error('GGUF: a colibri container without colibri.expert_dims (written before this engine read containers): import the GGUF again')
	}
	ty := g.get('colibri.expert_types') or { return layout_keys }
	by := g.get('colibri.expert_bytes') or { return layout_keys }
	st := g.get('colibri.expert_stride') or { return layout_keys }
	di := di_o
	if ty.typ != gguf.t_array || by.typ != gguf.t_array || st.typ != gguf.t_array
		|| di.typ != gguf.t_array || ty.atype != gguf.t_u32 || by.atype != gguf.t_u64
		|| st.atype != gguf.t_u64 || di.atype != gguf.t_u64 || ty.n != 3 * st.n || by.n != 3 * st.n
		|| di.n != 6 * st.n {
		return error("GGUF: the colibri container's expert layout keys disagree")
	}
	return Packed{
		types:  u32s(ty.arr)
		bytes:  u64s(by.arr)
		stride: u64s(st.arr)
		dims:   u64s(di.arr)
		layers: int(st.n)
	}
}

struct GgCount {
mut:
	experts int
	renamed int
	kept    int
}

fn (mut s Store) gguf_push(name string, shard int, off i64, nbytes i64, typ int, shape []i64) {
	dt, gg := gguf_dtype(typ)
	mut numel := i64(1)
	for d in shape {
		numel *= d
	}
	s.tensors << Tensor{
		name:   name
		path:   s.shards[shard].path
		shard:  shard
		off:    off
		nbytes: nbytes
		dtype:  dt
		ggml:   gg
		numel:  numel
		shape:  shape
	}
}

// gguf_index: one GGUF's tensors into the store under the checkpoint's names. `last` is the number '@' stands
// for in a draft head's names. Returns whether the file is a colibri container.
fn (mut s Store) gguf_index(g &gguf.Ctx, map_ GgMap, path string, arch string, last int, mut c GgCount) !bool {
	if g.paths.len > max_shards {
		return error('${path}: ${g.paths.len} parts')
	}
	mut shards := []int{}
	for p in g.paths {
		shards << s.open_fd(p)!
	}
	pk_opt := gguf_packed(g) or {
		if err.msg() == '' {
			Packed{
				layers: -1
			}
		} else {
			return err
		}
	}
	packed := pk_opt.layers >= 0
	pk := pk_opt
	for t in g.t {
		mut hf := ''
		mut has_hf := false
		mut l := -1
		mut off := 0
		if packed && t.name.starts_with('blk.') {
			if pl, p := scan_int(t.name, 4) {
				if t.name[p..] == '.ffn_exps_colibri' {
					// a layer's packed experts: each kind's matrices at their place in every slot
					kinds := ['ffn_gate_exps.weight', 'ffn_up_exps.weight', 'ffn_down_exps.weight']
					if pl < 0 || pl >= pk.layers || t.n_dims != 2 || u64(t.ne[0]) != pk.stride[pl] {
						return error("GGUF ${t.name}: disagrees with the container's layout")
					}
					mut inner := u64(0)
					for k in 0 .. 3 {
						mut kh := ''
						mut found := false
						for e in map_.layer {
							if kinds[k] == e.gg {
								kh = e.hf
								found = true
								break
							}
						}
						if !found || !kh.contains('%d') {
							return error("${path}: this engine reads ${arch}'s routed experts as whole tensors, not a colibri container: import with --mode link")
						}
						b := pk.bytes[3 * pl + k]
						cols := pk.dims[6 * pl + 2 * k]
						rows := pk.dims[6 * pl + 2 * k + 1]
						rb := gguf.gq_row_bytes(int(pk.types[3 * pl + k]), i64(cols))
						if rb < 0 || u64(rb) * rows != b || inner + b > pk.stride[pl] {
							return error("GGUF ${t.name}: matrix ${k} disagrees with the container's layout")
						}
						for x in 0 .. int(t.ne[1]) {
							nm := fmt_d(map_.layer_fmt, pl) + fmt_d(kh, x)
							s.gguf_push(nm, shards[t.file], i64(t.off + u64(x) * pk.stride[pl] + inner),
								i64(b), int(pk.types[3 * pl + k]), [i64(rows), i64(cols)])
						}
						inner += b
						c.experts++
					}
					continue
				}
			}
		}
		for e in map_.globals {
			if t.name == e.gg {
				hf = e.hf
				has_hf = true
				break
			}
		}
		if !has_hf && t.name.starts_with('blk.') {
			if v, p := scan_int(t.name, 4) {
				if p < t.name.len && t.name[p] == `.` && v >= 0 {
					l = v
					off = p + 1
				}
			}
		}
		if !has_hf && l >= 0 && off > 0 {
			for e in map_.layer {
				if t.name[off..] == e.gg {
					hf = e.hf
					has_hf = true
					break
				}
			}
		}
		// "a|b": the tensor under each name (st_gg_ink_layer's ffn_gscale)
		mut rest_names := ''
		mut has_rest := false
		if has_hf && hf.contains('|') {
			alias := if hf.len > 255 { hf[..255] } else { hf }
			bar := alias.index('|') or { alias.len }
			hf = alias[..bar]
			rest_names = alias[bar + 1..]
			has_rest = true
		}
		for {
			per_expert := has_hf && hf.contains('%d')
			dims := if per_expert { 2 } else { t.n_dims }
			e_n := if per_expert { int(t.ne[2]) } else { 1 }
			if per_expert && (t.n_dims != 3 || t.nbytes % i64(e_n) != 0) {
				return error('GGUF ${t.name}: not [experts][rows][cols]')
			}
			slice := t.nbytes / i64(e_n)
			for x in 0 .. e_n {
				mut nm := ''
				if !has_hf {
					nm = t.name
				} else {
					rest := if per_expert { fmt_d(hf, x) } else { hf }
					nm = if l >= 0 { fmt_d(map_.layer_fmt, l) + rest } else { rest }
					if at := nm.index('@') {
						nm = nm[..at] + last.str() + nm[at + 1..]
					}
				}
				mut shape := []i64{}
				for d in 0 .. dims {
					shape << t.ne[dims - 1 - d]
				}
				s.gguf_push(nm, shards[t.file], i64(t.off) + slice * i64(x), slice, t.typ, shape)
			}
			if per_expert {
				c.experts++
			} else if has_hf {
				c.renamed++
			} else {
				c.kept++
			}
			if has_rest {
				bar := rest_names.index('|') or { -1 }
				if bar >= 0 {
					hf = rest_names[..bar]
					rest_names = rest_names[bar + 1..]
					has_rest = true
				} else {
					hf = rest_names
					rest_names = ''
					has_rest = false
				}
				continue
			}
			break
		}
	}
	return packed
}

// init_gguf: true when snap_dir is a gguf_import folder and the store now indexes its GGUF (and the draft
// head's, gguf.json "dspark"), false when it is not one. A broken GGUF or one of an architecture without a map is
// an error.
fn (mut s Store) init_gguf(snap_dir string) !bool {
	gj := '${snap_dir}/gguf.json'
	raw := os.read_bytes(gj) or { return false }
	buf := if raw.len > 16383 { raw[..16383] } else { raw }
	root := cjson.parse(buf.bytestr())
	gv := root.get('gguf')
	hv := root.get('dspark')
	mut g_path := ''
	if v := gv {
		if v.kind == .str {
			g_path = cstr(v.str)
		}
	}
	if g_path == '' {
		return error('${gj}: no "gguf" path')
	}
	path := gguf.path_join(snap_dir, g_path)
	mut hpath := ''
	if v := hv {
		if v.kind == .str && cstr(v.str) != '' {
			hpath = gguf.path_join(snap_dir, cstr(v.str))
		}
	}
	header_only := os.getenv_opt('GGUF_CHECK') != none
	mut meta := gguf.open(path, header_only) or { return error('GGUF: ${err.msg()}') }
	arch_o := meta.get_str('general.architecture')
	arch := arch_o or { '' }
	m := gg_map(arch) or {
		return error("${path}: GGUF architecture '${if arch_o != none {
			arch
		} else {
			'?'
		}}' has no tensor map in st_gguf.h")
	}
	mut c := GgCount{}
	packed := s.gguf_index(&meta, m, path, arch, 0, mut c)!
	s.gguf_meta = &meta
	mut head := 0
	if hpath != '' {
		mut h := gguf.open(hpath, header_only) or { return error('GGUF ${hpath}: ${err.msg()}') }
		ha_o := h.get_str('general.architecture')
		ha := ha_o or { '' }
		stages := h.get_int('dflash.block_count', 0)
		if ha_o == none || ha != 'dflash' || stages < 1 || stages > 16 {
			return error("${hpath}: not a DSpark head (architecture '${if ha_o != none {
				ha
			} else {
				'?'
			}}', ${stages} stages)")
		}
		was := s.tensors.len
		s.gguf_index(&h, gg_map('dflash') or { m }, hpath, ha, int(stages) - 1, mut c)!
		head = s.tensors.len - was
		s.gguf_head = &h
	}
	for i, t in s.tensors {
		if t.name in s.by_name {
			return error('GGUF: two tensors map to ${t.name}')
		}
		s.by_name[t.name] = i
	}
	s.gguf_active = true
	eprintln("[gguf] ${path}: ${arch}, ${meta.paths.len} part(s), ${s.tensors.len} tensors under the checkpoint's names (${c.renamed} renamed, ${c.experts} ${if packed {
		"matrices of the container's expert slots"
	} else {
		'fused expert tensors split'
	}} per expert, ${c.kept} kept as named), read in place")
	if head > 0 {
		eprintln('[gguf] the DSpark head from ${hpath}: ${head} of those tensors')
	}
	return true
}
