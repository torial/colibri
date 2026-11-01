module safetensors

// A port of the core of c/st.h: the safetensors files of a model directory, indexed by tensor name,
// with the same validation the C code does of a header that comes from a mirror nobody vouches for.
// What is ported: shard discovery (`*.safetensors`, sorted by path), the 8-byte header length and its
// bounds, the header as one checked JSON object, every tensor's dtype / shape / data_offsets and the
// overflow and bounds checks on them, the numel-times-element-size cross-check, and the float readers
// (BF16 / F16 / F32 to f32, the raw reader, F8_E8M0 scales).
// What is not (yet): multi-drive splits, replica mirrors, O_DIRECT, mmap'd experts, the
// `colibri.fmt` metadata stamps, the model.safetensors.index.json overlay rule (a name in two shards is
// an error here, as in C when there is no index), GGUF. Where C calls exit(1) this returns an error.
// tests/st_diff compares the tensor table and the bytes read, shard by shard, against the C code.
import os
import math
import cjson

const max_header = i64(512) << 20
const max_rank = 8
const max_int64 = i64(9223372036854775807)

// the codes are st.h's: 3 covers U8 and I8, 4 and 5 the fp8 types, 6 and 7 the integers read raw
pub enum Dtype {
	bf16    = 0
	f16     = 1
	f32     = 2
	u8      = 3
	f8_e4m3 = 4
	f8_e8m0 = 5
	i64     = 6
	i32     = 7
}

pub fn dtype_of(name string) !Dtype {
	return match name {
		'BF16' { Dtype.bf16 }
		'F16' { Dtype.f16 }
		'F32' { Dtype.f32 }
		'U8', 'I8' { Dtype.u8 }
		'F8_E4M3', 'F8_E4M3FN', 'float8_e4m3fn' { Dtype.f8_e4m3 }
		'F8_E8M0', 'F8_E8M0FNU' { Dtype.f8_e8m0 }
		'I64', 'U64' { Dtype.i64 }
		'I32', 'U32' { Dtype.i32 }
		else { error('unsupported dtype: ${name}') }
	}
}

// bytes per element: the one place that knows it
pub fn (d Dtype) size() int {
	return match d {
		.f32, .i32 { 4 }
		.u8, .f8_e4m3, .f8_e8m0 { 1 }
		.i64 { 8 }
		else { 2 }
	}
}

pub struct Tensor {
pub:
	name   string
	path   string // the shard that holds it
	off    i64    // absolute offset of the data in that file
	nbytes i64
	dtype  Dtype
	numel  i64
	shape  []i64
}

pub struct Store {
pub mut:
	tensors []Tensor
	by_name map[string]int
	paths   []string
}

pub fn bf16_to_f32(h u16) f32 {
	return math.f32_from_bits(u32(h) << 16)
}

pub fn f16_to_f32(h u16) f32 {
	sign := u32(h & 0x8000) << 16
	mut exp := u32((h >> 10) & 0x1F)
	mut man := u32(h & 0x3FF)
	mut u := u32(0)
	if exp == 0 {
		if man == 0 {
			u = sign
		} else {
			// subnormal: shift the mantissa up until its leading bit is the implicit one
			exp = 127 - 15 + 1
			for (man & 0x400) == 0 {
				man <<= 1
				exp--
			}
			man &= 0x3FF
			u = sign | (exp << 23) | (man << 13)
		}
	} else if exp == 0x1F {
		u = sign | 0x7F800000 | (man << 13)
	} else {
		u = sign | ((exp - 15 + 127) << 23) | (man << 13)
	}
	return math.f32_from_bits(u)
}

// 2^(v - 127); 0xff is NaN. ldexp and not the `v << 23` trick: that gives exact zero for v == 0.
pub fn ue8m0_to_f32(v u8) f32 {
	if v == 0xff {
		return math.f32_from_bits(0x7fc00000)
	}
	return f32(math.ldexp(1.0, int(v) - 127))
}

fn is_whole(x f64, limit f64) bool {
	return x == x && x - x == 0 && x >= 0.0 && x < limit && math.floor(x) == x
}

const two63 = 9223372036854775808.0

// read_shard appends the tensors of one shard file to the store
fn (mut s Store) read_shard(path string) ! {
	size := i64(os.file_size(path))
	mut f := os.open(path)!
	defer {
		f.close()
	}
	if size < 8 {
		return error('${path}: bad safetensors header length (file ${size} bytes)')
	}
	len_bytes := f.read_bytes_at(8, 0)
	if len_bytes.len < 8 {
		return error('${path}: short read of the header length')
	}
	mut hlen := u64(0)
	for i in 0 .. 8 {
		hlen |= u64(len_bytes[i]) << (8 * u64(i)) // little endian, as the C code reads a u64 on x86-64 / arm64
	}
	if hlen > u64(size - 8) || hlen > u64(max_header) {
		return error('${path}: bad safetensors header length ${hlen} (file ${size} bytes)')
	}
	hdr := f.read_bytes_at(int(hlen), 8)
	if hdr.len != int(hlen) {
		return error('${path}: short read of the header')
	}
	data_start := i64(8) + i64(hlen)
	// a partial object must not index weights; an embedded NUL is not padding
	if hdr.any(it == 0) {
		return error('${path}: safetensors header is not a JSON object')
	}
	root := cjson.parse_checked(hdr.bytestr()) or {
		return error('${path}: safetensors header is not a JSON object')
	}
	if root.kind != .obj {
		return error('${path}: safetensors header is not a JSON object')
	}
	for i, name in root.keys {
		if name == '__metadata__' {
			continue
		}
		m := root.kids[i]
		dt := m.get('dtype') or { return error('${path}: tensor `${name}` has malformed dtype/data_offsets/shape') }
		off := m.get('data_offsets') or { return error('${path}: tensor `${name}` has malformed dtype/data_offsets/shape') }
		shp := m.get('shape') or { return error('${path}: tensor `${name}` has malformed dtype/data_offsets/shape') }
		if dt.kind != .str || off.kind != .arr || off.kids.len < 2 || shp.kind != .arr
			|| shp.kids.len > max_rank {
			return error('${path}: tensor `${name}` has malformed dtype/data_offsets/shape')
		}
		oa := off.kids[0]
		ob := off.kids[1]
		if oa.kind != .num || ob.kind != .num || !is_whole(oa.num, two63) || !is_whole(ob.num, two63) {
			return error('${path}: tensor `${name}` has invalid data_offsets')
		}
		a0 := i64(oa.num)
		b0 := i64(ob.num)
		// offsets: ordered, and inside the file
		if b0 < a0 || b0 > max_int64 - data_start || data_start + b0 > size {
			return error('${path}: tensor `${name}` data_offsets [${a0},${b0}] out of file bounds (${size})')
		}
		// the shape comes from an untrusted file: a product that wraps must not pass the numel check
		mut numel := i64(1)
		mut shape := []i64{cap: shp.kids.len}
		for dim in shp.kids {
			if dim.kind != .num || !is_whole(dim.num, two63) {
				return error('${path}: tensor `${name}` shape overflows int64 - refusing (hostile or corrupt file)')
			}
			d := i64(dim.num)
			if d != 0 && numel > max_int64 / d {
				return error('${path}: tensor `${name}` shape overflows int64 - refusing (hostile or corrupt file)')
			}
			numel *= d
			shape << d
		}
		if name in s.by_name {
			return error('${path}: tensor `${name}` is also indexed from shard `${s.tensors[s.by_name[name]].path}` -- duplicate tensor name across indexed shards, refusing')
		}
		dtype := dtype_of(dt.str)!
		nbytes := b0 - a0
		esz := dtype.size()
		// float dtypes: the declared element count must match the byte span; U8/I8 are read by byte count
		if dtype != .u8 && (numel > max_int64 / i64(esz) || nbytes != numel * i64(esz)) {
			return error('${path}: tensor `${name}` numel ${numel} disagrees with byte span ${nbytes} (esz ${esz})')
		}
		s.by_name[name] = s.tensors.len
		s.tensors << Tensor{
			name:   name
			path:   path
			off:    data_start + a0
			nbytes: nbytes
			dtype:  dtype
			numel:  numel
			shape:  shape
		}
	}
}

// open indexes every `*.safetensors` file of `dir`, in path order
pub fn open(dir string) !Store {
	mut files := []string{}
	for e in os.ls(dir) or { return error('${dir}: cannot list') } {
		if e.ends_with('.safetensors') && e.len >= '.safetensors'.len {
			files << '${dir}/${e}'
		}
	}
	files.sort()
	mut s := Store{}
	for p in files {
		s.paths << p
		s.read_shard(p)!
	}
	return s
}

pub fn (s &Store) find(name string) ?&Tensor {
	i := s.by_name[name] or { return none }
	return &s.tensors[i]
}

// read_raw returns the tensor's bytes as the file holds them (any dtype)
pub fn (s &Store) read_raw(name string) ![]u8 {
	t := s.find(name) or { return error('missing tensor ${name}') }
	mut f := os.open(t.path)!
	defer {
		f.close()
	}
	b := f.read_bytes_at(int(t.nbytes), u64(t.off))
	if i64(b.len) != t.nbytes {
		return error('${name}: short read (${b.len} of ${t.nbytes} bytes)')
	}
	return b
}

// read_f32 reads a BF16, F16 or F32 tensor as f32; the other dtypes are refused by name
pub fn (s &Store) read_f32(name string) ![]f32 {
	t := s.find(name) or { return error('missing tensor ${name}') }
	if int(t.dtype) >= 3 {
		return error('${name}: tensor `${name}` is ${t.dtype} - not a float tensor; read it with read_raw')
	}
	esz := t.dtype.size()
	if t.numel < 0 || t.numel > t.nbytes / i64(esz) || t.numel * i64(esz) != t.nbytes {
		return error('${name}: tensor `${name}` shape/bytes mismatch (numel ${t.numel}, ${t.nbytes} bytes)')
	}
	raw := s.read_raw(name)!
	mut out := []f32{len: int(t.numel)}
	match t.dtype {
		.f32 {
			for i in 0 .. int(t.numel) {
				out[i] = math.f32_from_bits(u32(raw[4 * i]) | u32(raw[4 * i + 1]) << 8 | u32(raw[4 * i + 2]) << 16 | u32(raw[4 * i + 3]) << 24)
			}
		}
		.bf16 {
			for i in 0 .. int(t.numel) {
				out[i] = bf16_to_f32(u16(raw[2 * i]) | u16(raw[2 * i + 1]) << 8)
			}
		}
		else {
			for i in 0 .. int(t.numel) {
				out[i] = f16_to_f32(u16(raw[2 * i]) | u16(raw[2 * i + 1]) << 8)
			}
		}
	}
	return out
}
