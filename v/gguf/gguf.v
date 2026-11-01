module gguf

// A port of c/gguf.h: a GGUF model read in place. The header of a GGUF file (or of every part of a split
// model, NAME-00001-of-000NN.gguf) becomes a metadata table and a tensor table; tensor bytes are read straight
// from the file with pread. The header is untrusted: every count, length and offset is checked against the file
// size and against overflow before it is used, and a malformed file is refused with the C code's message.
// Differences: no Windows reader; `read_f32` (dequantizing a tensor to f32) is not ported, the ggml dequantization
// kernels stay in C (c/ggml_quants.h); `header_only` (GGUF_CHECK) is a parameter of `open`, not a global.
// tests/gguf_diff compares tensor and metadata tables, errors included, with the C code.
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>

fn C.open(path &char, flags int, mode int) int
fn C.close(fd int) int
fn C.pread(fd int, buf voidptr, n u64, off i64) i64
fn C.lseek(fd int, off i64, whence int) i64
fn C.strerror(e int) &char

// GGUF value types
pub const t_u8 = 0
pub const t_i8 = 1
pub const t_u16 = 2
pub const t_i16 = 3
pub const t_u32 = 4
pub const t_i32 = 5
pub const t_f32 = 6
pub const t_bool = 7
pub const t_string = 8
pub const t_array = 9
pub const t_u64 = 10
pub const t_i64 = 11
pub const t_f64 = 12

pub struct Kv {
pub mut:
	key   string
	typ   int
	u     u64
	i     i64
	f     f64
	s     string
	atype int  // arrays: element type
	n     u64  // arrays: element count
	arr   []u8 // numeric array elements, native (little endian) width
	strs  []string
}

pub struct Tensor {
pub mut:
	name   string
	typ    int // ggml type
	n_dims int
	ne     [4]i64 // ggml order: ne[0] is the contiguous (row) length
	file   int    // which part holds it
	off    u64    // absolute offset in that part
	nbytes i64
	used   bool
}

pub struct Ctx {
pub mut:
	fds   []int
	paths []string
	kv    []Kv
	t     []Tensor
	idx   map[string]int
	err   string
}

fn errno_str() string {
	return unsafe { cstring_to_vstring(C.strerror(C.errno)) }
}

fn clip(s string) string {
	return if s.len > 511 { s[..511] } else { s }
}

// ---- ggml type traits ----

fn r16_base_of(t int) int {
	return if t >= gq_r16_first && t < gq_r16_end { r16_base[t - gq_r16_first] } else { -1 }
}

pub fn gq_type_name(t int) string {
	if r16_base_of(t) >= 0 {
		return r16_names[t - gq_r16_first]
	}
	return if t >= 0 && t < gq_type_max && gq_types[t].name != '' { gq_types[t].name } else { '?' }
}

// gq_row_bytes: bytes of one row of n elements, or -1 when n is not a whole number of blocks
pub fn gq_row_bytes(t int, n i64) i64 {
	b := r16_base_of(t)
	if b >= 0 {
		return if n % i64(gq_types[b].blck) != 0 {
			i64(-1)
		} else {
			n / i64(gq_types[b].blck) * i64(r16_group_bytes[t - gq_r16_first] / 16)
		}
	}
	if t < 0 || t >= gq_type_max || gq_types[t].blck <= 0 {
		return -1
	}
	if n % i64(gq_types[t].blck) != 0 {
		return -1
	}
	return n / i64(gq_types[t].blck) * i64(gq_types[t].bytes)
}

// ---- header reader: a pread window over the file ----

struct Rd {
	fd   int
	size u64
mut:
	pos     u64
	bad     bool
	err     string
	buf     []u8
	buf_off u64
}

const window = 1 << 20

fn (mut r Rd) fail(msg string) {
	if !r.bad {
		r.bad = true
		r.err = clip(msg)
	}
}

// fill returns `n` bytes at the current position, or none (and a failure noted)
fn (mut r Rd) bytes(n u64) ?[]u8 {
	if r.bad {
		return none
	}
	if n > r.size - r.pos || r.pos > r.size {
		r.fail('truncated header at byte ${r.pos}')
		return none
	}
	if n == 0 {
		return []u8{}
	}
	mut out := []u8{len: int(n)}
	if n >= u64(window) {
		if !pread_all(r.fd, out.data, i64(n), i64(r.pos)) {
			r.fail('read error at byte ${r.pos}')
			return none
		}
	} else {
		if r.pos < r.buf_off || r.pos + n > r.buf_off + u64(r.buf.len) {
			want := if r.size - r.pos < u64(window) { int(r.size - r.pos) } else { window }
			r.buf = []u8{len: want}
			if !pread_all(r.fd, r.buf.data, i64(want), i64(r.pos)) {
				r.fail('read error at byte ${r.pos}')
				return none
			}
			r.buf_off = r.pos
		}
		o := int(r.pos - r.buf_off)
		for i in 0 .. int(n) {
			out[i] = r.buf[o + i]
		}
	}
	r.pos += n
	return out
}

fn pread_all(fd int, buf voidptr, n i64, off i64) bool {
	mut got := i64(0)
	for got < n {
		r := C.pread(fd, unsafe { &u8(buf) + got }, u64(n - got), off + got)
		if r < 0 {
			if C.errno == C.EINTR {
				continue
			}
			return false
		}
		if r == 0 {
			return false
		}
		got += r
	}
	return true
}

fn le(b []u8) u64 {
	mut v := u64(0)
	for i in 0 .. b.len {
		v |= u64(b[i]) << (8 * u64(i))
	}
	return v
}

fn (mut r Rd) u64() u64 {
	b := r.bytes(8) or { return 0 }
	return le(b)
}

fn (mut r Rd) u32() u32 {
	b := r.bytes(4) or { return 0 }
	return u32(le(b))
}

// str: a length-prefixed string; none on failure (r.bad set)
fn (mut r Rd) string_() ?string {
	n := r.u64()
	if r.bad {
		return none
	}
	if n > r.size - r.pos || n > (u64(1) << 30) {
		r.fail('string of ${n} bytes at ${r.pos} exceeds the file')
		return none
	}
	b := r.bytes(n) or { return none }
	return b.bytestr()
}

fn scalar_size(t int) int {
	return match t {
		t_u8, t_i8, t_bool { 1 }
		t_u16, t_i16 { 2 }
		t_u32, t_i32, t_f32 { 4 }
		t_u64, t_i64, t_f64 { 8 }
		else { 0 }
	}
}

fn scalar_decode(t int, b []u8, mut kv Kv) {
	match t {
		t_u8, t_bool {
			kv.u = u64(b[0])
			kv.i = i64(kv.u)
		}
		t_i8 {
			kv.i = i64(i8(b[0]))
		}
		t_u16 {
			kv.u = le(b)
		}
		t_i16 {
			kv.i = i64(i16(u16(le(b))))
		}
		t_u32 {
			kv.u = le(b)
		}
		t_i32 {
			kv.i = i64(i32(u32(le(b))))
		}
		t_f32 {
			kv.f = f64(f32_from(u32(le(b))))
		}
		t_u64 {
			kv.u = le(b)
		}
		t_i64 {
			kv.i = i64(le(b))
		}
		t_f64 {
			kv.f = f64_from(le(b))
		}
		else {}
	}
}

fn f32_from(u u32) f32 {
	return unsafe { *(&f32(&u)) }
}

fn f64_from(u u64) f64 {
	return unsafe { *(&f64(&u)) }
}

// ---- lookups ----

pub fn (g &Ctx) find(name string) ?&Tensor {
	i := g.idx[name] or { return none }
	return &g.t[i]
}

// mark: find and record that the caller consumed it
pub fn (mut g Ctx) mark(name string) ?&Tensor {
	i := g.idx[name] or { return none }
	g.t[i].used = true
	return &g.t[i]
}

pub fn (g &Ctx) get(key string) ?&Kv {
	for i in 0 .. g.kv.len {
		if g.kv[i].key == key {
			return &g.kv[i]
		}
	}
	return none
}

// get_int: `def` when the key is absent or of the wrong kind
pub fn (g &Ctx) get_int(key string, def i64) i64 {
	k := g.get(key) or { return def }
	return match k.typ {
		t_u8, t_u16, t_u32, t_u64, t_bool { i64(k.u) }
		t_i8, t_i16, t_i32, t_i64 { k.i }
		else { def }
	}
}

pub fn (g &Ctx) get_float(key string, def f64) f64 {
	k := g.get(key) or { return def }
	if k.typ == t_f32 || k.typ == t_f64 {
		return k.f
	}
	if k.typ == t_array || k.typ == t_string {
		return def
	}
	return f64(g.get_int(key, i64(def)))
}

pub fn (g &Ctx) get_str(key string) ?string {
	k := g.get(key) or { return none }
	if k.typ != t_string {
		return none
	}
	return k.s
}

// ---- paths ----

pub fn path_abs(p string) bool {
	if p.len == 0 {
		return false
	}
	c := p[0]
	if c == `/` || c == `\\` {
		return true
	}
	return ((c | 32) >= `a` && (c | 32) <= `z`) && p.len > 2 && p[1] == `:` && (p[2] == `\\` || p[2] == `/`)
}

pub fn path_join(dir string, p string) string {
	return if path_abs(p) { p } else { '${dir}/${p}' }
}

// ---- parsing ----

// atoi: C's, on a string that may run on after the number
fn atoi(s string) int {
	mut i := 0
	for i < s.len && (s[i] == ` ` || (s[i] >= 9 && s[i] <= 13)) {
		i++
	}
	mut neg := false
	if i < s.len && (s[i] == `-` || s[i] == `+`) {
		neg = s[i] == `-`
		i++
	}
	mut v := i64(0)
	for i < s.len && s[i] >= `0` && s[i] <= `9` {
		v = v * 10 + i64(s[i] - `0`)
		if v > 0xFFFFFFFFF {
			break
		}
		i++
	}
	return int(if neg { -v } else { v })
}

fn (mut g Ctx) parse_one(part int, path string, want_kv bool, header_only bool) bool {
	fd := C.open(&char(path.str), C.O_RDONLY, 0)
	if fd < 0 {
		g.err = clip('${path}: ${errno_str()}')
		return false
	}
	defer {
		C.close(fd)
	}
	size := C.lseek(fd, 0, C.SEEK_END)
	if size < 0 {
		g.err = clip('${path}: stat failed')
		return false
	}
	mut r := Rd{
		fd:   fd
		size: u64(size)
	}
	magic := r.bytes(4) or { []u8{} }
	if magic.len != 4 || magic.bytestr() != 'GGUF' {
		if !r.bad {
			g.err = clip('${path}: not a GGUF file')
		} else {
			g.err = r.err
		}
		return false
	}
	ver := r.u32()
	if !r.bad && (ver < 2 || ver > 3) {
		g.err = clip('${path}: GGUF version ${ver} is not supported (2 or 3)')
		return false
	}
	nt := r.u64()
	nkv := r.u64()
	if r.bad || nt > (u64(1) << 24) || nkv > (u64(1) << 20) {
		g.err = if !r.bad {
			clip('${path}: implausible header counts (${nt} tensors, ${nkv} keys)')
		} else {
			r.err
		}
		return false
	}
	mut alignment := u64(32)
	for k := u64(0); k < nkv && !r.bad; k++ {
		mut kv := Kv{}
		kv.key = r.string_() or { '' }
		kv.typ = int(r.u32())
		if r.bad {
			break
		}
		if kv.typ == t_string {
			kv.s = r.string_() or { '' }
		} else if kv.typ == t_array {
			kv.atype = int(r.u32())
			kv.n = r.u64()
			if r.bad {
				break
			}
			if kv.atype == t_string {
				if kv.n > (r.size - r.pos) / 8 {
					r.fail('${path}: string array ${kv.key} too long')
					break
				}
				for j := u64(0); j < kv.n && !r.bad; j++ {
					kv.strs << (r.string_() or { '' })
				}
			} else {
				es := scalar_size(kv.atype)
				if es == 0 || kv.n > (r.size - r.pos) / u64(es) {
					r.fail('${path}: array ${kv.key}: bad element type ${kv.atype} or length')
					break
				}
				kv.arr = r.bytes(kv.n * u64(es)) or { []u8{} }
			}
		} else {
			es := scalar_size(kv.typ)
			if es == 0 {
				r.fail('${path}: key ${kv.key} has unknown type ${kv.typ}')
				break
			}
			if b := r.bytes(u64(es)) {
				scalar_decode(kv.typ, b, mut kv)
			}
		}
		if kv.key == 'general.alignment' && kv.typ == t_u32 {
			alignment = kv.u
		}
		if want_kv {
			g.kv << kv
		}
	}
	if !r.bad && (alignment == 0 || (alignment & (alignment - 1)) != 0 || alignment > (u64(1) << 20)) {
		r.fail('${path}: bad alignment ${alignment}')
	}
	t0 := g.t.len
	for _ in u64(0) .. nt {
		if r.bad {
			break
		}
		mut t := Tensor{}
		t.name = r.string_() or { '' }
		nd := r.u32()
		if r.bad {
			break
		}
		if nd < 1 || nd > 4 {
			r.fail('${path}: tensor ${t.name} has ${nd} dims')
			break
		}
		t.n_dims = int(nd)
		mut numel := i64(1)
		for d in 0 .. 4 {
			t.ne[d] = 1
		}
		for d in u32(0) .. nd {
			v := r.u64()
			if v == 0 || v > (u64(1) << 40) || (numel > 0 && i64(v) > i64(9223372036854775807) / numel) {
				r.fail('${path}: tensor ${t.name} dimension ${d} = ${v}')
				break
			}
			t.ne[d] = i64(v)
			numel *= i64(v)
		}
		t.typ = int(r.u32())
		rel := r.u64()
		if r.bad {
			break
		}
		rb := gq_row_bytes(t.typ, t.ne[0])
		if rb < 0 {
			r.fail('${path}: tensor ${t.name}: type ${t.typ} (${gq_type_name(t.typ)}) cannot hold rows of ${t.ne[0]}')
			break
		}
		t.nbytes = rb * (numel / t.ne[0])
		t.file = part
		t.off = rel // made absolute below
		g.t << t
	}
	if !r.bad {
		data := (r.pos + alignment - 1) / alignment * alignment
		for i in t0 .. g.t.len {
			t := g.t[i]
			if t.off % alignment != 0 || (!header_only && (t.off > r.size || data > r.size - t.off
				|| u64(t.nbytes) > r.size - data - t.off)) {
				r.fail('${path}: tensor ${t.name} lies outside the file (offset ${t.off}, ${t.nbytes} bytes)')
				break
			}
			g.t[i].off += data
		}
	}
	if r.bad {
		g.err = r.err
		return false
	}
	return true
}

// close releases the descriptors
pub fn (mut g Ctx) close() {
	for fd in g.fds {
		if fd >= 0 {
			C.close(fd)
		}
	}
	g.fds = []int{}
	g.paths = []string{}
	g.kv = []Kv{}
	g.t = []Tensor{}
	g.idx = map[string]int{}
}

// is_file: 1 if path names a GGUF file (by magic)
pub fn is_file(path string) bool {
	fd := C.open(&char(path.str), C.O_RDONLY, 0)
	if fd < 0 {
		return false
	}
	defer {
		C.close(fd)
	}
	mut m := [4]u8{}
	return C.pread(fd, &m[0], 4, 0) == 4 && m[0] == `G` && m[1] == `G` && m[2] == `U` && m[3] == `F`
}

// open a GGUF model: a single file, or the first part of a split model (NAME-00001-of-000NN.gguf), in which
// case every part is opened. On failure the error is the C code's g->err. `header_only` (GGUF_CHECK): the file
// is a model's header fetched alone, so tensor data may be absent.
pub fn open(path string, header_only bool) !Ctx {
	mut g := Ctx{}
	of := path.index('-of-') or { -1 }
	mut parts := 1
	if of >= 0 && of >= 6 && path.len >= 5 && path.ends_with('.gguf') {
		num := of - 5
		a := atoi(path[num..])
		b := atoi(path[of + 4..])
		if path[num - 1] == `-` && a == 1 && b >= 1 && b <= 9999 {
			parts = b
		}
	}
	g.fds = []int{len: parts, init: -1}
	g.paths = []string{len: parts}
	for p in 0 .. parts {
		mut pp := path
		if parts > 1 {
			n := of - 5
			tmp := ((p + 1) % 100000).str()
			pp = path[..n] + '0'.repeat(5 - tmp.len) + tmp + path[n + 5..]
		}
		g.paths[p] = pp
		if !g.parse_one(p, pp, p == 0, header_only) {
			e := g.err
			g.close()
			return error(e)
		}
		g.fds[p] = C.open(&char(pp.str), C.O_RDONLY, 0)
		if g.fds[p] < 0 {
			e := clip('${pp}: ${errno_str()}')
			g.close()
			return error(e)
		}
	}
	for i in 0 .. g.t.len {
		if g.t[i].name in g.idx {
			e := clip('duplicate tensor ${g.t[i].name}')
			g.close()
			return error(e)
		}
		g.idx[g.t[i].name] = i
	}
	return g
}

// read n bytes at byte `off` inside tensor t into dst; false on failure
pub fn (g &Ctx) read(t &Tensor, dst voidptr, off i64, n i64) bool {
	if off < 0 || n < 0 || off > t.nbytes || n > t.nbytes - off {
		return false
	}
	return pread_all(g.fds[t.file], dst, n, i64(t.off) + off)
}
