module safetensors

// A port of c/st.h: the safetensors files of a model directory, indexed by tensor name, with the same
// validation the C code does of headers that come from a mirror nobody vouches for.
// Ported: shard discovery over a search path of directories (`open_multi`, "SPLIT": a shard is taken from the
// first directory that has it), the header checks, the tensor table, the model.safetensors.index.json overlay
// rule for a name held by two shards, the `colibri.fmt` metadata stamps, read replicas on other drives
// (mirrors: accepted only when size and header are byte-identical), every reader (f32, scale sidecars,
// raw, capped, tensor slices, physical ranges), the replica-aware O_DIRECT range read, prefetch and
// readahead advice, whole-tensor and per-shard read-only mappings, and the diagnostic st.h prints when a
// core tensor is missing. Where C calls exit(1) / coli_load_fail this returns an error carrying the
// same text.
// Differences: POSIX only (no Windows overlapped descriptors, no macOS F_NOCACHE twin); O_DIRECT twins
// on Linux; the second descriptor st.h pairs with a shard on Windows is not needed; the chunk size of a
// pread is a field (`pread_chunk`) rather than a compile-time macro; strings from the JSON are cut at an
// embedded NUL as C strings are. GGUF folders (ST_GGUF) are in the gguf module.
// tests/st_diff compares the tensor table and the bytes read, shard by shard, against the C code;
// tests/st_diff/run_ext.py does the same for the rest of the surface.
import os
import math
import sync
import cjson

#flag -D_GNU_SOURCE
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>

fn C.open(path &char, flags int, mode int) int
fn C.close(fd int) int
fn C.pread(fd int, buf voidptr, n u64, off i64) i64
fn C.lseek(fd int, off i64, whence int) i64
fn C.posix_fadvise(fd int, off i64, len i64, advice int) int
fn C.mmap(addr voidptr, len u64, prot int, flags int, fd int, off i64) voidptr
fn C.munmap(addr voidptr, len u64) int
fn C.sysconf(name int) i64
fn C.strerror(e int) &char

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

fn cstr(s string) string {
	nul := s.index_u8(0)
	return if nul >= 0 { s[..nul] } else { s }
}

// the readable dtype name of st_dtype_name, for refusals
pub fn (d Dtype) name() string {
	return match d {
		.bf16 { 'BF16' }
		.f16 { 'F16' }
		.f32 { 'F32' }
		.u8 { 'U8/I8' }
		.f8_e4m3 { 'F8_E4M3' }
		.f8_e8m0 { 'F8_E8M0' }
		.i64 { 'I64' }
		.i32 { 'I32' }
	}
}

const max_shards = 512
const max_mirrors = 4
const fmt_stamp_max = 4096
// the slack a caller adds behind each destination of an O_DIRECT range read
pub const direct_align = 4096

pub struct Tensor {
pub:
	name   string
	path   string // the shard that holds it
	shard  int    // its index in Store.shards
	off    i64    // absolute offset of the data in that file
	nbytes i64
	dtype  Dtype
	numel  i64
	shape  []i64
}

pub struct Shard {
pub:
	path string
	size i64
pub mut:
	fd  int
	dfd int // O_DIRECT twin, -1 if none
}

struct Replica {
mut:
	fds      []int
	dfds     []int
	accepted int
}

struct ShardMaps {
mut:
	mu   &sync.Mutex = unsafe { nil }
	base []voidptr
	len  []i64
	done []bool
}

pub struct Store {
pub mut:
	tensors     []Tensor
	by_name     map[string]int
	paths       []string
	shards      []Shard
	n_overlay   int // names present in two shards, resolved by model.safetensors.index.json
	pread_chunk i64 = i64(1) << 30
	map_experts bool // COLI_MAP_EXPERTS: serve ranges from per-shard mappings
	fmt_names   []string
	fmt_vals    []string
mut:
	replicas []Replica
	fmt_idx  map[string]int
	maps     &ShardMaps = unsafe { nil }
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

fn errno_str() string {
	return unsafe { cstring_to_vstring(C.strerror(C.errno)) }
}

fn basename(p string) string {
	i := p.last_index_u8(`/`)
	return if i >= 0 { p[i + 1..] } else { p }
}

// pread_full: chunked, EINTR-retried, honest about short reads
fn pread_full(fd int, buf voidptr, n i64, off i64, tag string, chunk i64) ! {
	mut got := i64(0)
	for got < n {
		mut want := n - got
		if want > chunk {
			want = chunk
		}
		r := C.pread(fd, unsafe { &u8(buf) + got }, u64(want), off + got)
		if r < 0 {
			if C.errno == C.EINTR {
				continue
			}
			return error('${tag}: ${errno_str()} (off ${off}, ${got}/${n} bytes)')
		}
		if r == 0 {
			return error('${tag}: short read at EOF (off ${off}, ${got}/${n} bytes) — truncated file?')
		}
		got += r
	}
}

fn (s &Store) pread_into(fd int, buf voidptr, n i64, off i64, tag string) ! {
	pread_full(fd, buf, n, off, tag, s.pread_chunk)!
}

fn advise_dontneed(fd int, off i64, n i64) {
	C.posix_fadvise(fd, off, n, C.POSIX_FADV_DONTNEED)
}

// ---- the overlay index: model.safetensors.index.json, consulted only for a name two shards both hold ----

struct IdxEntry {
	ok    bool
	shard string
}

struct OverlayIndex {
mut:
	tried  bool
	loaded bool
	map    map[string]IdxEntry
}

fn (mut ix OverlayIndex) load(dir string) {
	if ix.tried {
		return
	}
	ix.tried = true
	data := os.read_bytes('${dir}/model.safetensors.index.json') or { return }
	if data.len <= 0 || data.len > (256 << 20) {
		return
	}
	if data.any(it == 0) {
		return
	}
	root := cjson.parse_checked(data.bytestr()) or { return }
	wm := root.get('weight_map') or { return }
	if wm.kind != .obj {
		return
	}
	ix.loaded = true
	for i, k0 in wm.keys {
		k := cstr(k0)
		if k in ix.map {
			continue // the first of equal keys is the one a lookup finds
		}
		kid := wm.kids[i]
		ix.map[k] = IdxEntry{
			ok:    kid.kind == .str
			shard: if kid.kind == .str { cstr(kid.str) } else { '' }
		}
	}
}

// the shard basename the index declares for `name`; none if there is no index, no such name, or a non-string
fn (ix &OverlayIndex) shard_of(name string) ?string {
	if !ix.loaded {
		return none
	}
	e := ix.map[name] or { return none }
	if !e.ok {
		return none
	}
	return e.shard
}

// ---- opening ----

fn scan_dir(dir string, mut files []string) !int {
	if !os.exists(dir) {
		return error('${dir}: No such file or directory')
	}
	if !os.is_dir(dir) {
		return error('${dir}: Not a directory')
	}
	entries := os.ls(dir) or { return error('${dir}: Permission denied') }
	base_n := files.len
	for e in entries {
		if !e.ends_with('.safetensors') {
			continue
		}
		mut dup := false
		for f in files {
			if basename(f) == e {
				dup = true // already taken from a higher-priority drive
				break
			}
		}
		if dup {
			continue
		}
		if files.len >= max_shards {
			return error('too many shards (>${max_shards}): raise ST_MAX_SHARDS')
		}
		files << '${dir}/${e}'
	}
	return files.len - base_n
}

fn (mut s Store) open_fd(path string) !int {
	fd := C.open(&char(path.str), C.O_RDONLY, 0)
	if fd < 0 {
		return error('${path}: ${errno_str()}')
	}
	size := C.lseek(fd, 0, C.SEEK_END)
	if size < 0 {
		e := errno_str()
		C.close(fd)
		return error('fstat shard: ${e}')
	}
	mut dfd := -1
	$if linux {
		dfd = C.open(&char(path.str), C.O_RDONLY | C.O_DIRECT, 0)
	}
	s.shards << Shard{
		path: path
		size: size
		fd:   fd
		dfd:  dfd
	}
	s.paths << path
	return s.shards.len - 1
}

// open indexes every `*.safetensors` file of `dir`, in path order
pub fn open(dir string) !Store {
	return open_multi(dir, '')!
}

// open_multi: the shards of `snap_dir`, plus those of the directories in `extra_dirs` (';' or ',' separated),
// a search path: a shard present in several is taken from the first. Everything is indexed in path order.
pub fn open_multi(snap_dir string, extra_dirs string) !Store {
	mut s := Store{
		maps: &ShardMaps{
			mu: sync.new_mutex()
		}
	}
	s.map_experts = os.getenv('COLI_MAP_EXPERTS').int() != 0
	s.index(snap_dir, extra_dirs) or {
		s.close()
		return err
	}
	return s
}

fn (mut s Store) index(snap_dir string, extra_dirs string) ! {
	mut files := []string{}
	c0 := scan_dir(snap_dir, mut files)!
	if extra_dirs.len > 0 {
		mut ndir := 1
		buf := if extra_dirs.len > 4095 { extra_dirs[..4095] } else { extra_dirs }
		mut p := 0
		for p <= buf.len {
			mut sep := p
			for sep < buf.len && buf[sep] != `;` && buf[sep] != `,` {
				sep++
			}
			mut part := buf[p..sep]
			part = part.trim_left(' ').trim_right(' ')
			if part.len > 0 {
				cn := scan_dir(part, mut files)!
				eprintln('[SPLIT] +${part} -> ${cn} shard(s)')
				ndir++
			}
			p = sep + 1
		}
		eprintln('[SPLIT] model across ${ndir} dir(s): ${files.len} shard(s) total (primary ${snap_dir} -> ${c0} shard(s)), no duplication')
	}
	files.sort()
	mut ix := OverlayIndex{}
	for path in files {
		si := s.open_fd(path)!
		s.read_shard(si, snap_dir, mut ix)!
	}
	if s.n_overlay > 0 {
		eprintln('[st] ${s.n_overlay} tensor name(s) present in two shards, resolved by model.safetensors.index.json (the copy the index maps each name to is read, the other ignored)')
	}
}

// read_shard appends the tensors of shard `si` to the store
fn (mut s Store) read_shard(si int, snap_dir string, mut ix OverlayIndex) ! {
	path := s.shards[si].path
	fd := s.shards[si].fd
	size := s.shards[si].size
	mut hlen := u64(0)
	s.pread_into(fd, &hlen, 8, 0, 'pread hlen')! // little endian, as the C code reads a u64 on x86-64 / arm64
	if size < 8 || hlen > u64(size - 8) || hlen > u64(max_header) {
		return error('${path}: bad safetensors header length ${hlen} (file ${size} bytes)')
	}
	mut hdr := []u8{len: int(hlen)}
	if hlen > 0 {
		s.pread_into(fd, hdr.data, i64(hlen), 8, 'pread hdr')!
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
	s.fmt_stamp_ingest(&root, path)!
	for i, name0 in root.keys {
		name := cstr(name0)
		if name == '__metadata__' {
			continue
		}
		m := root.kids[i]
		dt := m.get('dtype') or { return error("${path}: tensor '${name}' has malformed dtype/data_offsets/shape") }
		off := m.get('data_offsets') or { return error("${path}: tensor '${name}' has malformed dtype/data_offsets/shape") }
		shp := m.get('shape') or { return error("${path}: tensor '${name}' has malformed dtype/data_offsets/shape") }
		if dt.kind != .str || off.kind != .arr || off.kids.len < 2 || shp.kind != .arr
			|| shp.kids.len > max_rank {
			return error("${path}: tensor '${name}' has malformed dtype/data_offsets/shape")
		}
		oa := off.kids[0]
		ob := off.kids[1]
		if oa.kind != .num || ob.kind != .num || !is_whole(oa.num, two63) || !is_whole(ob.num, two63) {
			return error("${path}: tensor '${name}' has invalid data_offsets")
		}
		a0 := i64(oa.num)
		b0 := i64(ob.num)
		// offsets: ordered, and inside the file
		if b0 < a0 || b0 > max_int64 - data_start || data_start + b0 > size {
			return error("${path}: tensor '${name}' data_offsets [${a0},${b0}] out of file bounds (${size})")
		}
		// the shape comes from an untrusted file: a product that wraps must not pass the numel check
		mut numel := i64(1)
		mut shape := []i64{cap: shp.kids.len}
		for dim in shp.kids {
			if dim.kind != .num || !is_whole(dim.num, two63) {
				return error("${path}: tensor '${name}' shape overflows int64 - refusing (hostile or corrupt file)")
			}
			d := i64(dim.num)
			if d != 0 && numel > max_int64 / d {
				return error("${path}: tensor '${name}' shape overflows int64 - refusing (hostile or corrupt file)")
			}
			numel *= d
			shape << d
		}
		// a name already indexed from an earlier shard: the checkpoint's own index decides which copy is read
		mut replace_idx := -1
		if name in s.by_name {
			prev := s.by_name[name]
			prev_path := s.tensors[prev].path
			ix.load(snap_dir)
			auth_o := ix.shard_of(name)
			has_auth := auth_o != none
			auth := auth_o or { '' }
			if has_auth && auth == basename(path) {
				replace_idx = prev
				s.n_overlay++
			} else if has_auth && auth == basename(prev_path) {
				s.n_overlay++
				continue
			} else {
				why := if ix.loaded {
					if has_auth {
						'model.safetensors.index.json maps it to a third shard; the index and the files disagree'
					} else {
						'model.safetensors.index.json does not list this tensor, so it cannot say which copy is authoritative'
					}
				} else {
					"no model.safetensors.index.json here to say which copy is authoritative; the checkpoint's index would resolve this"
				}
				return error('${path}: tensor \'${name}\' is also indexed from shard \'${prev_path}\' -- duplicate tensor name across indexed shards, refusing (a directory holding both would silently read one shard\'s bytes for this name while any __metadata__["colibri.fmt"] stamp for it still applies uniformly).\n  ${why}')
			}
		}
		dtype := dtype_of(cstr(dt.str))!
		nbytes := b0 - a0
		esz := dtype.size()
		// float dtypes: the declared element count must match the byte span; U8/I8 are read by byte count
		if dtype != .u8 && (numel > max_int64 / i64(esz) || nbytes != numel * i64(esz)) {
			return error("${path}: tensor '${name}' numel ${numel} disagrees with byte span ${nbytes} (esz ${esz})")
		}
		t := Tensor{
			name:   name
			path:   path
			shard:  si
			off:    data_start + a0
			nbytes: nbytes
			dtype:  dtype
			numel:  numel
			shape:  shape
		}
		if replace_idx >= 0 {
			s.tensors[replace_idx] = t // the overlay wins: same slot, new bytes
		} else {
			s.by_name[name] = s.tensors.len
			s.tensors << t
		}
	}
}

// close releases every descriptor and mapping (st_destroy)
pub fn (mut s Store) close() {
	s.mirror_reset()
	for sh in s.shards {
		if sh.fd >= 0 {
			C.close(sh.fd)
		}
		if sh.dfd >= 0 && sh.dfd != sh.fd {
			C.close(sh.dfd)
		}
	}
	if !isnil(s.maps) {
		for i, b in s.maps.base {
			if !isnil(b) {
				C.munmap(b, u64(s.maps.len[i]))
			}
		}
	}
	s.shards = []Shard{}
	s.tensors = []Tensor{}
	s.by_name = map[string]int{}
	s.fmt_names = []string{}
	s.fmt_vals = []string{}
	s.fmt_idx = map[string]int{}
}

pub fn (s &Store) find(name string) ?&Tensor {
	i := s.by_name[name] or { return none }
	return &s.tensors[i]
}

pub fn (s &Store) has(name string) bool {
	return name in s.by_name
}

// numel / nbytes: -1 for a name that is not there
pub fn (s &Store) numel(name string) i64 {
	t := s.find(name) or { return -1 }
	return t.numel
}

pub fn (s &Store) nbytes(name string) i64 {
	t := s.find(name) or { return -1 }
	return t.nbytes
}

// ---- format stamps: __metadata__["colibri.fmt"] ----

fn (mut s Store) fmt_stamp_ingest(root &cjson.Value, shard_path string) ! {
	meta := root.get('__metadata__') or { return }
	if meta.kind != .obj {
		return
	}
	stamp := meta.get('colibri.fmt') or { return }
	if stamp.kind != .str {
		return error('${shard_path}: __metadata__["colibri.fmt"] is not a JSON string -- malformed stamp, refusing (untrusted container)')
	}
	not_obj := error('${shard_path}: __metadata__["colibri.fmt"] does not parse as a JSON object -- malformed stamp, refusing (untrusted container)')
	inner := cjson.parse_checked(cstr(stamp.str)) or { return not_obj }
	if inner.kind != .obj {
		return not_obj
	}
	for i, key0 in inner.keys {
		key := cstr(key0)
		v := inner.kids[i]
		if v.kind != .str {
			return error("${shard_path}: colibri.fmt entry '${key}' is not a string -- malformed stamp, refusing (untrusted container)")
		}
		val := cstr(v.str)
		if key in s.fmt_idx {
			prev := s.fmt_vals[s.fmt_idx[key]]
			if prev == val {
				continue // an agreeing duplicate: keep one
			}
			return error('${shard_path}: __metadata__["colibri.fmt"] stamps tensor \'${key}\' as \'${val}\', but an earlier shard\'s map already stamped it \'${prev}\' -- conflicting format claims, refusing (untrusted container)')
		}
		if s.fmt_names.len >= fmt_stamp_max {
			return error('${shard_path}: __metadata__["colibri.fmt"] stamps more than ${fmt_stamp_max} tensor names across this container\'s shards -- stamps are a resident-tensor convention (docs/FORMATS.md), not a bulk migration path; a container stamping this many names is malformed, refusing (untrusted container)')
		}
		s.fmt_idx[key] = s.fmt_names.len
		s.fmt_names << key
		s.fmt_vals << val
	}
}

// fmt_stamp: the stamped format name of a tensor, none if no shard stamped it
pub fn (s &Store) fmt_stamp(name string) ?string {
	i := s.fmt_idx[name] or { return none }
	return s.fmt_vals[i]
}

// ---- mirrors: read-only copies of the model on other drives ----

pub fn (s &Store) nrep() int {
	return s.replicas.len
}

// fd of replica `rep` (0 = the primary) for shard `shard`; -1 if absent
pub fn (s &Store) fd_rep(shard int, rep int) int {
	if shard < 0 || shard >= s.shards.len {
		return -1
	}
	if rep == 0 {
		return s.shards[shard].fd
	}
	if rep > s.replicas.len {
		return -1
	}
	return s.replicas[rep - 1].fds[shard]
}

pub fn (s &Store) direct_fd(shard int) int {
	if shard < 0 || shard >= s.shards.len {
		return -1
	}
	return s.shards[shard].dfd
}

pub fn (s &Store) direct_fd_rep(shard int, rep int) int {
	if rep == 0 {
		return s.direct_fd(shard)
	}
	if shard < 0 || shard >= s.shards.len || rep > s.replicas.len {
		return -1
	}
	return s.replicas[rep - 1].dfds[shard]
}

// no_readahead: tell the kernel a shard is read at random, on every drive that holds it
pub fn (s &Store) no_readahead(shard int) {
	for r in 0 .. s.replicas.len + 1 {
		f := s.fd_rep(shard, r)
		if f >= 0 {
			C.posix_fadvise(f, 0, 0, C.POSIX_FADV_RANDOM)
		}
	}
}

pub fn (mut s Store) mirror_reset() {
	for r in s.replicas {
		for fd in r.fds {
			if fd >= 0 {
				C.close(fd)
			}
		}
		for fd in r.dfds {
			if fd >= 0 {
				C.close(fd)
			}
		}
	}
	s.replicas = []Replica{}
}

// mirror_add registers <dir>/<basename> as the next read replica of every indexed shard. A file is accepted
// only if its size and safetensors header are byte-identical to the primary's, so every data_offsets is valid
// on any copy; missing or divergent files stay on the primary. Returns the number of files accepted; a
// directory that contributes none claims no replica slot.
pub fn (mut s Store) mirror_add(dir string) int {
	if s.replicas.len >= max_mirrors {
		return 0
	}
	n := s.shards.len
	mut r := Replica{
		fds:  []int{len: n, init: -1}
		dfds: []int{len: n, init: -1}
	}
	for i in 0 .. n {
		mp := '${dir}/${basename(s.shards[i].path)}'
		mfd := C.open(&char(mp.str), C.O_RDONLY, 0)
		if mfd < 0 {
			continue // a partial mirror: this shard stays on the primary
		}
		sza := C.lseek(s.shards[i].fd, 0, C.SEEK_END)
		szb := C.lseek(mfd, 0, C.SEEK_END)
		if sza != szb {
			eprintln('[MIRROR] ${mp}: size differs from the primary copy — file skipped')
			C.close(mfd)
			continue
		}
		mut ha := u64(0)
		mut hb := u64(0)
		mut ok := true // an identical header means identical data_offsets
		if C.pread(s.shards[i].fd, &ha, 8, 0) != 8 || C.pread(mfd, &hb, 8, 0) != 8 || ha != hb
			|| ha == 0 || ha > u64(256) << 20 || i64(8 + ha) > sza {
			ok = false
		}
		if ok {
			mut ba := []u8{len: int(ha)}
			mut bb := []u8{len: int(ha)}
			if C.pread(s.shards[i].fd, ba.data, ha, 8) != i64(ha) || C.pread(mfd, bb.data, ha, 8) != i64(ha)
				|| ba != bb {
				ok = false
			}
		}
		if !ok {
			eprintln('[MIRROR] ${mp}: header differs from the primary copy — file skipped')
			C.close(mfd)
			continue
		}
		r.fds[i] = mfd
		$if linux {
			r.dfds[i] = C.open(&char(mp.str), C.O_RDONLY | C.O_DIRECT, 0)
		}
		r.accepted++
	}
	if r.accepted > 0 {
		s.replicas << r
		return r.accepted
	}
	for fd in r.fds {
		if fd >= 0 {
			C.close(fd)
		}
	}
	return 0
}

// the single-mirror entry point: drop every replica, add this one
pub fn (mut s Store) mirror_init(dir string) int {
	s.mirror_reset()
	return s.mirror_add(dir)
}

// ---- readers ----

fn (s &Store) tensor(name string) !&Tensor {
	return s.find(name) or { error(s.missing_report(name)) }
}

fn (s &Store) fd_of(t &Tensor) int {
	return s.shards[t.shard].fd
}

pub fn (s &Store) prefetch(name string) {
	t := s.find(name) or { return }
	C.posix_fadvise(s.fd_of(t), t.off, t.nbytes, C.POSIX_FADV_WILLNEED)
}

// prefetch on replica `rep`'s drive: the WILLNEED must warm the cache of the fd the later pread hits
pub fn (s &Store) prefetch_rep(name string, rep int) {
	t := s.find(name) or { return }
	mut fd := s.fd_rep(t.shard, rep)
	if fd < 0 {
		fd = s.fd_of(t)
	}
	C.posix_fadvise(fd, t.off, t.nbytes, C.POSIX_FADV_WILLNEED)
}

fn convert_f32(dtype Dtype, raw []u8, n i64) []f32 {
	mut out := []f32{len: int(n)}
	match dtype {
		.f32 {
			for i in 0 .. int(n) {
				out[i] = math.f32_from_bits(u32(raw[4 * i]) | u32(raw[4 * i + 1]) << 8 | u32(raw[4 * i + 2]) << 16 | u32(raw[4 * i + 3]) << 24)
			}
		}
		.bf16 {
			for i in 0 .. int(n) {
				out[i] = bf16_to_f32(u16(raw[2 * i]) | u16(raw[2 * i + 1]) << 8)
			}
		}
		else {
			for i in 0 .. int(n) {
				out[i] = f16_to_f32(u16(raw[2 * i]) | u16(raw[2 * i + 1]) << 8)
			}
		}
	}
	return out
}

// read_f32 reads a BF16, F16 or F32 tensor as f32; the other dtypes are refused by name
pub fn (s &Store) read_f32(name string) ![]f32 {
	return s.read_f32_drop(name, false)
}

// read_f32_drop: drop = advise the kernel to discard the pages afterwards (experts in streaming)
pub fn (s &Store) read_f32_drop(name string, drop bool) ![]f32 {
	t := s.tensor(name)!
	if int(t.dtype) >= 3 {
		return error("${name}: tensor '${name}' is ${t.dtype.name()} — not a float tensor; read it with st_read_raw")
	}
	esz := t.dtype.size()
	if t.numel < 0 || t.numel > t.nbytes / i64(esz) || t.numel * i64(esz) != t.nbytes {
		return error("${name}: tensor '${name}' shape/bytes mismatch (numel ${t.numel}, ${t.nbytes} bytes, dtype ${int(t.dtype)}) — refusing (hostile or corrupt file)")
	}
	mut raw := []u8{len: int(t.nbytes)}
	if t.nbytes > 0 {
		s.pread_into(s.fd_of(t), raw.data, t.nbytes, t.off, 'pread data')!
	}
	out := convert_f32(t.dtype, raw, t.numel)
	if drop {
		advise_dontneed(s.fd_of(t), t.off, t.nbytes)
	}
	return out
}

// read_f32_cap refuses a tensor of more than `cap` elements: for destinations sized from config dims
pub fn (s &Store) read_f32_cap(name string, cap i64, drop bool) ![]f32 {
	t := s.tensor(name)!
	if t.numel < 0 || t.numel > cap {
		return error('tensor ${name}: numel ${t.numel} exceeds destination capacity ${cap}')
	}
	return s.read_f32_drop(name, drop)
}

// read_scale_f32: a scale sidecar as f32, from F32, BF16, F16 or F8_E8M0 (one byte, 2^(v-127))
pub fn (s &Store) read_scale_f32(name string, cap i64, drop bool) ![]f32 {
	t := s.tensor(name)!
	if t.numel < 0 || t.numel > cap {
		return error('scale ${name}: numel ${t.numel} exceeds destination capacity ${cap}')
	}
	if t.dtype == .f32 || t.dtype == .bf16 || t.dtype == .f16 {
		return s.read_f32_drop(name, drop)
	}
	if t.dtype != .f8_e8m0 {
		return error('scale ${name}: dtype ${t.dtype.name()} is neither F32 nor F8_E8M0')
	}
	if t.nbytes != t.numel {
		return error('scale ${name}: F8_E8M0 numel ${t.numel} disagrees with ${t.nbytes} bytes')
	}
	mut raw := []u8{len: int(t.nbytes)}
	if t.nbytes > 0 {
		s.pread_into(s.fd_of(t), raw.data, t.nbytes, t.off, 'pread ue8m0 scale')!
	}
	mut out := []f32{len: int(t.numel)}
	for i in 0 .. int(t.numel) {
		out[i] = ue8m0_to_f32(raw[i])
	}
	if drop {
		advise_dontneed(s.fd_of(t), t.off, t.nbytes)
	}
	return out
}

// read_raw returns the tensor's bytes as the file holds them (any dtype)
pub fn (s &Store) read_raw(name string) ![]u8 {
	return s.read_raw_drop(name, false)
}

pub fn (s &Store) read_raw_drop(name string, drop bool) ![]u8 {
	t := s.tensor(name)!
	mut b := []u8{len: int(t.nbytes)}
	if t.nbytes > 0 {
		s.pread_into(s.fd_of(t), b.data, t.nbytes, t.off, 'pread raw')!
	}
	if drop {
		advise_dontneed(s.fd_of(t), t.off, t.nbytes)
	}
	return b
}

// read_raw_cap refuses a tensor that declares more than `cap` bytes
pub fn (s &Store) read_raw_cap(name string, cap i64, drop bool) ![]u8 {
	t := s.tensor(name)!
	if t.nbytes < 0 || t.nbytes > cap {
		return error('${name}: tensor declares ${t.nbytes} bytes, destination holds ${cap} — refusing (untrusted container)')
	}
	return s.read_raw_drop(name, drop)
}

// read_range_raw_cap reads an exact physical range of one indexed shard into `out` (capacity `cap`)
pub fn (s &Store) read_range_raw_cap(shard int, off i64, nbytes i64, out voidptr, cap i64, drop bool, tag string) ! {
	if shard < 0 || shard >= s.shards.len {
		return error('physical range uses an unindexed shard fd')
	}
	size := s.shards[shard].size
	if off < 0 || nbytes < 0 || cap < 0 || nbytes > cap || off > size || nbytes > size - off {
		return error('physical shard range [${off},+${nbytes}) exceeds file/destination bounds (file ${size}, cap ${cap}) — refusing (untrusted container)')
	}
	if nbytes > 0 && isnil(out) {
		return error('physical range has a NULL destination')
	}
	fd := s.shards[shard].fd
	if nbytes > 0 {
		s.pread_into(fd, out, nbytes, off, if tag.len > 0 { tag } else { 'pread physical range' })!
	}
	if drop && nbytes > 0 {
		advise_dontneed(fd, off, nbytes)
	}
}

fn pread_aligned_try(fd int, buf voidptr, n i64, off i64) bool {
	for {
		r := C.pread(fd, buf, u64(n), off)
		if r == n {
			return true
		}
		if r < 0 && C.errno == C.EINTR {
			continue
		}
		return false
	}
	return false
}

// read_range_rep: the range served by replica `rep` (0 = primary), through its O_DIRECT twin when `direct`
// and the caller reserved `off % direct_align` bytes of scratch behind `out` with `out - pad` block-aligned;
// otherwise buffered. A partial mirror or a missing twin silently takes the buffered path.
pub fn (s &Store) read_range_rep(shard int, rep int, off i64, nbytes i64, out voidptr, cap i64, drop bool, direct bool, tag string) ! {
	if shard < 0 || shard >= s.shards.len {
		return error('replica range uses an unindexed shard fd')
	}
	size := s.shards[shard].size
	if off < 0 || nbytes < 0 || cap < 0 || nbytes > cap || off > size || nbytes > size - off {
		return error('replica range [${off},+${nbytes}) exceeds file/destination bounds (file ${size}, cap ${cap}) — refusing (untrusted container)')
	}
	if nbytes == 0 {
		return
	}
	if isnil(out) {
		return error('replica range has a NULL destination')
	}
	label := if tag.len > 0 { tag } else { 'pread replica range' }
	mut r := rep
	mut rfd := s.fd_rep(shard, r)
	if rfd < 0 {
		rfd = s.shards[shard].fd
		r = 0
	}
	pad := off % direct_align
	window := unsafe { &u8(out) - pad }
	dfd := if direct { s.direct_fd_rep(shard, r) } else { -1 }
	if dfd >= 0 && (u64(window) & u64(direct_align - 1)) == 0 && pad + nbytes <= s.pread_chunk {
		want := pad + nbytes
		left := size - (off - pad)
		mut bulk := want & ~i64(direct_align - 1)
		if bulk > left {
			bulk = left & ~i64(direct_align - 1)
		}
		if bulk > 0 && pread_aligned_try(dfd, window, bulk, off - pad) {
			tail := want - bulk
			if tail > 0 {
				s.pread_into(rfd, unsafe { window + bulk }, tail, off - pad + bulk, label)!
			}
			return
		}
	}
	s.pread_into(rfd, out, nbytes, off, label)!
	if drop {
		advise_dontneed(rfd, off, nbytes)
	}
}

// read_slice_raw_cap reads an exact byte slice of a tensor (offsets relative to the tensor) into `out`
pub fn (s &Store) read_slice_raw_cap(name string, byte_off i64, nbytes i64, out voidptr, cap i64, drop bool) ! {
	t := s.tensor(name)!
	if byte_off < 0 || nbytes < 0 {
		return error('slice ${name} [${byte_off},+${nbytes}) has negative offset/length')
	}
	if cap < 0 || nbytes > cap {
		return error('slice ${name} requests ${nbytes} bytes, destination holds ${cap} — refusing')
	}
	if t.nbytes < 0 || byte_off > t.nbytes || nbytes > t.nbytes - byte_off {
		return error('slice ${name} [${byte_off},+${nbytes}) out of tensor byte bounds (${t.nbytes})')
	}
	if nbytes > 0 && isnil(out) {
		return error('slice ${name} has a NULL destination for ${nbytes} bytes')
	}
	if t.off < 0 || byte_off > max_int64 - t.off {
		return error('slice ${name} file offset overflows int64')
	}
	boff := t.off + byte_off
	if nbytes > 0 {
		s.pread_into(s.fd_of(t), out, nbytes, boff, 'pread raw slice')!
		if drop {
			advise_dontneed(s.fd_of(t), boff, nbytes)
		}
	}
}

// read_slice returns the bytes of a tensor slice
pub fn (s &Store) read_slice(name string, byte_off i64, nbytes i64, drop bool) ![]u8 {
	if nbytes < 0 || nbytes > i64(1) << 40 {
		return error('slice ${name} requests ${nbytes} bytes')
	}
	mut b := []u8{len: int(nbytes)}
	s.read_slice_raw_cap(name, byte_off, nbytes, b.data, nbytes, drop)!
	return b
}

// read_slice_f32: n_elems elements from element elem_off of a float tensor
pub fn (s &Store) read_slice_f32(name string, elem_off i64, n_elems i64, drop bool) ![]f32 {
	t := s.tensor(name)!
	if int(t.dtype) >= 3 {
		return error('slice ${name}: tensor is ${t.dtype.name()} — not a float tensor')
	}
	esz := i64(t.dtype.size())
	if elem_off < 0 || n_elems < 0 || elem_off > t.numel || n_elems > t.numel - elem_off {
		return error('slice ${name} [${elem_off},+${n_elems}) out of tensor bounds (numel ${t.numel})')
	}
	if elem_off > max_int64 / esz || n_elems > max_int64 / esz || t.off < 0
		|| elem_off * esz > max_int64 - t.off {
		return error('slice ${name} byte arithmetic/destination is invalid — refusing')
	}
	boff := t.off + elem_off * esz
	nb := n_elems * esz
	mut raw := []u8{len: int(nb)}
	if nb > 0 {
		s.pread_into(s.fd_of(t), raw.data, nb, boff, 'pread slice')!
	}
	out := convert_f32(t.dtype, raw, n_elems)
	if drop && nb > 0 {
		advise_dontneed(s.fd_of(t), boff, nb)
	}
	return out
}

// ---- read-only mappings ----

pub struct MappedRaw {
pub mut:
	base   voidptr
	length u64
pub:
	data   voidptr
	nbytes i64
}

// map_readonly maps [off, off+len) of fd; the mapping starts at the enclosing page, `data` at the byte asked
fn map_readonly(fd int, off i64, length u64) !(voidptr, u64, voidptr) {
	if off < 0 || length == 0 {
		return error('invalid argument')
	}
	mut pg := C.sysconf(C._SC_PAGESIZE)
	if pg <= 0 {
		pg = 4096
	}
	aligned := u64(off) - (u64(off) % u64(pg))
	delta := u64(off) - aligned
	if length > max_u64 - delta {
		return error('value too large')
	}
	view := delta + length
	base := C.mmap(unsafe { nil }, view, C.PROT_READ, C.MAP_SHARED, fd, i64(aligned))
	if u64(base) == max_u64 {
		return error(errno_str())
	}
	return base, view, unsafe { &u8(base) + delta }
}

// map_raw: a read-only view of one tensor's stored bytes, no copy
pub fn (s &Store) map_raw(name string) !MappedRaw {
	t := s.find(name) or { return error('missing tensor: ${name}') }
	if t.nbytes <= 0 {
		return error('invalid argument')
	}
	base, view, data := map_readonly(s.fd_of(t), t.off, u64(t.nbytes))!
	return MappedRaw{
		base:   base
		length: view
		data:   data
		nbytes: t.nbytes
	}
}

pub fn (mut m MappedRaw) unmap() {
	if !isnil(m.base) {
		C.munmap(m.base, m.length)
		m.base = unsafe { nil }
	}
}

// shard_mapped: the persistent whole-file mapping of a shard (COLI_MAP_EXPERTS=1), mapped once; none means
// "use pread" in every case: disabled, failed, or empty
pub fn (s &Store) shard_mapped(shard int) ?voidptr {
	if shard < 0 || shard >= s.shards.len || !s.map_experts || isnil(s.maps) {
		return none
	}
	mut m := unsafe { s.maps }
	m.mu.lock()
	defer {
		m.mu.unlock()
	}
	if m.base.len == 0 {
		m.base = []voidptr{len: s.shards.len, init: unsafe { nil }}
		m.len = []i64{len: s.shards.len}
		m.done = []bool{len: s.shards.len}
	}
	if m.done[shard] {
		b := m.base[shard]
		return if isnil(b) { none } else { b }
	}
	m.done[shard] = true
	length := C.lseek(s.shards[shard].fd, 0, C.SEEK_END)
	if length <= 0 {
		return none
	}
	_, _, data := map_readonly(s.shards[shard].fd, 0, u64(length)) or { return none }
	m.len[shard] = length
	m.base[shard] = data
	return data
}

// map_shard_range: [off, off+nbytes) of a shard served out of its mapping, none if it cannot be
pub fn (s &Store) map_shard_range(shard int, off i64, nbytes i64) ?voidptr {
	if off < 0 || nbytes <= 0 {
		return none
	}
	base := s.shard_mapped(shard) or { return none }
	if off > s.maps.len[shard] - nbytes {
		return none
	}
	return unsafe { &u8(base) + off }
}

// ---- the report for a missing tensor (st_die_missing) ----

fn is_space(c u8) bool {
	return c == ` ` || (c >= 9 && c <= 13)
}

// scan_int: sscanf's %d from position p: leading white space, an optional sign, at least one digit
fn scan_int(s string, p int) ?(int, int) {
	mut i := p
	for i < s.len && is_space(s[i]) {
		i++
	}
	mut neg := false
	if i < s.len && (s[i] == `-` || s[i] == `+`) {
		neg = s[i] == `-`
		i++
	}
	st := i
	mut v := i64(0)
	for i < s.len && s[i] >= `0` && s[i] <= `9` {
		v = v * 10 + i64(s[i] - `0`)
		i++
	}
	if i == st {
		return none
	}
	return int(if neg { -v } else { v }), i
}

// sscanf(b, "model-%d-of-%d")
fn scan_model_of(b string) ?(int, int) {
	if !b.starts_with('model-') {
		return none
	}
	idx, p := scan_int(b, 6) or { return none }
	if !b[p..].starts_with('-of-') {
		return none
	}
	tot, _ := scan_int(b, p + 4) or { return none }
	return idx, tot
}

// sscanf(b, "out-%d")
fn scan_out(b string) ?int {
	if !b.starts_with('out-') {
		return none
	}
	idx, _ := scan_int(b, 4) or { return none }
	return idx
}

fn pad5(n int) string {
	s := n.str()
	return if s.len < 5 { '0'.repeat(5 - s.len) + s } else { s }
}

// missing_report: what st_die_missing prints for a core tensor that is not indexed, as one string
pub fn (s &Store) missing_report(name string) string {
	mut o := 'missing ${name}\n\n'
	if s.shards.len == 0 || s.tensors.len == 0 {
		o += '  No safetensors tensors were indexed at all. --model must point AT the\n  container directory -- the one holding the *.safetensors shards.\n'
		return o
	}
	mut lo := -1
	mut hi := -1
	mut declared := 0
	mut numbered := 0
	for sh in s.shards {
		b := basename(sh.path)
		mut idx := 0
		if i, tot := scan_model_of(b) {
			idx = i
			declared = tot
		} else if i := scan_out(b) {
			idx = i
		} else {
			continue // out-mtp-* etc: not numbered
		}
		numbered++
		if lo < 0 || idx < lo {
			lo = idx
		}
		if idx > hi {
			hi = idx
		}
	}
	complete := declared > 0 && numbered >= declared
	o += '  indexed ${s.tensors.len} tensors from ${s.shards.len} shard file(s)\n'
	if declared > 0 && numbered < declared {
		o += '  shard filenames declare ${declared} shards, but only ${numbered} are here -- ${declared - numbered} MISSING\n'
	} else if declared > 0 {
		o += '  shard filenames declare ${declared} shards and ${numbered} are here\n'
	} else if numbered > 0 {
		o += '  shards numbered ${pad5(lo)}..${pad5(hi)}, ${numbered} file(s)${if numbered == hi - lo + 1 {
			' (contiguous: a gap would be at the tail)'
		} else {
			' (GAPS in the numbering)'
		}}\n'
	}
	// the checkpoint's own index says more than a file count can
	p0 := s.shards[0].path
	slash := p0.last_index_u8(`/`)
	dir := if slash >= 0 { p0[..slash] } else { '.' }
	mut ix := OverlayIndex{}
	ix.load(dir)
	if ix.loaded {
		if auth := ix.shard_of(name) {
			mut present := false
			for sh in s.shards {
				if basename(sh.path) == auth {
					present = true
					break
				}
			}
			if !present {
				o += "\n  model.safetensors.index.json maps '${name}' to ${auth}, which is not in this\n  directory: that shard is MISSING. Fetch just that file:\n      hf download <repo> ${auth} --local-dir <model-dir>\n"
			} else {
				o += "\n  model.safetensors.index.json maps '${name}' to ${auth}, which is here but whose\n  header does not declare it: the file is not the one the index describes (an\n  interrupted transfer that kept the name, or a shard from another revision of\n  the repo). Re-download that one shard and compare its size with the repo's.\n"
			}
		} else {
			o += "\n  model.safetensors.index.json does not list '${name}': this checkpoint never had a\n  tensor by that name. The engine running here expects one, so it is not this\n  checkpoint's engine. coli picks the engine from config.json -- run\n  `coli info --model <dir>` and check the family it names; without a config.json\n  there is no family, and the files to copy next to the shards are config.json,\n  tokenizer.json and model.safetensors.index.json from the model repo.\n"
		}
		return o
	}
	// follow the evidence: telling someone who already has every declared shard to re-download helps nobody
	if complete {
		o += "\n  '${name}' is a core tensor the engine cannot run without, and every shard the\n  filenames declare is present -- so this is NOT the usual truncated download.\n  Either the container was built without this tensor, or the engine is looking\n  for the wrong name. Please report it with this output.\n"
		return o
	}
	if name.contains('model.norm') || name.contains('lm_head') {
		o += "\n  '${name}' is written into one of the LAST shards, so a container whose tail\n  never arrived indexes almost everything and then fails exactly here.\n"
	} else {
		o += "\n  '${name}' is a core tensor: the engine cannot run without it.\n"
	}
	o += '  An incomplete transfer is far more likely than a corrupt engine. Re-running the\n  download resumes only the missing shards; the HF xet backend is known to stall\n  mid-transfer (#452), so disable it:\n      HF_HUB_DISABLE_XET=1 hf download <repo> --local-dir <model-dir>\n  If your shard count is already complete, that IS a bug worth reporting.\n'
	return o
}
