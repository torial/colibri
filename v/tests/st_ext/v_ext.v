// V side of the extended safetensors differential test: same arguments, same output as c_ext.c.
module main

import os
import math
import safetensors

fn C.posix_memalign(memptr &voidptr, alignment u64, size u64) int

fn hex(s string) string {
	digits := '0123456789abcdef'
	mut o := []u8{}
	for b in s.bytes() {
		o << digits[b >> 4]
		o << digits[b & 15]
	}
	return o.bytestr()
}

fn unhex(h string) string {
	mut o := []u8{}
	for i in 0 .. h.len / 2 {
		o << u8(('0x' + h[2 * i..2 * i + 2]).u64())
	}
	return o.bytestr()
}

fn fnv(b []u8) u64 {
	mut h := u64(14695981039346656037)
	for x in b {
		h ^= u64(x)
		h *= u64(1099511628211)
	}
	return h
}

fn fnv_ptr(p voidptr, n int) u64 {
	return fnv(unsafe { (&u8(p)).vbytes(n) })
}

fn fnvf(f []f32) u64 {
	mut b := []u8{cap: f.len * 4}
	for x in f {
		v := math.f32_bits(x)
		b << u8(v)
		b << u8(v >> 8)
		b << u8(v >> 16)
		b << u8(v >> 24)
	}
	return fnv(b)
}

fn die(msg string) {
	if msg.ends_with('\n') {
		eprint(msg)
	} else {
		eprintln(msg)
	}
	exit(1)
}

fn b2i(b bool) int {
	return if b { 1 } else { 0 }
}

fn main() {
	extra := if os.args[2] == '-' { '' } else { os.args[2] }
	mut s := safetensors.open_multi(os.args[1], extra) or {
		die('${err}')
		return
	}
	if os.getenv('ST_CHUNK') != '' {
		s.pread_chunk = os.getenv('ST_CHUNK').i64()
	}
	for t in s.tensors {
		shape := t.shape.map(it.str()).join(',')
		println('T ${hex(t.name)} ${int(t.dtype)} ${t.off} ${t.nbytes} ${t.numel} ${t.shape.len} [${shape}] ${os.file_name(t.path)} ${t.shard}')
	}
	for i in 0 .. s.fmt_names.len {
		println('F ${hex(s.fmt_names[i])} ${hex(s.fmt_vals[i])}')
	}
	tok := os.read_file(os.args[3]) or { exit(2) }.fields()
	mut p := 0
	for p < tok.len {
		op := tok[p]
		p++
		match op {
			'mirror', 'mirrorinit' {
				dir := tok[p]
				p++
				n := if op == 'mirror' { s.mirror_add(dir) } else { s.mirror_init(dir) }
				println('mirror ${n} nrep=${s.nrep()}')
			}
			'mirrorreset' {
				s.mirror_reset()
				println('mirrorreset nrep=${s.nrep()}')
			}
			'find' {
				n := unhex(tok[p])
				p++
				println('find ${s.numel(n)} ${s.nbytes(n)} ${b2i(s.has(n))}')
				println('stamp ${s.fmt_stamp(n) or { '(none)' }}')
			}
			'f32', 'scale' {
				n := unhex(tok[p])
				a := tok[p + 1].i64()
				b := tok[p + 2].int() != 0
				p += 3
				r := if op == 'f32' {
					if a < 0 {
						s.read_f32_drop(n, b) or {
							die('${err}')
							return
						}
					} else {
						s.read_f32_cap(n, a, b) or {
							die('${err}')
							return
						}
					}
				} else {
					s.read_scale_f32(n, a, b) or {
						die('${err}')
						return
					}
				}
				println('${op} ${r.len} ${fnvf(r):016x}')
			}
			'raw' {
				n := unhex(tok[p])
				a := tok[p + 1].i64()
				b := tok[p + 2].int() != 0
				p += 3
				nb := s.nbytes(n)
				r := if a < 0 {
					s.read_raw_drop(n, b) or {
						die('${err}')
						return
					}
				} else {
					s.read_raw_cap(n, a, b) or {
						die('${err}')
						return
					}
				}
				println('raw ${nb} ${fnv(r):016x}')
			}
			'slice' {
				n := unhex(tok[p])
				a := tok[p + 1].i64()
				b := tok[p + 2].i64()
				c := tok[p + 3].i64()
				d := tok[p + 4].int() != 0
				p += 5
				sz := if c > 0 { int(c) + 1 } else { 1 }
				mut o := []u8{len: sz, init: 0xAB}
				s.read_slice_raw_cap(n, a, b, voidptr(o.data), c, d) or {
					die('${err}')
					return
				}
				println('slice ${b} ${fnv(o):016x}')
			}
			'slicef' {
				n := unhex(tok[p])
				a := tok[p + 1].i64()
				b := tok[p + 2].i64()
				c := tok[p + 3].int() != 0
				p += 4
				r := s.read_slice_f32(n, a, b, c) or {
					die('${err}')
					return
				}
				println('slicef ${b} ${fnvf(r):016x}')
			}
			'range' {
				a := tok[p].int()
				b := tok[p + 1].i64()
				c := tok[p + 2].i64()
				d := tok[p + 3].i64()
				e := tok[p + 4].int() != 0
				p += 5
				sz := if d > 0 { int(d) + 1 } else { 1 }
				mut o := []u8{len: sz, init: 0xAB}
				s.read_range_raw_cap(a, b, c, if c > 0 { voidptr(o.data) } else { unsafe { nil } }, d,
					e, 'range') or {
					die('${err}')
					return
				}
				println('range ${fnv(o):016x}')
			}
			'rrep' {
				a := tok[p].int()
				rep := tok[p + 1].int()
				off := tok[p + 2].i64()
				nb := tok[p + 3].i64()
				cap := tok[p + 4].i64()
				drop := tok[p + 5].int() != 0
				direct := tok[p + 6].int() != 0
				p += 7
				pad := if off > 0 { off % safetensors.direct_align } else { 0 }
				total := ((pad + (if cap > 0 { cap } else { 0 }) + safetensors.direct_align) / safetensors.direct_align +
					1) * safetensors.direct_align
				mut buf := unsafe { nil }
				if C.posix_memalign(&buf, u64(safetensors.direct_align), u64(total)) != 0 {
					exit(5)
				}
				unsafe { C.memset(buf, 0xAB, total) }
				s.read_range_rep(a, rep, off, nb, unsafe { &u8(buf) + pad + (if tok[p - 1] == '2' {
					16
				} else {
					0
				}) }, cap, drop, direct, 'rrep') or {
					die('${err}')
					return
				}
				println('rrep ${fnv_ptr(buf, int(total)):016x}')
				unsafe { C.free(buf) }
			}
			'nrep' {
				println('nrep ${s.nrep()}')
			}
			'fdrep' {
				a := tok[p].int()
				b := tok[p + 1].int()
				p += 2
				println('fdrep ${b2i(s.fd_rep(a, b) >= 0)} ${b2i(s.direct_fd_rep(a, b) >= 0)} ${b2i(s.direct_fd(a) >= 0)}')
			}
			'prefetch' {
				n := unhex(tok[p])
				a := tok[p + 1].int()
				p += 2
				s.prefetch(n)
				s.prefetch_rep(n, a)
				println('prefetch')
			}
			'noreadahead' {
				s.no_readahead(tok[p].int())
				p++
				println('noreadahead')
			}
			'map' {
				n := unhex(tok[p])
				p++
				mut m := s.map_raw(n) or {
					println('map failed')
					continue
				}
				println('map ${m.nbytes} ${fnv_ptr(m.data, int(m.nbytes)):016x}')
				m.unmap()
			}
			'shardmap' {
				a := tok[p].int()
				b := tok[p + 1].i64()
				c := tok[p + 2].i64()
				p += 3
				if q := s.map_shard_range(a, b, c) {
					println('shardmap yes ${fnv_ptr(q, int(c)):016x}')
				} else {
					println('shardmap no ${0:016x}')
				}
			}
			'missing' {
				n := unhex(tok[p])
				p++
				s.read_raw(n) or {
					die('${err}')
					return
				}
			}
			else {
				die('unknown op ${op}')
			}
		}
	}
}
