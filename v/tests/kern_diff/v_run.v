// The V side of the kernel binding test: the same case file, through the V `kernels` module.
module main

import os
import math
import kernels

fn slurp(p string) []u8 {
	if p == '-' {
		return []u8{}
	}
	return os.read_bytes(p) or { panic(err) }
}

fn f32s(b []u8) []f32 {
	mut o := []f32{len: b.len / 4}
	for i in 0 .. o.len {
		o[i] = math.f32_from_bits(u32(b[4 * i]) | u32(b[4 * i + 1]) << 8 | u32(b[4 * i + 2]) << 16 | u32(b[4 * i + 3]) << 24)
	}
	return o
}

fn i8s(b []u8) []i8 {
	return b.map(i8(it))
}

fn u16s(b []u8) []u16 {
	mut o := []u16{len: b.len / 2}
	for i in 0 .. o.len {
		o[i] = u16(b[2 * i]) | u16(b[2 * i + 1]) << 8
	}
	return o
}

fn report(name string, y []f32, rc int) {
	mut h := u64(14695981039346656037)
	mut nan := 0
	for v in y {
		mut u := math.f32_bits(v)
		if v != v {
			u = 0x7fc00000
			nan++
		}
		for k in 0 .. 4 {
			h ^= u64((u >> (8 * u32(k))) & 255)
			h *= u64(1099511628211)
		}
	}
	println('${name} rc=${rc} n=${y.len} nan=${nan} ${h:016x}')
}

fn main() {
	toks := os.read_file(os.args[1]) or { panic(err) }.fields()
	mut p := 0
	for p + 10 <= toks.len {
		name := toks[p]
		s := toks[p + 1].int()
		i := toks[p + 2].int()
		o := toks[p + 3].int()
		extra := toks[p + 4].int()
		threads := toks[p + 5].int()
		f := [slurp(toks[p + 6]), slurp(toks[p + 7]), slurp(toks[p + 8]), slurp(toks[p + 9])]
		p += 10
		kernels.set_threads(threads)
		mut rc := 0
		mut y := []f32{}
		match name {
			'f32' {
				y = kernels.matmul_f32(f32s(f[0]), f32s(f[1]), s, i, o) or { panic(err) }
			}
			'dot' {
				y = [kernels.dot_f32(f32s(f[0]), f32s(f[1])) or { panic(err) }]
			}
			'q' {
				y = kernels.matmul_q(f32s(f[0]), i8s(f[1]), f32s(f[2]), s, i, o) or { panic(err) }
			}
			'i4' {
				y = kernels.matmul_i4(f32s(f[0]), f[1], f32s(f[2]), s, i, o) or { panic(err) }
			}
			'i4g' {
				y = kernels.matmul_i4_grouped(f32s(f[0]), f[1], f32s(f[2]), s, i, o, extra) or {
					panic(err)
				}
			}
			'fp8' {
				y = kernels.matmul_fp8(f32s(f[0]), f[1], f32s(f[2]), s, i, o) or { panic(err) }
			}
			'mx4' {
				y = kernels.matmul_mxfp4(f32s(f[0]), f[1], f[2], s, i, o) or { panic(err) }
			}
			'gq' {
				y = kernels.gq_matmul(f32s(f[0]), f[1], extra, s, i, o) or {
					rc = -1
					[]f32{len: s * o}
				}
			}
			'gqrow' {
				y = kernels.gq_dequant_row(extra, f[0], i) or {
					rc = -1
					[]f32{len: i}
				}
			}
			'qif32' {
				y = kernels.qi_gemm_f32(f32s(f[0]), s, f32s(f[1]), o, i, f32s(f[3])) or { panic(err) }
			}
			'qibf16' {
				y = kernels.qi_gemm_bf16(f32s(f[0]), s, u16s(f[1]), o, i, f32s(f[3])) or { panic(err) }
			}
			'qii8' {
				y = kernels.qi_gemm_i8(f32s(f[0]), s, i8s(f[1]), f32s(f[2]), o, i, f32s(f[3])) or {
					panic(err)
				}
			}
			else {
				eprintln('unknown kernel ${name}')
				exit(4)
			}
		}
		report(name, y, rc)
	}
}
