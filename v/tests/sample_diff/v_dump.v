// V side of the sampling differential test: same case file, same output lines as c_dump.c.
// usage: v_dump cases.txt tokenizer.json
module main

import os
import math
import sample
import tokenizer

fn bits2f(u u32) f32 {
	return unsafe { *(&f32(&u)) }
}

fn f2bits(f f32) u32 {
	return unsafe { *(&u32(&f)) }
}

fn d2bits(d f64) u64 {
	if math.is_nan(d) {
		return u64(0x7ff8000000000000)
	}
	return unsafe { *(&u64(&d)) }
}

fn h16(u u64) string {
	s := u.hex()
	return '0'.repeat(16 - s.len) + s
}

fn h8(u u32) string {
	s := u.hex()
	return '0'.repeat(8 - s.len) + s
}

fn main() {
	toks := os.read_file(os.args[1]) or { panic(err) }.fields()
	mut p := 0
	mut tk := &tokenizer.Tok(unsafe { nil })
	mut out := []string{}
	for p < toks.len {
		kind := toks[p]
		p++
		match kind[0] {
			`S` {
				t := u32(('0x' + toks[p]).u64())
				n := u32(('0x' + toks[p + 1]).u64())
				rng := ('0x' + toks[p + 2]).u64()
				v := toks[p + 3].int()
				steps := toks[p + 4].int()
				p += 5
				mut s := sample.Sampler{
					temp: bits2f(t)
					nuc:  bits2f(n)
					rng:  rng
				}
				mut lo := []f32{len: v}
				for _ in 0 .. steps {
					ban := toks[p].int()
					p++
					for i in 0 .. v {
						lo[i] = bits2f(u32(('0x' + toks[p]).u64()))
						p++
					}
					pk := s.pick_tok(lo, ban)
					mut h := u64(14695981039346656037)
					if s.temp > 0 {
						for i in 0 .. v {
							h ^= u64(f2bits(s.pbuf[i]))
							h *= u64(1099511628211)
						}
					}
					out << 'P ${pk} ${h16(h)} ${h16(s.rng)}'
					r := sample.logprob_row_checked(lo)
					tgt := (if ban >= 0 { ban } else { pk }) % v
					st2, lp := sample.logprob_from_row_checked(lo, tgt, &r)
					mut line := 'L ${r.status.name()} ${h8(f2bits(r.max))} ${h16(d2bits(r.logse))} ${h16(d2bits(r.logz))} ${r.argmax} ${st2.name()} ${h16(d2bits(if st2 == .finite {
						lp
					} else {
						0.0
					}))}'
					if v > 0 && math.is_finite(f64(lo[0])) {
						val, am := sample.logprob_target(lo, tgt)
						line += ' ${h16(d2bits(val))} ${int(am)}'
					}
					out << line
				}
			}
			`T` {
				eos := toks[p].int()
				sb := toks[p + 1].int() != 0
				al := toks[p + 2].int() != 0
				nc := toks[p + 3].int()
				p += 4
				mut ids := []int{}
				for _ in 0 .. nc {
					ids << toks[p].int()
					p++
				}
				if isnil(tk) {
					tk = &tokenizer.Tok{}
					unsafe {
						*tk = tokenizer.load(os.args[2]) or { panic(err) }
					}
				}
				o := sample.StopOpts{
					serve_batch: sb
					all_stops:   al
					quiet:       true
				}
				a := sample.arm(ids, eos, unsafe { nil }, o)
				out << 'A' + a.ids.map(' ${it}').join('')
				b := sample.arm(ids, eos, tk, o)
				out << 'B' + b.ids.map(' ${it}').join('')
			}
			`M` {
				s := toks[p]
				p++
				out << 'G ${int(sample.mt_is_glm(if s[0] == `-` { '' } else { s }))}'
			}
			else {}
		}
	}
	println(out.join('\n'))
}
