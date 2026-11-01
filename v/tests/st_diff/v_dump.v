// Dump of a model directory through the V safetensors port, in the format c_dump.c prints for st.h:
// every tensor (name, dtype code, offset, bytes, numel, shape) and an FNV-1a hash of what the readers return.
module main

import os
import math
import safetensors

fn hex(s string) string {
	digits := '0123456789abcdef'
	mut o := []u8{}
	for b in s.bytes() {
		o << digits[b >> 4]
		o << digits[b & 15]
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

fn main() {
	s := safetensors.open(os.args[1]) or {
		eprintln('${err}')
		exit(1)
	}
	for t in s.tensors {
		shape := t.shape.map(it.str()).join(',')
		println('T ${hex(t.name)} ${int(t.dtype)} ${t.off} ${t.nbytes} ${t.numel} ${t.shape.len} [${shape}]')
		raw := s.read_raw(t.name) or {
			eprintln('${err}')
			exit(1)
		}
		println('W ${fnv(raw):016x}')
		if int(t.dtype) < 3 {
			f := s.read_f32(t.name) or {
				eprintln('${err}')
				exit(1)
			}
			mut bits := []u8{cap: f.len * 4}
			for x in f {
				v := math.f32_bits(x)
				bits << u8(v)
				bits << u8(v >> 8)
				bits << u8(v >> 16)
				bits << u8(v >> 24)
			}
			println('R ${fnv(bits):016x}')
		}
	}
}
