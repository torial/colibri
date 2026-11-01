module main

import tokenizer

fn main() {
	mut h := u64(14695981039346656037)
	for c in 0 .. 0x110000 {
		m := tokenizer.unicode_class(u32(c))
		h ^= u64(m & 255)
		h *= u64(1099511628211)
		h ^= u64(m >> 8)
		h *= u64(1099511628211)
	}
	println('classes ${h:016x}')
}
