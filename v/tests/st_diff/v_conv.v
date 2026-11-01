module main

import math
import safetensors

fn main() {
	mut h := [u64(14695981039346656037), u64(14695981039346656037), u64(14695981039346656037)]
	for i in 0 .. 65536 {
		x := math.f32_bits(safetensors.f16_to_f32(u16(i)))
		y := math.f32_bits(safetensors.bf16_to_f32(u16(i)))
		for k in 0 .. 4 {
			h[0] ^= u64((x >> (8 * u32(k))) & 255)
			h[0] *= u64(1099511628211)
			h[1] ^= u64((y >> (8 * u32(k))) & 255)
			h[1] *= u64(1099511628211)
		}
	}
	for i in 0 .. 256 {
		z := math.f32_bits(safetensors.ue8m0_to_f32(u8(i)))
		for k in 0 .. 4 {
			h[2] ^= u64((z >> (8 * u32(k))) & 255)
			h[2] *= u64(1099511628211)
		}
	}
	println('f16 ${h[0]:016x}\nbf16 ${h[1]:016x}\ne8m0 ${h[2]:016x}')
}
