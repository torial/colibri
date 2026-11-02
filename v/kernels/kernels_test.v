module kernels

import os
import gguf

fn close(a f32, b f32, tol f32) bool {
	d := if a > b { a - b } else { b - a }
	return d <= tol * (1 + if b < 0 { -b } else { b })
}

fn test_reports_what_it_was_built_for() {
	s := isa()
	assert s.contains('openmp')
	set_threads(2)
	assert threads() == 2
	set_threads(4)
}

fn test_f32_and_dot() {
	y := matmul_f32([f32(1), 2, 3], [f32(1), 1, 1, 0, 1, 0], 1, 3, 2)!
	assert y == [f32(6), 2]
	assert dot_f32([f32(1), 2], [f32(3), 4])! == 11
}

fn test_int8_scale_per_row() {
	y := matmul_q([f32(1), 2, 3], [i8(1), -1, 2, 0, 1, 0], [f32(0.5), 2], 1, 3, 2)!
	assert y == [f32(2.5), 4]
}

fn test_int4_nibbles() {
	// row [1,-1,0,2] as nibbles 9,7,8,10: low nibble is the even column
	y := matmul_i4([f32(1), 2, 3, 4], [u8(0x79), 0xA8], [f32(0.5)], 1, 4, 1)!
	assert y == [f32(3.5)]
	// the same row, one scale per group of 2 columns
	g := matmul_i4_grouped([f32(1), 2, 3, 4], [u8(0x79), 0xA8], [f32(1), 2], 1, 4, 1, 2)!
	assert g == [f32(-1 + 2 * 8)] // (1*1 + 2*-1)*1 + (3*0 + 4*2)*2 = -1 + 16
}

fn test_fp8_and_mxfp4() {
	// e4m3 0x38 is 1.0; one 128x128 block scale of 2
	y := matmul_fp8([f32(1), 2, 3], [u8(0x38), 0x38, 0x38], [f32(2)], 1, 3, 1)!
	assert y == [f32(12)]
	// e2m1 code 2 is 1.0; ue8m0 exponent 127 is 2^0
	mut q := []u8{len: 16, init: 0x22}
	m := matmul_mxfp4([]f32{len: 32, init: 1}, q, [u8(127)], 1, 32, 1)!
	assert m == [f32(32)]
	q[0] = 0x2A // column 0 becomes code 10: -1.0
	assert matmul_mxfp4([]f32{len: 32, init: 1}, q, [u8(127)], 1, 32, 1)![0] == 30
}

fn q8_0_block(d_f16 u16, q []i8) []u8 {
	mut b := [u8(d_f16), u8(d_f16 >> 8)]
	for v in q {
		b << u8(v)
	}
	return b
}

fn test_ggml_rows_decode_exactly() {
	mut qs := []i8{}
	for i in 0 .. 32 {
		qs << i8(i - 10)
	}
	row := gq_dequant_row(gguf.gq_q8_0, q8_0_block(0x3800, qs), 32)! // d = 0.5
	for i in 0 .. 32 {
		assert row[i] == 0.5 * f32(i - 10)
	}
	// Q4_0: d = 1.0; elements 0..15 are the low nibbles, 16..31 the high ones
	mut b := [u8(0x00), 0x3C]
	for j in 0 .. 16 {
		b << u8(j | (((j + 1) & 15) << 4))
	}
	r4 := gq_dequant_row(gguf.gq_q4_0, b, 32)!
	for j in 0 .. 16 {
		assert r4[j] == f32(j - 8)
		assert r4[16 + j] == f32(((j + 1) & 15) - 8)
	}
	assert gq_row_bytes(gguf.gq_q8_0, 64) == 68
}

fn test_ggml_matmul_close_to_the_float_product() {
	mut qs := []i8{}
	for i in 0 .. 32 {
		qs << i8(i - 10)
	}
	w := q8_0_block(0x3800, qs)
	x := []f32{len: 32, init: f32(index % 5) - 2}
	y := gq_matmul(x, w, gguf.gq_q8_0, 1, 32, 1)!
	mut want := f32(0)
	for i in 0 .. 32 {
		want += x[i] * 0.5 * f32(i - 10)
	}
	// the integer path quantizes the activations to int8: close to the float product, not equal to it
	assert (if y[0] > want { y[0] - want } else { want - y[0] }) < 0.1
}

fn test_qi_gemm_matches_the_plain_product() {
	x := []f32{len: 3 * 40, init: f32(index % 7) - 3}
	w := []f32{len: 5 * 40, init: f32(index % 11) * 0.25 - 1}
	a := qi_gemm_f32(x, 3, w, 5, 40, [])!
	b := matmul_f32(x, w, 3, 40, 5)!
	for i in 0 .. a.len {
		assert close(a[i], b[i], 1e-5)
	}
	bias := [f32(1), 2, 3, 4, 5]
	c := qi_gemm_f32(x, 3, w, 5, 40, bias)!
	assert close(c[7], b[7] + 3, 1e-5)
}

fn test_wrong_shapes_are_errors_not_overruns() {
	if _ := matmul_f32([f32(1), 2], [f32(1), 2, 3], 1, 2, 2) {
		assert false
	}
	if _ := matmul_f32([f32(1)], [f32(1)], 1, 0, 1) {
		assert false
	}
	if _ := matmul_q([f32(1), 2], [i8(1), 2], [f32(1), 1], 1, 2, 1) {
		assert false // two scales for one row
	}
	if _ := matmul_i4([f32(1), 2, 3], [u8(0)], [f32(1)], 1, 3, 1) {
		assert false // 3 columns need 2 bytes
	}
	if _ := matmul_fp8([f32(1)], [u8(0x38)], [], 1, 1, 1) {
		assert false
	}
	if _ := gq_dequant_row(gguf.gq_q8_0, []u8{len: 10}, 32) {
		assert false
	}
	if _ := gq_dequant_row(gguf.gq_q8_0, []u8{len: 40}, 33) {
		assert false // not whole blocks
	}
	if _ := gq_dequant_row(31, []u8{len: 100}, 32) {
		assert false // no such type
	}
	assert gq_type_ok(gguf.gq_q4_k) && !gq_type_ok(gguf.gq_q8_1) && !gq_type_ok(31)
	if _ := qi_gemm_f32([f32(1)], 1, [f32(1)], 1, 1, [f32(1), 2]) {
		assert false // bias of the wrong length
	}
}

fn le_bytes(v u64, n int) []u8 {
	mut b := []u8{}
	for i in 0 .. n {
		b << u8(v >> (8 * u64(i)))
	}
	return b
}

fn test_reads_a_whole_gguf_tensor() {
	mut h := 'GGUF'.bytes()
	h << le_bytes(3, 4)
	h << le_bytes(1, 8)
	h << le_bytes(0, 8)
	h << le_bytes(1, 8)
	h << 'w'.bytes()
	h << le_bytes(2, 4)
	h << le_bytes(32, 8)
	h << le_bytes(2, 8)
	h << le_bytes(u64(gguf.gq_q8_0), 4)
	h << le_bytes(0, 8)
	for h.len % 32 != 0 {
		h << u8(0)
	}
	mut want := []f32{}
	for r in 0 .. 2 {
		mut qs := []i8{}
		for i in 0 .. 32 {
			qs << i8(i - 16 + r * 3)
			want << 0.5 * f32(i - 16 + r * 3)
		}
		h << q8_0_block(0x3800, qs)
	}
	p := os.join_path(os.vtmp_dir(), 'kern_gguf_${os.getpid()}.gguf')
	os.write_file_array(p, h)!
	defer {
		os.rm(p) or {}
	}
	mut g := gguf.open(p, false)!
	t := g.find('w')?
	got := read_f32(&g, t)!
	assert got == want
	g.close()
}

fn test_int8_activations_are_close_to_the_float_product() {
	x := []f32{len: 64, init: f32(index % 9) - 4}
	q := []i8{len: 2 * 64, init: i8(index % 7 - 3)}
	sc := [f32(0.5), 0.25]
	a := matmul_q(x, q, sc, 1, 64, 2)!
	b := matmul_q_idot(x, q, sc, 1, 64, 2)!
	for i in 0 .. 2 {
		assert (if a[i] > b[i] { a[i] - b[i] } else { b[i] - a[i] }) < 0.05 * (1 + (if a[i] < 0 {
			-a[i]
		} else {
			a[i]
		}))
	}
	if _ := matmul_q_idot(x, q, [f32(1)], 1, 64, 2) {
		assert false // one scale for two rows
	}
}
