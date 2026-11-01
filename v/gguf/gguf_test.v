module gguf

import os

fn w_str(s string) []u8 {
	mut b := []u8{}
	n := u64(s.len)
	for i in 0 .. 8 {
		b << u8(n >> (8 * u64(i)))
	}
	b << s.bytes()
	return b
}

fn le_bytes(v u64, n int) []u8 {
	mut b := []u8{}
	for i in 0 .. n {
		b << u8(v >> (8 * u64(i)))
	}
	return b
}

// one F32 tensor "w" [3 rows of 2], and two keys
fn tiny() []u8 {
	mut h := 'GGUF'.bytes()
	h << le_bytes(3, 4)
	h << le_bytes(1, 8) // tensors
	h << le_bytes(2, 8) // keys
	h << w_str('general.architecture')
	h << le_bytes(u64(t_string), 4)
	h << w_str('olmoe')
	h << w_str('n.i16')
	h << le_bytes(u64(t_i16), 4)
	h << le_bytes(0xFFFE, 2) // -2
	h << w_str('w')
	h << le_bytes(2, 4)
	h << le_bytes(2, 8)
	h << le_bytes(3, 8)
	h << le_bytes(u64(gq_f32), 4)
	h << le_bytes(0, 8)
	for h.len % 32 != 0 {
		h << u8(0)
	}
	for i in 0 .. 24 {
		h << u8(i)
	}
	return h
}

fn test_parses_a_tiny_file() {
	p := os.join_path(os.vtmp_dir(), 'gguf_tiny_${os.getpid()}.gguf')
	os.write_file_array(p, tiny()) or { panic(err) }
	defer {
		os.rm(p) or {}
	}
	assert is_file(p)
	mut g := open(p, false) or { panic(err) }
	assert g.get_str('general.architecture')? == 'olmoe'
	assert g.get_int('n.i16', 0) == -2
	assert g.get_int('absent', 7) == 7
	t := g.find('w')?
	assert t.ne[0] == 2 && t.ne[1] == 3 && t.nbytes == 24
	mut buf := []u8{len: 8}
	assert g.read(t, buf.data, 8, 8)
	assert buf == [u8(8), 9, 10, 11, 12, 13, 14, 15]
	assert !g.read(t, buf.data, 20, 8) // past the tensor
	g.close()
}

fn test_refuses_what_c_refuses() {
	p := os.join_path(os.vtmp_dir(), 'gguf_bad_${os.getpid()}.gguf')
	defer {
		os.rm(p) or {}
	}
	mut b := tiny()
	b[0] = `X`
	os.write_file_array(p, b) or { panic(err) }
	if _ := open(p, false) {
		assert false
	} else {
		assert err.msg().ends_with('not a GGUF file')
	}
	b = tiny()
	b[4] = 9 // version
	os.write_file_array(p, b) or { panic(err) }
	if _ := open(p, false) {
		assert false
	} else {
		assert err.msg().contains('version 9 is not supported')
	}
	os.write_file_array(p, tiny()[..60]) or { panic(err) }
	if _ := open(p, false) {
		assert false
	}
}

fn test_row_bytes() {
	assert gq_row_bytes(gq_q4_k, 512) == 288
	assert gq_row_bytes(gq_q4_k, 100) == -1
	assert gq_row_bytes(gq_f32, 5) == 20
	assert gq_row_bytes(31, 32) == -1
	assert gq_type_name(gq_q8_0) == 'Q8_0' && gq_type_name(31) == '?'
}
