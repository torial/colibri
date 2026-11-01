module kernels

// gguf.Ctx.read_f32 of c/gguf.h, the part the gguf module leaves to the kernels: a whole tensor of any decodable ggml
// type as f32, row by row through the C dequantizers (c/ggml_quants.h).
import gguf

// read_f32 returns tensor `t` of `g` as f32 (numel values, row-major in ggml's order)
pub fn read_f32(g &gguf.Ctx, t &gguf.Tensor) ![]f32 {
	numel := t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3]
	if numel <= 0 || numel > i64(1) << 34 {
		return error('tensor ${t.name}: ${numel} elements')
	}
	mut raw := []u8{len: int(t.nbytes)}
	if t.nbytes > 0 && !g.read(t, raw.data, 0, t.nbytes) {
		return error('tensor ${t.name}: read failed')
	}
	if t.typ == gguf.gq_f32 {
		mut out := []f32{len: int(numel)}
		for i in 0 .. int(numel) {
			out[i] = bits(raw, 4 * i)
		}
		return out
	}
	rows := numel / t.ne[0]
	rbytes := gq_row_bytes(t.typ, t.ne[0])
	if rbytes < 0 {
		return error('tensor ${t.name}: type ${t.typ} cannot hold rows of ${t.ne[0]}')
	}
	mut out := []f32{len: int(numel)}
	for r in 0 .. int(rows) {
		row := gq_dequant_row(t.typ, raw[int(i64(r) * rbytes)..int((i64(r) + 1) * rbytes)], t.ne[0])!
		for j in 0 .. row.len {
			out[int(i64(r) * t.ne[0]) + j] = row[j]
		}
	}
	return out
}

fn bits(b []u8, o int) f32 {
	return bf32(u32(b[o]) | u32(b[o + 1]) << 8 | u32(b[o + 2]) << 16 | u32(b[o + 3]) << 24)
}

fn bf32(u u32) f32 {
	return unsafe { *(&f32(&u)) }
}
