module kernels

// The C kernels of colibri, called from V. The arithmetic stays in C (c/quant.h, c/ggml_quants.h,
// c/matmul_f32.h, c/qi_gemm.h: SIMD, OpenMP, the per-ISA paths), reached through shim.c, one call per kernel,
// built into libcolikern.a by build.sh with the engines' flags. What this module adds is what a C caller must
// do by hand and the kernels do not: every function checks the length of each slice against the shape it is
// asked for, so a wrong shape is an error and not a read past a buffer. Results are returned in fresh arrays.
// Differences from the engines: none in the numbers (bit-identical, see tests/kern_diff); the in-place,
// caller-owned output buffers and strided variants of the C API are reached through the `_into` functions.
import math

#flag @VMODROOT/kernels/libcolikern.a
#flag -fopenmp
#flag -Wl,-lm

fn C.coli_k_isa(buf &char, n int) int
fn C.coli_k_set_threads(n int)
fn C.coli_k_threads() int
fn C.coli_k_matmul_f32(y &f32, x &f32, w &f32, s int, i int, o int)
fn C.coli_k_dot_f32(x &f32, w &f32, i int) f32
fn C.coli_k_olm_matmul_q(y &f32, x &f32, q &u8, scale &f32, i int, o int)
fn C.coli_k_matmul_q(y &f32, x &f32, q &i8, scale &f32, s int, i int, o int)
fn C.coli_k_matmul_i4(y &f32, x &f32, q4 &u8, scale &f32, s int, i int, o int)
fn C.coli_k_matmul_i4_grouped(y &f32, x &f32, q4 &u8, scale &f32, s int, i int, o int, gs int)
fn C.coli_k_matmul_fp8(y &f32, x &f32, q8 &u8, bscale &f32, s int, i int, o int)
fn C.coli_k_matmul_mxfp4(y &f32, x &f32, q4 &u8, e8s &u8, s int, i int, o int)
fn C.coli_k_gq_dequant_row(t int, src &u8, dst &f32, n i64) int
fn C.coli_k_gq_matmul(y &f32, x &f32, w &u8, t int, s int, i int, o int) int
fn C.coli_k_gq_row_bytes(t int, n i64) i64
fn C.coli_k_gq_has_int(t int) int
fn C.coli_k_gq_type_ok(t int) int
fn C.coli_k_qi_gemm(y &f32, x &f32, m int, fmt int, n int, k int, w voidptr, sc &f32, ld int, bias &f32)

const fp8_block = 128

// isa: the instruction sets the kernels were compiled for ("avx2 fma f16c openmp ...")
pub fn isa() string {
	mut buf := [512]u8{}
	n := C.coli_k_isa(&char(&buf[0]), 512)
	return unsafe { tos(&buf[0], n) }.trim_space()
}

// set_threads sets the OpenMP thread count of every kernel; threads reports it
pub fn set_threads(n int) {
	C.coli_k_set_threads(n)
}

pub fn threads() int {
	return C.coli_k_threads()
}

fn shape(s int, i int, o int) ! {
	if s < 0 || i <= 0 || o <= 0 {
		return error('bad shape S=${s} I=${i} O=${o}')
	}
	if i64(s) * i64(i) > 0x7fffffff || i64(s) * i64(o) > 0x7fffffff || i64(o) * i64(i) > i64(0x7fffffffffff) {
		return error('shape S=${s} I=${i} O=${o} is too large')
	}
}

fn need(what string, got int, want i64) ! {
	if i64(got) != want {
		return error('${what}: ${got} elements, shape asks for ${want}')
	}
}

// matmul_f32: y[S,O] = x[S,I] . W[O,I]^T, f32 weights, a fixed 32-lane summation order
pub fn matmul_f32(x []f32, w []f32, s int, i int, o int) ![]f32 {
	shape(s, i, o)!
	need('x', x.len, i64(s) * i)!
	need('W', w.len, i64(o) * i)!
	mut y := []f32{len: s * o}
	if s > 0 {
		C.coli_k_matmul_f32(y.data, x.data, w.data, s, i, o)
	}
	return y
}

// dot_f32: the 32-lane dot product matmul_f32 is made of
pub fn dot_f32(x []f32, w []f32) !f32 {
	if x.len != w.len || x.len == 0 {
		return error('dot_f32: lengths ${x.len} and ${w.len}')
	}
	return C.coli_k_dot_f32(x.data, w.data, x.len)
}

// olm_matmul_q: one activation row against int8 weights [O,I] (bytes, read as int8) with a scale per row, as the OLMoE
// engine computes it (c/olmoe.c); y has O values
pub fn olm_matmul_q(x []f32, q []u8, scale []f32, i int, o int) ![]f32 {
	shape(1, i, o)!
	need('x', x.len, i)!
	need('q', q.len, i64(o) * i)!
	need('scale', scale.len, o)!
	mut y := []f32{len: o}
	C.coli_k_olm_matmul_q(y.data, x.data, q.data, scale.data, i, o)
	return y
}

// matmul_q: int8 weights [O,I] with one f32 scale per output row
pub fn matmul_q(x []f32, q []i8, scale []f32, s int, i int, o int) ![]f32 {
	shape(s, i, o)!
	need('x', x.len, i64(s) * i)!
	need('q', q.len, i64(o) * i)!
	need('scale', scale.len, o)!
	mut y := []f32{len: s * o}
	if s > 0 {
		C.coli_k_matmul_q(y.data, x.data, q.data, scale.data, s, i, o)
	}
	return y
}

// matmul_i4: int4 weights, two per byte (low nibble = even column, value = nibble - 8), a scale per row
pub fn matmul_i4(x []f32, q4 []u8, scale []f32, s int, i int, o int) ![]f32 {
	shape(s, i, o)!
	need('x', x.len, i64(s) * i)!
	need('q4', q4.len, i64(o) * ((i + 1) / 2))!
	need('scale', scale.len, o)!
	mut y := []f32{len: s * o}
	if s > 0 {
		C.coli_k_matmul_i4(y.data, x.data, q4.data, scale.data, s, i, o)
	}
	return y
}

// matmul_i4_grouped: int4 weights with a scale per group of `gs` columns
pub fn matmul_i4_grouped(x []f32, q4 []u8, scale []f32, s int, i int, o int, gs int) ![]f32 {
	shape(s, i, o)!
	if gs < 1 {
		return error('group size ${gs}')
	}
	ng := (i + gs - 1) / gs
	need('x', x.len, i64(s) * i)!
	need('q4', q4.len, i64(o) * ((i + 1) / 2))!
	need('scale', scale.len, i64(o) * ng)!
	mut y := []f32{len: s * o}
	if s > 0 {
		C.coli_k_matmul_i4_grouped(y.data, x.data, q4.data, scale.data, s, i, o, gs)
	}
	return y
}

// matmul_fp8: e4m3 bytes [O,I] with an f32 scale per 128x128 block, [ceil(O/128), ceil(I/128)]
pub fn matmul_fp8(x []f32, q8 []u8, bscale []f32, s int, i int, o int) ![]f32 {
	shape(s, i, o)!
	need('x', x.len, i64(s) * i)!
	need('q8', q8.len, i64(o) * i)!
	need('bscale', bscale.len, i64((o + fp8_block - 1) / fp8_block) * ((i + fp8_block - 1) / fp8_block))!
	mut y := []f32{len: s * o}
	if s > 0 {
		C.coli_k_matmul_fp8(y.data, x.data, q8.data, bscale.data, s, i, o)
	}
	return y
}

// matmul_mxfp4: OCP microscaling FP4: e2m1 nibbles [O, I/2] and one ue8m0 exponent per 32 columns
pub fn matmul_mxfp4(x []f32, q4 []u8, e8s []u8, s int, i int, o int) ![]f32 {
	shape(s, i, o)!
	need('x', x.len, i64(s) * i)!
	need('q4', q4.len, i64(o) * ((i + 1) / 2))!
	need('e8s', e8s.len, i64(o) * ((i + 31) / 32))!
	mut y := []f32{len: s * o}
	if s > 0 {
		C.coli_k_matmul_mxfp4(y.data, x.data, q4.data, e8s.data, s, i, o)
	}
	return y
}

// ---- ggml block formats ----

// gq_row_bytes: bytes of a row of n elements of ggml type t, -1 if n is not whole blocks
pub fn gq_row_bytes(t int, n i64) i64 {
	return C.coli_k_gq_row_bytes(t, n)
}

// gq_has_int: the type has an integer (quantized-activation) matmul path
pub fn gq_has_int(t int) bool {
	return C.coli_k_gq_has_int(t) != 0
}

// gq_type_ok: the kernels can decode this type
pub fn gq_type_ok(t int) bool {
	return C.coli_k_gq_type_ok(t) != 0
}

// gq_dequant_row decodes one row of n elements of ggml type t, bit for bit what the reference dequantizer gives
pub fn gq_dequant_row(t int, src []u8, n i64) ![]f32 {
	rb := gq_row_bytes(t, n)
	if rb < 0 || n <= 0 {
		return error('ggml type ${t} cannot hold a row of ${n}')
	}
	if i64(src.len) < rb {
		return error('row of ${n} (type ${t}) needs ${rb} bytes, got ${src.len}')
	}
	mut dst := []f32{len: int(n)}
	if C.coli_k_gq_dequant_row(t, src.data, dst.data, n) != 0 {
		return error('ggml type ${t} has no decoder')
	}
	return dst
}

// gq_matmul: y[S,O] = x[S,I] . W^T for W in ggml type t, [O] rows of gq_row_bytes(t, I) bytes
pub fn gq_matmul(x []f32, w []u8, t int, s int, i int, o int) ![]f32 {
	shape(s, i, o)!
	rb := gq_row_bytes(t, i)
	if rb < 0 {
		return error('ggml type ${t} cannot hold a row of ${i}')
	}
	need('x', x.len, i64(s) * i)!
	need('W', w.len, i64(o) * rb)!
	mut y := []f32{len: s * o}
	if s > 0 {
		if C.coli_k_gq_matmul(y.data, x.data, w.data, t, s, i, o) != 0 {
			return error('ggml type ${t} has no matmul')
		}
	}
	return y
}

// ---- the image engine's GEMM ----

fn qi_check(m int, n int, k int, x_len int, bias []f32) ! {
	if m < 0 || n <= 0 || k <= 0 {
		return error('bad shape M=${m} N=${n} K=${k}')
	}
	need('X', x_len, i64(m) * k)!
	if bias.len != 0 {
		need('bias', bias.len, n)!
	}
}

fn bias_ptr(bias []f32) &f32 {
	if bias.len == 0 {
		return unsafe { nil }
	}
	return bias.data
}

// qi_gemm_f32: Y[M,N] = X[M,K] . W[N,K]^T (+ bias[N]), f32 weights
pub fn qi_gemm_f32(x []f32, m int, w []f32, n int, k int, bias []f32) ![]f32 {
	qi_check(m, n, k, x.len, bias)!
	need('W', w.len, i64(n) * k)!
	mut y := []f32{len: m * n}
	if m > 0 {
		C.coli_k_qi_gemm(y.data, x.data, m, 0, n, k, w.data, unsafe { nil }, 0, bias_ptr(bias))
	}
	return y
}

// qi_gemm_bf16: the same with bf16 weights (raw u16 patterns)
pub fn qi_gemm_bf16(x []f32, m int, w []u16, n int, k int, bias []f32) ![]f32 {
	qi_check(m, n, k, x.len, bias)!
	need('W', w.len, i64(n) * k)!
	mut y := []f32{len: m * n}
	if m > 0 {
		C.coli_k_qi_gemm(y.data, x.data, m, 1, n, k, w.data, unsafe { nil }, 0, bias_ptr(bias))
	}
	return y
}

// qi_gemm_i8: the same with int8 weights and one f32 scale per output row
pub fn qi_gemm_i8(x []f32, m int, w []i8, sc []f32, n int, k int, bias []f32) ![]f32 {
	qi_check(m, n, k, x.len, bias)!
	need('W', w.len, i64(n) * k)!
	need('scale', sc.len, n)!
	mut y := []f32{len: m * n}
	if m > 0 {
		C.coli_k_qi_gemm(y.data, x.data, m, 2, n, k, w.data, sc.data, 0, bias_ptr(bias))
	}
	return y
}

// f16_to_bf16-free helper for tests and callers: the f32 a bf16 pattern stands for
pub fn bf16_to_f32(h u16) f32 {
	return math.f32_from_bits(u32(h) << 16)
}
