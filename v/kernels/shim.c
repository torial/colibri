/* shim.c -- the C kernels of colibri behind plain extern functions, for the V `kernels` module.
 *
 * The kernels are `static` functions of header-only files (quant.h, ggml_quants.h, matmul_f32.h, qi_gemm.h),
 * compiled into each engine with that engine's flags. This file includes them once, with the flags build.sh
 * passes (-O3 -march=$ARCH -fopenmp), and gives each a coli_k_ name a foreign-function interface can see. It
 * adds no arithmetic of its own: every function is one call, so the results are the engines' bits.
 * Checking sizes is the V side's job (kernels.v); these functions trust their arguments as the kernels do. */
#include "quant.h"
#include "ggml_quants.h"
#include "qi_gemm.h"
#ifdef _OPENMP
#include <omp.h>
#endif

/* what this build of the kernels was compiled for, as text */
int coli_k_isa(char *buf, int n) {
    char t[512]; t[0] = 0;
#ifdef __AVX512F__
    strcat(t, "avx512f ");
#endif
#ifdef __AVX512BW__
    strcat(t, "avx512bw ");
#endif
#ifdef __AVX512VNNI__
    strcat(t, "avx512vnni ");
#endif
#ifdef __AVXVNNI__
    strcat(t, "avxvnni ");
#endif
#ifdef __AVX2__
    strcat(t, "avx2 ");
#endif
#ifdef __FMA__
    strcat(t, "fma ");
#endif
#ifdef __F16C__
    strcat(t, "f16c ");
#endif
#ifdef __SSE4_1__
    strcat(t, "sse4.1 ");
#endif
#ifdef __ARM_NEON
    strcat(t, "neon ");
#endif
#ifdef _OPENMP
    strcat(t, "openmp ");
#endif
    snprintf(buf, (size_t)n, "%s", t);
    return (int)strlen(t);
}
void coli_k_set_threads(int n) {
#ifdef _OPENMP
    if (n > 0) omp_set_num_threads(n);
#else
    (void)n;
#endif
}
int coli_k_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

/* f32 weights */
void coli_k_matmul_f32(float *y, const float *x, const float *W, int S, int I, int O) { matmul(y, x, W, S, I, O); }
float coli_k_dot_f32(const float *x, const float *w, int I) { return dot_f32_lanes(x, w, I); }

/* the OLMoE engine's own int8 matmul (c/olmoe.c matmul_q: one activation row, `omp simd reduction` per output),
 * which is not quant.h's: the V olmoe port calls this one so its logits are the engine's bits */
void coli_k_olm_matmul_q(float *y, const float *x, const int8_t *q, const float *scale, int I, int O) {
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        float acc = 0.f;
        #pragma omp simd reduction(+:acc)
        for (int i = 0; i < I; i++) acc += x[i] * (float)w[i];
        y[o] = acc * scale[o];
    }
}

/* int8 / int4 / fp8 / mxfp4 weights, f32 activations */
void coli_k_matmul_q(float *y, const float *x, const int8_t *q, const float *scale, int S, int I, int O) { matmul_q(y, x, q, scale, S, I, O); }
void coli_k_matmul_i4(float *y, const float *x, const uint8_t *q4, const float *scale, int S, int I, int O) { matmul_i4(y, x, q4, scale, S, I, O); }
void coli_k_matmul_i4_grouped(float *y, const float *x, const uint8_t *q4, const float *scale, int S, int I, int O, int gs) { matmul_i4_grouped(y, x, q4, scale, S, I, O, gs); }
void coli_k_matmul_fp8(float *y, const float *x, const uint8_t *q8, const float *bscale, int S, int I, int O) { matmul_fp8(y, x, q8, bscale, S, I, O); }
void coli_k_matmul_mxfp4(float *y, const float *x, const uint8_t *q4, const uint8_t *e8s, int S, int I, int O) { matmul_mxfp4(y, x, q4, e8s, S, I, O); }

/* ggml block formats */
int coli_k_gq_dequant_row(int t, const uint8_t *src, float *dst, int64_t n) { return gq_dequant_row(t, src, dst, n); }
int coli_k_gq_matmul(float *y, const float *x, const uint8_t *W, int t, int S, int I, int O) { return gq_matmul(y, x, W, t, S, I, O); }
int64_t coli_k_gq_row_bytes(int t, int64_t n) { return gq_row_bytes(t, n); }
int coli_k_gq_has_int(int t) { return gq_has_int(t); }
int coli_k_gq_type_ok(int t) { return gq_type_ok(t); }

/* the image engine's GEMM: Y[M][N] = X[M][K] . W^T (+ bias); fmt 0 f32, 1 bf16, 2 int8 + per-row scale */
void coli_k_qi_gemm(float *Y, const float *X, int M, int fmt, int N, int K, const void *w, const float *sc, int ld, const float *bias) {
    QiMat W = { fmt, N, K, w, sc, ld };
    qi_gemm(Y, X, M, &W, bias);
}
