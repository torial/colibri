/* cuda_attn_f32.cu -- full attention in f32 on a GPU with no tensor cores (Maxwell and later).
 *
 * out[h][t][:] = softmax_s( scale * q[h][t] . k[h][s] ) v[h][s]
 *
 * The sequences of a video model (~25k tokens for 4 s of MiniMax-H3, 93k for 15 s) make the
 * Tq x Tk score matrix of a head far too large to keep, so the queries go in chunks: for a
 * chunk of `qchunk` queries and every head, one batched cuBLAS SGEMM makes the scores
 * (S = Q K^T), one kernel takes the row softmax in place, one batched SGEMM makes the output
 * (O = P V). cuBLAS runs at ~90% of an f32 peak on Maxwell (a Quadro M6000: ~5 of ~6.1
 * TFLOP/s); the scores cost one write, one read-modify-write and one read of heads x qchunk x Tk
 * floats a chunk (qchunk 128, 56 heads, 25k keys: 717 MB), about 0.4x the GEMM time.
 * Inputs are [T][heads*hd] row-major with row strides ld* (floats) and head h at column h*hd:
 * the layout of a projection output. RoPE and qk-norm are the caller's.
 *
 * Row-major C(m x n) = A(m x k) B(k x n) is cuBLAS's column-major C^T = B^T A^T, so below:
 *   scores:  S^T(Tk x Bq) = K_cm^T (Tk x hd) . Q_cm (hd x Bq)       (OP_T on K, ld = ldk)
 *   output:  O^T(hd x Bq) = V_cm (hd x Tk) . P^T (Tk x Bq)          (no transpose, ld = ldv)
 * with K, Q, V, O the row-major arrays read as column-major matrices of hd rows, and the
 * batch stride over heads hd. Built for sm_52 with CUDA 12 (see docs/dell-7820.md); the CPU
 * emulation test (tests/test_cuda_attn_emu.cpp) checks the strides and the chunking against
 * the naive softmax with a host stand-in for cuBLAS. Never run on a Maxwell card. */
#ifdef __CUDA_EMU__
#include "tests/cuda_emu.h"
#else
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define QA_LAUNCH(K, GRID, BLOCK, ST, ...) K<<<GRID, BLOCK, 0, ST>>>(__VA_ARGS__)
#endif
#include <stddef.h>

#define QA_THREADS 256

/* row softmax in place: S is [heads][Bq][Tk]; one block a row. The scale is applied inside the exponent. */
__global__ void qa_softmax(float *S, int Bq, int Tk, float scale) {
    __shared__ float red[QA_THREADS];
    float *row = S + ((size_t)blockIdx.y * Bq + blockIdx.x) * (size_t)Tk;
    const int t = threadIdx.x;
    float mx = -3.402823466e38f;
    for (int j = t; j < Tk; j += QA_THREADS) { const float v = row[j] * scale; if (v > mx) mx = v; }
    red[t] = mx; __syncthreads();
    for (int s = QA_THREADS / 2; s > 0; s >>= 1) { if (t < s && red[t + s] > red[t]) red[t] = red[t + s]; __syncthreads(); }
    mx = red[0]; __syncthreads();
    float sum = 0.f;
    for (int j = t; j < Tk; j += QA_THREADS) { const float e = expf(row[j] * scale - mx); row[j] = e; sum += e; }
    red[t] = sum; __syncthreads();
    for (int s = QA_THREADS / 2; s > 0; s >>= 1) { if (t < s) red[t] += red[t + s]; __syncthreads(); }
    const float inv = 1.f / red[0];
    for (int j = t; j < Tk; j += QA_THREADS) row[j] *= inv;
}

/* floats of scratch the call needs */
extern "C" size_t coli_cuda_attention_f32_scratch(int Tk, int heads, int qchunk) {
    return (size_t)heads * (size_t)qchunk * (size_t)Tk;
}

/* 0 on success, else a cuBLAS or CUDA status + 1000 */
extern "C" int coli_cuda_attention_f32(cublasHandle_t blas, cudaStream_t st,
                                       float *O, int ldo, const float *Q, int ldq, const float *K, int ldk,
                                       const float *V, int ldv, int Tq, int Tk, int heads, int hd,
                                       float scale, int qchunk, float *S) {
    if (Tq <= 0 || Tk <= 0 || heads <= 0 || hd <= 0 || qchunk <= 0) return 0;
    cublasStatus_t rc = cublasSetStream(blas, st);
    if (rc != CUBLAS_STATUS_SUCCESS) return (int)rc + 1000;
    const float one = 1.f, zero = 0.f;
    for (int q0 = 0; q0 < Tq; q0 += qchunk) {
        const int bq = Tq - q0 < qchunk ? Tq - q0 : qchunk;
        /* S[h] (bq x Tk, row-major) = Q[h] K[h]^T */
        rc = cublasSgemmStridedBatched(blas, CUBLAS_OP_T, CUBLAS_OP_N, Tk, bq, hd, &one,
                                       K, ldk, (long long)hd, Q + (size_t)q0 * ldq, ldq, (long long)hd,
                                       &zero, S, Tk, (long long)bq * Tk, heads);
        if (rc != CUBLAS_STATUS_SUCCESS) return (int)rc + 1000;
        QA_LAUNCH(qa_softmax, dim3(bq, heads), dim3(QA_THREADS), st, S, bq, Tk, scale);
        /* O[h] (bq x hd) = P[h] V[h] */
        rc = cublasSgemmStridedBatched(blas, CUBLAS_OP_N, CUBLAS_OP_N, hd, bq, Tk, &one,
                                       V, ldv, (long long)hd, S, Tk, (long long)bq * Tk,
                                       &zero, O + (size_t)q0 * ldo, ldo, (long long)hd, heads);
        if (rc != CUBLAS_STATUS_SUCCESS) return (int)rc + 1000;
    }
    cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? 0 : (int)e + 1000;
}
