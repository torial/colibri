/* A host stand-in for the little of CUDA that cuda_attn_f32.cu uses, so its indexing, strides and
 * chunking can be tested without a GPU: "device" memory is host memory, a kernel launch runs every
 * block in turn with one pthread per thread of the block, __syncthreads is a barrier, and the cuBLAS
 * strided-batched SGEMM is a naive column-major implementation of its documented semantics.
 * (It cannot say anything about the speed or about cuBLAS's own behaviour.) */
#ifndef CUDA_EMU_H
#define CUDA_EMU_H
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define __global__
#define __shared__ static
#define __device__
#define __forceinline__ inline
struct dim3 { unsigned x, y, z; dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1) : x(a), y(b), z(c) {} };
static thread_local dim3 threadIdx, blockIdx;
static dim3 blockDim, gridDim;
static pthread_barrier_t g_emu_barrier;
static inline void __syncthreads() { pthread_barrier_wait(&g_emu_barrier); }
typedef int cudaStream_t; typedef int cudaError_t; typedef int cublasStatus_t; typedef int cublasHandle_t; typedef int cublasOperation_t;
enum { cudaSuccess = 0, CUBLAS_STATUS_SUCCESS = 0, CUBLAS_OP_N = 0, CUBLAS_OP_T = 1 };
static inline cudaError_t cudaGetLastError() { return 0; }
static inline cublasStatus_t cublasSetStream(cublasHandle_t, cudaStream_t) { return 0; }
/* column-major C (m x n, ldc) = alpha op(A) op(B) + beta C, batch strides in elements */
static inline cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
        const float *alpha, const float *A, int lda, long long sa, const float *B, int ldb, long long sb,
        const float *beta, float *C, int ldc, long long sc, int batch) {
    for (int b = 0; b < batch; b++) {
        const float *a = A + b * sa, *bb = B + b * sb; float *c = C + b * sc;
        for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) {
            double acc = 0;
            for (int p = 0; p < k; p++) {
                const double x = ta == CUBLAS_OP_N ? a[(long long)p * lda + i] : a[(long long)i * lda + p];
                const double y = tb == CUBLAS_OP_N ? bb[(long long)j * ldb + p] : bb[(long long)p * ldb + j];
                acc += x * y;
            }
            c[(long long)j * ldc + i] = (float)(*alpha * acc + (*beta == 0.f ? 0.0 : *beta * c[(long long)j * ldc + i]));
        }
    }
    return 0;
}
struct EmuArgs { void (*fn)(void *); void *a; unsigned tx; dim3 bi; };
static void *emu_thread(void *p) { EmuArgs *e = (EmuArgs *)p; threadIdx = dim3(e->tx, 0, 0); blockIdx = e->bi; e->fn(e->a); return NULL; }
template <typename F> static void emu_launch(dim3 grid, dim3 block, F f) {
    blockDim = block; gridDim = grid;
    for (unsigned by = 0; by < grid.y; by++) for (unsigned bx = 0; bx < grid.x; bx++) {
        pthread_barrier_init(&g_emu_barrier, NULL, block.x);
        pthread_t th[1024]; EmuArgs ar[1024];
        for (unsigned t = 0; t < block.x; t++) {
            ar[t].fn = [](void *a) { (*(F *)a)(); }; ar[t].a = &f; ar[t].tx = t; ar[t].bi = dim3(bx, by, 0);
            pthread_create(&th[t], NULL, emu_thread, &ar[t]);
        }
        for (unsigned t = 0; t < block.x; t++) pthread_join(th[t], NULL);
        pthread_barrier_destroy(&g_emu_barrier);
    }
}
#define QA_LAUNCH(K, GRID, BLOCK, ST, ...) emu_launch((GRID), (BLOCK), [&]() { K(__VA_ARGS__); })
#endif
