/* the same sum written with AVX2 intrinsics: eight fused lanes, folded, then a scalar tail (c/colibri.c, "#442") */
#include <immintrin.h>
float simd_dot(const float *x, const float *y, int n, float init) {
    int i = 0; float a = 0;
#ifdef __AVX2__
    __m256 acc = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) acc = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i), acc);
    __m128 lo = _mm256_castps256_ps128(acc), hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi); __m128 sh = _mm_movehl_ps(lo, lo); lo = _mm_add_ps(lo, sh);
    sh = _mm_shuffle_ps(lo, lo, 1); lo = _mm_add_ss(lo, sh); a = _mm_cvtss_f32(lo);
#endif
    a = i ? a + init : init;
    for (; i < n; i++) a += x[i] * y[i];
    return a;
}
