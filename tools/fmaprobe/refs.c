/* Reference summation orders for fmaprobe: what a float32 dot product can mean. init is the starting accumulator. */
#include <math.h>
#include <stddef.h>
float ref_seq(const float *x, const float *y, int n, float init)    { volatile float t; float a = init; for (int i = 0; i < n; i++) { t = x[i] * y[i]; a = a + t; } return a; }
float ref_seqfma(const float *x, const float *y, int n, float init) { float a = init; for (int i = 0; i < n; i++) a = fmaf(x[i], y[i], a); return a; }
static float lanes(const float *x, const float *y, int n, float init, int L, int fused, int hfold) {
    float acc[16] = {0}; int i = 0;
    for (; i + L <= n; i += L) for (int j = 0; j < L; j++) acc[j] = fused ? fmaf(x[i + j], y[i + j], acc[j]) : acc[j] + x[i + j] * y[i + j];
    float a;
    if (hfold && L == 8) { float l0 = acc[0] + acc[4], l1 = acc[1] + acc[5], l2 = acc[2] + acc[6], l3 = acc[3] + acc[7]; a = (l0 + l2) + (l1 + l3); }
    else if (hfold && L == 4) { a = (acc[0] + acc[2]) + (acc[1] + acc[3]); }
    else { a = 0; for (int j = 0; j < L; j++) a += acc[j]; }
    if (i == 0) a = init; else a = a + init;
    for (; i < n; i++) a = fmaf(x[i], y[i], a);   /* a scalar tail is the part of the loop the compiler keeps */
    return a;
}
float ref_lanes4(const float *x, const float *y, int n, float init) { return lanes(x, y, n, init, 4, 1, 1); }
float ref_lanes8(const float *x, const float *y, int n, float init) { return lanes(x, y, n, init, 8, 1, 1); }
float ref_lanes16(const float *x, const float *y, int n, float init) { return lanes(x, y, n, init, 16, 1, 0); }
float ref_pair(const float *x, const float *y, int n, float init) {
    float p[1024]; if (n > 1024) n = 1024; for (int i = 0; i < n; i++) p[i] = x[i] * y[i];
    for (int m = n; m > 1; m = (m + 1) / 2) for (int i = 0; i < m / 2; i++) p[i] = p[2 * i] + p[2 * i + 1];
    return n ? p[0] + init : init;
}
/* the shape gcc gives a plain `a += x[i]*y[i]` loop: whole blocks of W products formed with packed multiplies and added one by one
 * (multiply and add stay separate), the remainder as a fused scalar chain */
static float hybrid(const float *x, const float *y, int n, float init, int W) {
    volatile float t; float a = init; int i = 0;
    for (; i + W <= n; i += W) for (int j = 0; j < W; j++) { t = x[i + j] * y[i + j]; a = a + t; }
    for (; i < n; i++) a = fmaf(x[i], y[i], a);
    return a;
}
float ref_hyb4(const float *x, const float *y, int n, float init)  { return hybrid(x, y, n, init, 4); }
float ref_hyb8(const float *x, const float *y, int n, float init)  { return hybrid(x, y, n, init, 8); }
float ref_hyb16(const float *x, const float *y, int n, float init) { return hybrid(x, y, n, init, 16); }
