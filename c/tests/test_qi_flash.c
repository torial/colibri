/* qi_flash_attention against naive softmax attention in double, ragged shapes, and a
 * throughput line (argv[1] == "bench": 4 s of MiniMax-H3's shape, 25k tokens, 56 heads x 128). */
#include "../qi_flash.h"
#include <time.h>
static float frand(uint32_t *s){ *s = *s * 1664525u + 1013904223u; return ((*s >> 8) / 16777216.f) * 2.f - 1.f; }
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static int check(int Tq, int Tk, int H, int hd, int pad){
    const int ld = H * hd + pad; uint32_t s = 77u + Tq * 3 + Tk * 5 + H + hd;
    float *Q = malloc(sizeof(float) * Tq * ld), *K = malloc(sizeof(float) * Tk * ld), *V = malloc(sizeof(float) * Tk * ld);
    float *O = malloc(sizeof(float) * Tq * ld);
    for (int i = 0; i < Tq * ld; i++) Q[i] = 2.f * frand(&s);
    for (int i = 0; i < Tk * ld; i++) { K[i] = 2.f * frand(&s); V[i] = frand(&s); }
    const float scale = 1.f / sqrtf((float)hd);
    qi_flash_attention(O, ld, Q, ld, K, ld, V, ld, Tq, Tk, H, hd, scale);
    double worst = 0;
    double *p = malloc(sizeof(double) * Tk), *acc = malloc(sizeof(double) * hd);
    for (int h = 0; h < H; h++) for (int t = 0; t < Tq; t++) {
        double mx = -1e300;
        for (int j = 0; j < Tk; j++) { double d = 0; for (int e = 0; e < hd; e++) d += (double)Q[(int64_t)t*ld+h*hd+e] * K[(int64_t)j*ld+h*hd+e]; p[j] = d * scale; if (p[j] > mx) mx = p[j]; }
        double sum = 0; for (int j = 0; j < Tk; j++) { p[j] = exp(p[j] - mx); sum += p[j]; }
        for (int e = 0; e < hd; e++) acc[e] = 0;
        for (int j = 0; j < Tk; j++) for (int e = 0; e < hd; e++) acc[e] += p[j] * V[(int64_t)j*ld+h*hd+e];
        for (int e = 0; e < hd; e++) { double r = acc[e] / sum, g = O[(int64_t)t*ld+h*hd+e]; double err = fabs(g - r); if (err > worst) worst = err; }
    }
    free(Q); free(K); free(V); free(O); free(p); free(acc);
    if (worst > 1e-4) { fprintf(stderr, "FAIL Tq=%d Tk=%d H=%d hd=%d pad=%d: max abs err %.3g\n", Tq, Tk, H, hd, pad, worst); return 1; }
    return 0;
}
int main(int argc, char **argv){
    if (argc > 1 && !strcmp(argv[1], "bench")) {
        const int T = argc > 2 ? atoi(argv[2]) : 6000, H = argc > 3 ? atoi(argv[3]) : 4, hd = 128, ld = H * hd;
        float *Q = malloc(sizeof(float) * (size_t)T * ld), *K = malloc(sizeof(float) * (size_t)T * ld), *V = malloc(sizeof(float) * (size_t)T * ld), *O = malloc(sizeof(float) * (size_t)T * ld);
        uint32_t s = 5; for (size_t i = 0; i < (size_t)T * ld; i++) { Q[i] = frand(&s); K[i] = frand(&s); V[i] = frand(&s); }
        double t0 = now(); qi_flash_attention(O, ld, Q, ld, K, ld, V, ld, T, T, H, hd, 1.f / sqrtf(128.f)); double dt = now() - t0;
        printf("T=%d heads=%d hd=%d: %.2f s, %.1f GFLOP/s\n", T, H, hd, dt, 4.0 * T * (double)T * hd * H / dt / 1e9);
        return 0;
    }
    int bad = 0;
    bad += check(1, 1, 1, 16, 0);  bad += check(5, 7, 2, 32, 3);  bad += check(48, 256, 1, 128, 0);
    bad += check(49, 257, 2, 128, 5); bad += check(100, 513, 3, 64, 0); bad += check(130, 70, 2, 80, 1);
    bad += check(300, 777, 2, 128, 0);
    printf("qi_flash: %s\n", bad ? "FAILED" : "all ok");
    return bad != 0;
}
