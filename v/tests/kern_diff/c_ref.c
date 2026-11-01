/* The C side of the kernel binding test: runs the kernels of c/ directly, on the inputs of a case file, and prints a
 * hash of what they computed. Build with the flags of v/kernels/build.sh:
 *   gcc -O3 -march=x86-64-v3 -fopenmp -I../../../c -o c_ref c_ref.c -lm
 * usage: c_ref cases.txt   one case per line:  name S I O extra threads file...   (file "-" = none) */
#include "quant.h"
#include "ggml_quants.h"
#include "qi_gemm.h"
#include <omp.h>

static void *slurp(const char *p, size_t *n){
    if (!strcmp(p, "-")) { *n = 0; return NULL; }
    FILE *f = fopen(p, "rb"); if (!f) { perror(p); exit(2); }
    fseek(f, 0, SEEK_END); long z = ftell(f); fseek(f, 0, SEEK_SET);
    void *b = malloc((size_t)z + 16); if (fread(b, 1, (size_t)z, f) != (size_t)z) exit(3); fclose(f); *n = (size_t)z; return b;
}
static void report(const char *name, const float *y, int64_t n, int rc){
    uint64_t h = 14695981039346656037ULL; int64_t nonfinite = 0;
    for (int64_t i = 0; i < n; i++) {
        uint32_t u; memcpy(&u, &y[i], 4);
        if (y[i] != y[i]) { u = 0x7fc00000u; nonfinite++; }
        for (int k = 0; k < 4; k++) { h ^= (u >> (8*k)) & 255; h *= 1099511628211ULL; }
    }
    printf("%s rc=%d n=%lld nan=%lld %016llx\n", name, rc, (long long)n, (long long)nonfinite, (unsigned long long)h);
}
int main(int argc, char **argv){
    FILE *f = fopen(argv[1], "r"); if (!f) return 2;
    char name[32], files[4][512];
    int S, I, O, extra, threads;
    while (fscanf(f, "%31s %d %d %d %d %d %511s %511s %511s %511s", name, &S, &I, &O, &extra, &threads, files[0], files[1], files[2], files[3]) == 10) {
        size_t n[4]; void *a[4];
        for (int k = 0; k < 4; k++) a[k] = slurp(files[k], &n[k]);
        omp_set_num_threads(threads);
        float *y = NULL; int64_t ny = (int64_t)S * O; int rc = 0;
        if (!strcmp(name, "f32")) { y = calloc(ny + 1, 4); matmul(y, a[0], a[1], S, I, O); }
        else if (!strcmp(name, "dot")) { y = calloc(2, 4); y[0] = dot_f32_lanes(a[0], a[1], I); ny = 1; }
        else if (!strcmp(name, "q")) { y = calloc(ny + 1, 4); matmul_q(y, a[0], a[1], a[2], S, I, O); }
        else if (!strcmp(name, "i4")) { y = calloc(ny + 1, 4); matmul_i4(y, a[0], a[1], a[2], S, I, O); }
        else if (!strcmp(name, "i4g")) { y = calloc(ny + 1, 4); matmul_i4_grouped(y, a[0], a[1], a[2], S, I, O, extra); }
        else if (!strcmp(name, "fp8")) { y = calloc(ny + 1, 4); matmul_fp8(y, a[0], a[1], a[2], S, I, O); }
        else if (!strcmp(name, "mx4")) { y = calloc(ny + 1, 4); matmul_mxfp4(y, a[0], a[1], a[2], S, I, O); }
        else if (!strcmp(name, "gq")) { y = calloc(ny + 1, 4); rc = gq_matmul(y, a[0], a[1], extra, S, I, O); }
        else if (!strcmp(name, "gqrow")) { ny = I; y = calloc(ny + 1, 4); rc = gq_dequant_row(extra, a[0], y, I); }
        else if (!strncmp(name, "qi", 2)) {
            int fmt = !strcmp(name, "qif32") ? QI_F32 : !strcmp(name, "qibf16") ? QI_BF16 : QI_I8;
            QiMat W = { fmt, O, I, a[1], a[2], 0 };           /* N = O, K = I, M = S */
            y = calloc(ny + 1, 4); qi_gemm(y, a[0], S, &W, a[3]);
        } else { fprintf(stderr, "unknown kernel %s\n", name); return 4; }
        report(name, y, ny, rc);
        for (int k = 0; k < 4; k++) free(a[k]); free(y);
    }
    return 0;
}
