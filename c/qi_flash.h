/* qi_flash.h -- blocked (flash-style) full attention in f32 for long sequences.
 *
 * out[h][t][:] = softmax_s( scale * q[h][t] . k[h][s] ) v[h][s]      (no mask, no bias)
 *
 * Why: the image engine's attention builds the whole Tq x Tk score matrix of a head
 * (qi_gemm_ld, then a softmax over it). That is fine for the 1.3k tokens of a picture and
 * impossible for a video model: a 4 s 768p clip of MiniMax-H3 is ~25k tokens, a score
 * matrix of 2.5 GB per head, and 93k tokens for 15 s (35 GB). Here a head is processed in
 * tiles of QF_BR queries x QF_BC keys with an online softmax (running max and sum per
 * query), so the working set is a few hundred KB per thread whatever the length, and the
 * two products run through qi_gemm.h's register-tile kernel (qi_kernel, 6x16 f32 FMA):
 *   S tile  = Q (6 x hd) . K^T      keys packed 16 at a time, k-major: panel[d*16 + j]
 *   O tile += P (6 x bc) . V        values packed 16 columns at a time, k-major
 * Inputs are [T][heads][hd] row-major with arbitrary row strides (ldq, ldk, ldv, ldo in
 * floats), the layout of a projection output. Results equal the naive softmax attention up
 * to f32 summation order (tests/test_qi_flash.c checks 1e-5 relative to the naive double
 * result). Any RoPE / qk-norm is applied by the caller before the call. */
#ifndef COLIBRI_QI_FLASH_H
#define COLIBRI_QI_FLASH_H
#include "qi_gemm.h"
#include <math.h>
#include <float.h>

#define QF_BR 48     /* queries per tile: 8 groups of the kernel's 6 rows */
#define QF_BC 256    /* keys per tile: 16 panels of 16 */

static inline float qf_exp(float x){ return expf(x); }

/* pack keys [k0, k0+nk) of one head into panels of 16 keys, k-major, hd values per key:
 * panel p, depth d, lane j  ->  kp[(p*hd + d)*16 + j]; keys past nk are zero */
static void qf_pack_k(float *kp, const float *K, int ldk, int k0, int nk, int hd){
    const int np = (nk + 15) / 16;
    for (int p = 0; p < np; p++)
        for (int j = 0; j < 16; j++) {
            const int s = p * 16 + j;
            if (s < nk) { const float *r = K + (int64_t)(k0 + s) * ldk; for (int d = 0; d < hd; d++) kp[((int64_t)p * hd + d) * 16 + j] = r[d]; }
            else        for (int d = 0; d < hd; d++) kp[((int64_t)p * hd + d) * 16 + j] = 0.f;
        }
}
/* values [k0, k0+nk) into column panels: panel c (16 columns), depth s (key), lane j ->
 * vp[(c*nk_pad + s)*16 + j]; columns past hd and keys past nk are zero */
static void qf_pack_v(float *vp, const float *V, int ldv, int k0, int nk, int nk_pad, int hd){
    const int nc = (hd + 15) / 16;
    for (int c = 0; c < nc; c++)
        for (int s = 0; s < nk_pad; s++)
            for (int j = 0; j < 16; j++) {
                const int col = c * 16 + j;
                vp[((int64_t)c * nk_pad + s) * 16 + j] = (s < nk && col < hd) ? V[(int64_t)(k0 + s) * ldv + col] : 0.f;
            }
}

/* One head. q/k/v/o point at the head's first element (row stride ld*). */
static void qf_head(float *O, int ldo, const float *Q, int ldq, const float *K, int ldk, const float *V, int ldv,
                    int Tq, int Tk, int hd, float scale, float *kp_all, float *vp_all){
    const int nkb = (Tk + QF_BC - 1) / QF_BC, nqb = (Tq + QF_BR - 1) / QF_BR;
    const int nc = (hd + 15) / 16;
    /* pack every key / value block of the head once; all query tiles read them */
    #pragma omp parallel for schedule(static)
    for (int kb = 0; kb < nkb; kb++) {
        const int k0 = kb * QF_BC, nk = Tk - k0 < QF_BC ? Tk - k0 : QF_BC;
        qf_pack_k(kp_all + (int64_t)kb * QF_BC * hd, K, ldk, k0, nk, hd);
        qf_pack_v(vp_all + (int64_t)kb * QF_BC * (int64_t)nc * 16, V, ldv, k0, nk, (nk + 15) / 16 * 16, hd);
    }
    #pragma omp parallel
    {
        float *S = (float *)malloc(sizeof(float) * QF_BR * QF_BC);
        float *Acc = (float *)malloc(sizeof(float) * QF_BR * (size_t)nc * 16);
        float m[QF_BR], l[QF_BR];
        if (!S || !Acc) { fprintf(stderr, "OOM qi_flash\n"); exit(1); }
        #pragma omp for schedule(dynamic, 1)
        for (int qb = 0; qb < nqb; qb++) {
            const int q0 = qb * QF_BR, nq = Tq - q0 < QF_BR ? Tq - q0 : QF_BR;
            for (int i = 0; i < nq; i++) { m[i] = -FLT_MAX; l[i] = 0.f; }
            memset(Acc, 0, sizeof(float) * QF_BR * (size_t)nc * 16);
            for (int kb = 0; kb < nkb; kb++) {
                const int k0 = kb * QF_BC, nk = Tk - k0 < QF_BC ? Tk - k0 : QF_BC;
                const int np = (nk + 15) / 16, nkp = np * 16;
                const float *kp = kp_all + (int64_t)kb * QF_BC * hd;
                const float *vp = vp_all + (int64_t)kb * QF_BC * (int64_t)nc * 16;
                /* S[i][s] = Q[i] . K[s]: panels of 16 keys, tiles of 6 queries */
                for (int p = 0; p < np; p++)
                    for (int i = 0; i < nq; i += QI_MR) {
                        const int mr = nq - i < QI_MR ? nq - i : QI_MR;
                        qi_kernel(S + (int64_t)i * QF_BC + p * 16, QF_BC, Q + (int64_t)(q0 + i) * ldq, ldq,
                                  kp + (int64_t)p * hd * 16, hd, mr, 16, 0);
                    }
                /* online softmax of the tile: new max, rescale, exponentiate; padded keys get weight 0 */
                for (int i = 0; i < nq; i++) {
                    float *s = S + (int64_t)i * QF_BC, mx = m[i];
                    for (int j = 0; j < nk; j++) { s[j] *= scale; if (s[j] > mx) mx = s[j]; }
                    const float corr = m[i] == -FLT_MAX ? 0.f : qf_exp(m[i] - mx);
                    float sum = 0.f;
                    for (int j = 0; j < nk; j++) { s[j] = qf_exp(s[j] - mx); sum += s[j]; }
                    for (int j = nk; j < nkp; j++) s[j] = 0.f;
                    l[i] = l[i] * corr + sum; m[i] = mx;
                    float *a = Acc + (int64_t)i * nc * 16;
                    for (int d = 0; d < nc * 16; d++) a[d] *= corr;
                }
                /* Acc[i][:] += P[i][:] . V : column panels of 16, tiles of 6 queries (accumulate) */
                for (int c = 0; c < nc; c++)
                    for (int i = 0; i < nq; i += QI_MR) {
                        const int mr = nq - i < QI_MR ? nq - i : QI_MR;
                        qi_kernel(Acc + (int64_t)i * nc * 16 + c * 16, nc * 16, S + (int64_t)i * QF_BC, QF_BC,
                                  vp + (int64_t)c * nkp * 16, nkp, mr, 16, 1);
                    }
            }
            for (int i = 0; i < nq; i++) {
                const float inv = l[i] > 0.f ? 1.f / l[i] : 0.f;
                const float *a = Acc + (int64_t)i * nc * 16;
                float *o = O + (int64_t)(q0 + i) * ldo;
                for (int d = 0; d < hd; d++) o[d] = a[d] * inv;
            }
        }
        free(S); free(Acc);
    }
}

/* Tq queries and Tk keys, `heads` heads of width hd; q/k/v/o are [T][heads*hd]-style arrays
 * with row strides ldq/ldk/ldv/ldo (floats) and head h at column offset h*hd. */
static void qi_flash_attention(float *O, int ldo, const float *Q, int ldq, const float *K, int ldk, const float *V, int ldv,
                               int Tq, int Tk, int heads, int hd, float scale){
    if (Tq <= 0 || Tk <= 0 || heads <= 0 || hd <= 0) return;
    const int nkb = (Tk + QF_BC - 1) / QF_BC, nc = (hd + 15) / 16;
    float *kp = (float *)malloc(sizeof(float) * (size_t)nkb * QF_BC * hd);
    float *vp = (float *)malloc(sizeof(float) * (size_t)nkb * QF_BC * (size_t)nc * 16);
    if (!kp || !vp) { fprintf(stderr, "OOM qi_flash pack\n"); exit(1); }
    for (int h = 0; h < heads; h++)
        qf_head(O + (int64_t)h * hd, ldo, Q + (int64_t)h * hd, ldq, K + (int64_t)h * hd, ldk, V + (int64_t)h * hd, ldv,
                Tq, Tk, hd, scale, kp, vp);
    free(kp); free(vp);
}
#endif
