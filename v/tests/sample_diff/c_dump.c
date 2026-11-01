/* C side of the sampling differential test: replays a case file through c/sample.h and prints
 * what it does. Build: gcc -O1 -I../../../c -o c_dump c_dump.c -lm   usage: c_dump cases.txt tokenizer.json */
#include <stdint.h>
#include <stdlib.h>
typedef struct { int n_stop; int stop_ids[64]; } Cfg;
static float g_temp, g_nuc;
static float *falloc(int64_t n){ return (float*)calloc((size_t)n, sizeof(float)); }
#include "sample.h"

static float bits2f(uint32_t u){ float f; memcpy(&f, &u, 4); return f; }
static uint32_t f2bits(float f){ uint32_t u; memcpy(&u, &f, 4); return u; }
static uint64_t d2bits(double d){ uint64_t u; if (d != d) return 0x7ff8000000000000ULL; /* the sign of a NaN is whatever the hardware made */ memcpy(&u, &d, 8); return u; }

int main(int argc, char **argv){
    FILE *f = fopen(argv[1], "r"); if (!f) return 2;
    char kind[8];
    while (fscanf(f, "%7s", kind) == 1) {
        if (kind[0] == 'S') {                    /* S temp nuc rng V steps, then steps x (ban + V logits) */
            unsigned t, n; unsigned long long rng; int V, steps;
            if (fscanf(f, "%x %x %llx %d %d", &t, &n, &rng, &V, &steps) != 5) return 3;
            g_temp = bits2f(t); g_nuc = bits2f(n); g_rng = rng;
            free(g_pbuf); free(g_pidx); g_pbuf = NULL; g_pidx = NULL;
            float *lo = malloc(sizeof(float) * V);
            for (int s = 0; s < steps; s++) {
                int ban; if (fscanf(f, "%d", &ban) != 1) return 3;
                for (int i = 0; i < V; i++) { unsigned u; if (fscanf(f, "%x", &u) != 1) return 3; lo[i] = bits2f(u); }
                int p = pick_tok(lo, V, ban);
                uint64_t h = 14695981039346656037ULL;
                if (g_temp > 0) for (int i = 0; i < V; i++) { h ^= f2bits(g_pbuf[i]); h *= 1099511628211ULL; }
                printf("P %d %016llx %016llx\n", p, (unsigned long long)h, (unsigned long long)g_rng);
                LogprobRow r; LogprobStatus st = logprob_row_checked(lo, V, &r);
                int tgt = (ban >= 0 ? ban : p) % V; double lp; LogprobStatus st2 = logprob_from_row_checked(lo, tgt, &r, &lp);
                printf("L %s %08x %016llx %016llx %d %s %016llx", logprob_status_name(st), f2bits(r.max), (unsigned long long)d2bits(r.logse),
                       (unsigned long long)d2bits(r.logZ), r.argmax, logprob_status_name(st2), (unsigned long long)d2bits(st2 == LOGPROB_FINITE ? lp : 0));
                if (V > 0 && isfinite(lo[0])) { int am; double v = logprob_target(lo, V, tgt, &am); printf(" %016llx %d", (unsigned long long)d2bits(v), am); }
                printf("\n");
            }
            free(lo);
        } else if (kind[0] == 'T') {             /* T eos serve_batch all_stops ncfg ids... : stops without and with the tokenizer */
            int eos, sb, as, nc; if (fscanf(f, "%d %d %d %d", &eos, &sb, &as, &nc) != 4) return 3;
            Cfg c; c.n_stop = nc; for (int i = 0; i < nc; i++) if (fscanf(f, "%d", &c.stop_ids[i]) != 1) return 3;
            static Tok T; static int loaded = 0; if (!loaded) { tok_load(&T, argv[2]); loaded = 1; }
            if (sb) setenv("SERVE", "1", 1); else unsetenv("SERVE");
            if (sb) setenv("SERVE_BATCH", "1", 1); else unsetenv("SERVE_BATCH");
            if (as) setenv("COLI_SERVE_ALL_STOPS", "1", 1); else unsetenv("COLI_SERVE_ALL_STOPS");
            stops_arm(&c, eos); printf("A"); for (int i = 0; i < g_nstop; i++) printf(" %d", g_stop[i]); printf("\n");
            stops_arm_tok(&c, eos, &T); printf("B"); for (int i = 0; i < g_nstop; i++) printf(" %d", g_stop[i]); printf("\n");
        } else if (kind[0] == 'M') {             /* M hexstring: mt_is_glm */
            char s[256]; if (fscanf(f, "%255s", s) != 1) return 3; printf("G %d\n", mt_is_glm(s[0] == '-' ? "" : s));
        }
    }
    return 0;
}
