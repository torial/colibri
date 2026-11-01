/* The C side of the safetensors differential test: the same dump as v_dump.v, from c/st.h.
 * Build: gcc -O1 -I../../../c -o c_dump c_dump.c -lm */
#include "st.h"
static void hex(const char *s){ for (; *s; s++) printf("%02x", (unsigned char)*s); }
static uint64_t fnv(const unsigned char *b, size_t n){ uint64_t h = 14695981039346656037ULL; for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; } return h; }
int main(int argc, char **argv){
    shards S; st_init(&S, argv[1]);
    for (int i = 0; i < S.n; i++) {
        st_tensor *t = &S.t[i];
        printf("T "); hex(t->name);
        printf(" %d %lld %lld %lld %d [", t->dtype, (long long)t->off, (long long)t->nbytes, (long long)t->numel, t->rank);
        for (int k = 0; k < t->rank; k++) printf("%s%lld", k ? "," : "", (long long)t->shape[k]);
        printf("]\n");
        unsigned char *raw = malloc(t->nbytes ? t->nbytes : 1);
        st_read_raw(&S, t->name, raw, 0);
        printf("W %016llx\n", (unsigned long long)fnv(raw, (size_t)t->nbytes)); free(raw);
        if (t->dtype < 3) {
            float *f = malloc((t->numel ? t->numel : 1) * sizeof(float));
            st_read_f32(&S, t->name, f, 0);
            unsigned char *b = malloc((t->numel ? t->numel : 1) * 4);
            for (int64_t k = 0; k < t->numel; k++) { uint32_t v; memcpy(&v, &f[k], 4); b[4*k] = v; b[4*k+1] = v >> 8; b[4*k+2] = v >> 16; b[4*k+3] = v >> 24; }
            printf("R %016llx\n", (unsigned long long)fnv(b, (size_t)t->numel * 4)); free(f); free(b);
        }
    }
    return 0;
}
