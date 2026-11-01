/* C side of the extended safetensors differential test: opens a model directory (optionally a search path of
 * directories), prints the tensor table and the format stamps, then runs the operations of an ops file against
 * st.h and prints what each returned. A refusing operation exits like st.h does; the harness compares exit
 * status, stdout and stderr with the V side.
 * Build: gcc -O1 -I../../../c [-DST_PREAD_CHUNK=n] -o c_ext c_ext.c -lm   usage: c_ext dir extra|- ops */
#define ST_GGUF 1
#include "st.h"
static void hex(const char *s){ for (; *s; s++) printf("%02x", (unsigned char)*s); }
static uint64_t fnv(const unsigned char *b, size_t n){ uint64_t h = 14695981039346656037ULL; for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; } return h; }
static uint64_t fnvf(const float *f, int64_t n){ uint64_t h = 14695981039346656037ULL; for (int64_t i = 0; i < n; i++) { uint32_t v; memcpy(&v, &f[i], 4);
    for (int k = 0; k < 4; k++) { h ^= (v >> (8*k)) & 255; h *= 1099511628211ULL; } } return h; }
static const char *base(const char *p){ const char *b = strrchr(p, '/'); return b ? b + 1 : p; }
static char *unhex(const char *h){ size_t n = strlen(h) / 2; char *s = malloc(n + 1); for (size_t i = 0; i < n; i++) { unsigned v; sscanf(h + 2*i, "%2x", &v); s[i] = (char)v; } s[n] = 0; return s; }
int main(int argc, char **argv){
    shards S;
    st_init_multi(&S, argv[1], strcmp(argv[2], "-") ? argv[2] : NULL);
    for (int i = 0; i < S.n; i++) {
        st_tensor *t = &S.t[i];
        printf("T "); hex(t->name);
        printf(" %d %lld %lld %lld %d [", t->dtype, (long long)t->off, (long long)t->nbytes, (long long)t->numel, t->rank);
        for (int k = 0; k < t->rank; k++) printf("%s%lld", k ? "," : "", (long long)t->shape[k]);
        printf("] %s %d\n", base(S.paths[st_fidx(&S, t->fd)]), st_fidx(&S, t->fd));
    }
    for (int i = 0; i < S.fmt_n; i++) { printf("F "); hex(S.fmt_name[i]); printf(" "); hex(S.fmt_val[i]); printf("\n"); }
    if (st_gguf_active) {
        const gguf_ctx *Gs[2] = { &st_gguf_meta, st_gguf_head.n_files ? &st_gguf_head : NULL };
        for (int gi = 0; gi < 2; gi++) { const gguf_ctx *G = Gs[gi]; if (!G) continue;
            printf("G%d parts=%d kv=%d tensors=%d\n", gi, G->n_files, G->n_kv, G->n_tensors);
            for (int i = 0; i < G->n_kv; i++) { const gguf_kv *k = &G->kv[i]; printf("K "); hex(k->key); printf(" %d", k->type);
                switch (k->type) {
                case GGUF_T_U8: case GGUF_T_U16: case GGUF_T_U32: case GGUF_T_U64: case GGUF_T_BOOL: printf(" u%llu", (unsigned long long)k->v.u); break;
                case GGUF_T_I8: case GGUF_T_I16: case GGUF_T_I32: case GGUF_T_I64: printf(" i%lld", (long long)k->v.i); break;
                case GGUF_T_F32: case GGUF_T_F64: { double d = k->v.f; uint64_t b; memcpy(&b, &d, 8); printf(" f%016llx", (unsigned long long)b); break; }
                case GGUF_T_STRING: printf(" s"); hex(k->s ? k->s : ""); break;
                case GGUF_T_ARRAY: printf(" a%d n%llu", k->atype, (unsigned long long)k->n);
                    if (k->atype == GGUF_T_STRING) { for (uint64_t j = 0; j < k->n; j++) { printf(" "); hex(k->strs[j] ? k->strs[j] : ""); } }
                    else printf(" %016llx", (unsigned long long)fnv((const unsigned char *)k->arr, (size_t)(k->n * (uint64_t)gguf_scalar_size(k->atype)))); break; }
                printf("\n"); }
            for (int i = 0; i < G->n_tensors; i++) { const gguf_tensor *t = &G->t[i]; printf("GT "); hex(t->name);
                printf(" %d %d %lld %lld %lld %lld %d %llu %lld\n", t->type, t->n_dims, (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3], t->file, (unsigned long long)t->off, (long long)t->nbytes); }
        }
    }
    FILE *f = fopen(argv[3], "r"); if (!f) return 2;
    char op[32];
    while (fscanf(f, "%31s", op) == 1) {
        char nh[2048]; long long a, b, c, d, e, g;
        if (!strcmp(op, "mirror") || !strcmp(op, "mirrorinit")) {
            char dir[1024]; if (fscanf(f, "%1023s", dir) != 1) return 3;
            int n = !strcmp(op, "mirror") ? st_mirror_add(&S, dir) : st_mirror_init(&S, dir);
            printf("mirror %d nrep=%d\n", n, S.nrep);
        } else if (!strcmp(op, "mirrorreset")) { st_mirror_reset(&S); printf("mirrorreset nrep=%d\n", S.nrep);
        } else if (!strcmp(op, "find")) {
            if (fscanf(f, "%2047s", nh) != 1) return 3; char *n = unhex(nh);
            printf("find %lld %lld %d\n", (long long)st_numel(&S, n), (long long)st_nbytes(&S, n), st_has(&S, n));
            const char *st = st_fmt_stamp(&S, n); printf("stamp %s\n", st ? st : "(none)");
        } else if (!strcmp(op, "f32") || !strcmp(op, "scale")) {
            if (fscanf(f, "%2047s %lld %lld", nh, &a, &b) != 3) return 3; char *n = unhex(nh);
            st_tensor *t = st_find(&S, n); int64_t cnt = t ? t->numel : 0; if (cnt < 0) cnt = 0;
            float *o = malloc((size_t)(cnt ? cnt : 1) * 4); int64_t r;
            if (!strcmp(op, "f32")) r = a < 0 ? st_read_f32(&S, n, o, (int)b) : st_read_f32_cap(&S, n, o, a, (int)b);
            else r = st_read_scale_f32(&S, n, o, a, (int)b);
            printf("%s %lld %016llx\n", op, (long long)r, (unsigned long long)fnvf(o, r)); free(o);
        } else if (!strcmp(op, "raw")) {
            if (fscanf(f, "%2047s %lld %lld", nh, &a, &b) != 3) return 3; char *n = unhex(nh);
            int64_t nb = st_nbytes(&S, n); unsigned char *o = malloc((size_t)(nb > 0 ? nb : 1));
            if (a < 0) st_read_raw(&S, n, o, (int)b); else st_read_raw_cap(&S, n, o, a, (int)b);
            printf("raw %lld %016llx\n", (long long)nb, (unsigned long long)fnv(o, (size_t)nb)); free(o);
        } else if (!strcmp(op, "slice")) {      /* slice NAME byte_off nbytes cap drop */
            if (fscanf(f, "%2047s %lld %lld %lld %lld", nh, &a, &b, &c, &d) != 5) return 3; char *n = unhex(nh);
            unsigned char *o = malloc((size_t)(c > 0 ? c + 1 : 1)); memset(o, 0xAB, (size_t)(c > 0 ? c + 1 : 1));
            st_read_slice_raw_cap(&S, n, a, b, o, c, (int)d);
            printf("slice %lld %016llx\n", b, (unsigned long long)fnv(o, (size_t)(c > 0 ? c + 1 : 1))); free(o);
        } else if (!strcmp(op, "slicef")) {     /* slicef NAME elem_off n drop */
            if (fscanf(f, "%2047s %lld %lld %lld", nh, &a, &b, &c) != 4) return 3; char *n = unhex(nh);
            float *o = malloc((size_t)(b > 0 ? b : 1) * 4);
            st_read_slice_f32(&S, n, a, b, o, (int)c);
            printf("slicef %lld %016llx\n", b, (unsigned long long)fnvf(o, b > 0 ? b : 0)); free(o);
        } else if (!strcmp(op, "range")) {      /* range SHARD off nbytes cap drop */
            if (fscanf(f, "%lld %lld %lld %lld %lld", &a, &b, &c, &d, &e) != 5) return 3;
            int fd = (a >= 0 && a < S.nfd) ? S.fds[a] : -7;
            unsigned char *o = malloc((size_t)(d > 0 ? d + 1 : 1)); memset(o, 0xAB, (size_t)(d > 0 ? d + 1 : 1));
            st_read_range_raw_cap(&S, fd, b, c, c > 0 ? o : NULL, d, (int)e, "range");
            printf("range %016llx\n", (unsigned long long)fnv(o, (size_t)(d > 0 ? d + 1 : 1))); free(o);
        } else if (!strcmp(op, "rrep")) {       /* rrep SHARD REP off nbytes cap drop direct */
            long long v[7]; if (fscanf(f, "%lld %lld %lld %lld %lld %lld %lld", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6]) != 7) return 3;
            int fd = (v[0] >= 0 && v[0] < S.nfd) ? S.fds[v[0]] : -7;
            long long pad = v[2] > 0 ? v[2] % ST_DIRECT_ALIGN : 0, total = ((pad + (v[4] > 0 ? v[4] : 0) + ST_DIRECT_ALIGN) / ST_DIRECT_ALIGN + 1) * ST_DIRECT_ALIGN;
            unsigned char *buf; if (posix_memalign((void**)&buf, ST_DIRECT_ALIGN, (size_t)total)) return 5; memset(buf, 0xAB, (size_t)total);
            st_read_range_rep(&S, fd, (int)v[1], v[2], v[3], buf + pad + (v[6] == 2 ? 16 : 0), v[4], (int)v[5], v[6] != 0, "rrep");
            printf("rrep %016llx\n", (unsigned long long)fnv(buf, (size_t)total)); free(buf);
        } else if (!strcmp(op, "nrep")) { printf("nrep %d\n", S.nrep);
        } else if (!strcmp(op, "fdrep")) {
            if (fscanf(f, "%lld %lld", &a, &b) != 2) return 3;
            int fd = (a >= 0 && a < S.nfd) ? S.fds[a] : -7;
            printf("fdrep %d %d %d\n", st_fd_rep(&S, fd, (int)b) >= 0, st_direct_fd_rep(&S, fd, (int)b) >= 0, st_direct_fd(&S, fd) >= 0);
        } else if (!strcmp(op, "prefetch")) { if (fscanf(f, "%2047s %lld", nh, &a) != 2) return 3; char *n = unhex(nh); st_prefetch(&S, n); st_prefetch_rep(&S, n, (int)a); printf("prefetch\n");
        } else if (!strcmp(op, "noreadahead")) { if (fscanf(f, "%lld", &a) != 1) return 3; st_no_readahead(&S, S.fds[a]); printf("noreadahead\n");
        } else if (!strcmp(op, "map")) {
            if (fscanf(f, "%2047s", nh) != 1) return 3; char *n = unhex(nh); st_mapped_raw m;
            int r = st_map_raw(&S, n, &m);
            if (r == 0) { printf("map %lld %016llx\n", (long long)m.nbytes, (unsigned long long)fnv(m.data, (size_t)m.nbytes)); st_unmap_raw(&m); } else printf("map failed\n");
        } else if (!strcmp(op, "shardmap")) {   /* shardmap SHARD off nbytes */
            if (fscanf(f, "%lld %lld %lld", &a, &b, &c) != 3) return 3;
            int fd = (a >= 0 && a < S.nfd) ? S.fds[a] : -7;
            const void *p = st_map_shard_range(fd, b, c);
            printf("shardmap %s %016llx\n", p ? "yes" : "no", (unsigned long long)(p ? fnv(p, (size_t)c) : 0));
        } else if (!strcmp(op, "missing")) { if (fscanf(f, "%2047s", nh) != 1) return 3; char *n = unhex(nh); if (st_has(&S, n)) printf("present\n"); else st_read_raw(&S, n, NULL, 0);
        } else { fprintf(stderr, "unknown op %s\n", op); return 4; }
    }
    return 0;
}
