/* The C side of the json differential test: the same canonical dump as v_dump.v, from c/json.h.
 * Build: gcc -O1 -I../../../c -o c_dump c_dump.c -lm */
#include "json.h"
#include <stdint.h>
static void hex(const char *s){ for (; *s; s++) printf("%02x", (unsigned char)*s); }
static void dump(jval *v){
    switch (v->t) {
    case J_NULL: printf(" N"); break;
    case J_BOOL: printf(" B%d", v->boolean ? 1 : 0); break;
    case J_NUM: { uint64_t b; memcpy(&b, &v->num, 8); printf(" D%016llx", (unsigned long long)b); break; }
    case J_STR: printf(" S"); hex(v->str); break;
    case J_ARR: printf(" A%d", v->len); for (int i = 0; i < v->len; i++) dump(v->kids[i]); break;
    case J_OBJ: printf(" O%d", v->len); for (int i = 0; i < v->len; i++) { printf(" K"); hex(v->keys[i]); dump(v->kids[i]); } break;
    }
}
int main(int argc, char **argv){
    for (int a = 1; a < argc; a++) {
        FILE *f = fopen(argv[a], "rb"); if (!f) return 2;
        fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
        char *t = malloc(n + 1); if (fread(t, 1, n, f) != (size_t)n) return 3; t[n] = 0; fclose(f);
        const char *base = strrchr(argv[a], '/'); printf("#%s\n", base ? base + 1 : argv[a]);
        jval *v = json_parse(t, NULL);
        printf("legacy"); dump(v); printf("\n"); json_free(v);
        jval *c = json_parse_checked(t);
        if (!c) printf("checked NULL\n"); else { printf("checked"); dump(c); printf("\n"); json_free(c); }
        free(t);
    }
    return 0;
}
