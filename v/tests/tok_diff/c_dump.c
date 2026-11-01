/* C side of the tokenizer differential test: encode and decode every text file with c/tok.h.
 * Build: gcc -O1 -I../../../c -o c_dump c_dump.c -lm   usage: c_dump tokenizer.json text1 text2 ... */
#include "tok.h"
static void hex(const char *s, int n){ for (int i = 0; i < n; i++) printf("%02x", (unsigned char)s[i]); }
int main(int argc, char **argv){
    Tok T; tok_load(&T, argv[1]);
    for (int a = 2; a < argc; a++) {
        FILE *f = fopen(argv[a], "rb"); if (!f) return 2;
        fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
        char *t = malloc(n + 1); if (fread(t, 1, n, f) != (size_t)n) return 3; fclose(f);
        int *ids = malloc(sizeof(int) * (n * 4 + 16)); int m = tok_encode(&T, t, (int)n, ids, (int)(n * 4 + 16));
        printf("E"); for (int i = 0; i < m; i++) printf(" %d", ids[i]); printf("\n");
        char *buf = malloc(n * 8 + 64); int d = tok_decode(&T, ids, m, buf, (int)(n * 8 + 64));
        printf("D "); hex(buf, d); printf("\n");
        free(t); free(ids); free(buf);
    }
    printf("I %d %d\n", tok_id_of(&T, "<|endoftext|>"), tok_id_of(&T, "<think>"));
    return 0;
}
