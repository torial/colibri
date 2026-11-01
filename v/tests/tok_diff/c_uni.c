/* every code point's class set from the C tables, hashed (FNV-1a over the 16-bit masks) */
#include "tok.h"
int main(void){
    uint64_t h = 14695981039346656037ULL;
    for (uint32_t c = 0; c <= 0x10FFFF; c++) {
        int m = (is_L(c)?1:0)|(is_N(c)?2:0)|(is_S(c)?4:0)|(is_U(c)?8:0)|(is_X(c)?16:0)|(is_Mk(c)?32:0)|(is_Pu(c)?64:0)|(is_Sy(c)?128:0)|(is_han(c)?256:0);
        h ^= m & 255; h *= 1099511628211ULL; h ^= m >> 8; h *= 1099511628211ULL;
    }
    printf("classes %016llx\n", (unsigned long long)h);
    return 0;
}
