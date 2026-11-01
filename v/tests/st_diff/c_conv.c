/* hashes of the exact f32 bit patterns st.h's converters give for every f16 / bf16 / e8m0 input */
#include "st.h"
int main(void){
    uint64_t hh[3] = {14695981039346656037ULL, 14695981039346656037ULL, 14695981039346656037ULL};
    for (uint32_t i = 0; i < 65536; i++) {
        float a = f16_to_f32((uint16_t)i), b = bf16_to_f32((uint16_t)i); uint32_t x, y; memcpy(&x, &a, 4); memcpy(&y, &b, 4);
        for (int k = 0; k < 4; k++) { hh[0] ^= (x >> (8*k)) & 255; hh[0] *= 1099511628211ULL; hh[1] ^= (y >> (8*k)) & 255; hh[1] *= 1099511628211ULL; }
    }
    for (uint32_t i = 0; i < 256; i++) { float e = ue8m0_to_f32((uint8_t)i); uint32_t z; memcpy(&z, &e, 4); for (int k = 0; k < 4; k++) { hh[2] ^= (z >> (8*k)) & 255; hh[2] *= 1099511628211ULL; } }
    printf("f16 %016llx\nbf16 %016llx\ne8m0 %016llx\n", (unsigned long long)hh[0], (unsigned long long)hh[1], (unsigned long long)hh[2]);
    return 0;
}
