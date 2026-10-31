/* dpbusd512.h -- vpdpbusd on AVX-512BW CPUs without VNNI (shared by idot.h, expert_ffn.h,
 * lowbit_g16.h). See the block below for the saturation rule every caller must meet. */
#ifndef COLI_DPBUSD512_H
#define COLI_DPBUSD512_H
#if defined(__AVX512BW__)
#include <immintrin.h>
#endif
/* ---- AVX-512BW without VNNI (Skylake-SP / Xeon Scalable 1st gen, e.g. Gold 6132) ------
 * Those CPUs have 512-bit integer ALUs but not vpdpbusd, so before this tier they ran the
 * 256-bit AVX2 kernels at half the vector width. coli_dpbusd512 is vpdpbusd's value
 * computed as vpmaddubsw + vpmaddwd(ones) + vpaddd. vpmaddubsw saturates its int16 pair
 * sums, so it is only used where the pair sum provably fits (|pair| <= 32767), the very
 * condition the AVX2 kernels already rely on:
 *   unsigned operand abs(w) <= 128 (int8 rows) or <= 15 (int4 nibbles), signed operand
 *   |x| <= 127 (qrow_i8's contract)  ->  |pair| <= 2*128*127 = 32512.
 * The kernels with a biased uint8 operand up to 255 (i8r16 rows) are NOT routed here: they
 * keep their AVX2 form (xor 0x80 + sign), where the same product would saturate.
 * COLI_NO_AVX512BW_IDOT=1 at build time compiles this tier out (back to AVX2). */
#if defined(__AVX512BW__) && !defined(__AVX512VNNI__) && !defined(COLI_NO_AVX512BW_IDOT)
#define COLI_IDOT_AVX512BW 1
#endif
#if (defined(__AVX512VNNI__) && defined(__AVX512BW__)) || defined(COLI_IDOT_AVX512BW)
#define COLI_IDOT512 1
#else
#define COLI_IDOT512 0
#endif
#if defined(COLI_IDOT_AVX512BW)
static inline __m512i coli_dpbusd512(__m512i acc, __m512i a_u8, __m512i b_s8){
    return _mm512_add_epi32(acc,_mm512_madd_epi16(_mm512_maddubs_epi16(a_u8,b_s8),_mm512_set1_epi16(1)));
}
#elif COLI_IDOT512
#define coli_dpbusd512 _mm512_dpbusd_epi32
#endif
#endif /* COLI_DPBUSD512_H */
