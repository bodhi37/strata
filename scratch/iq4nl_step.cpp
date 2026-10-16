#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "../src/kernels/cpu/iq_avx512.cpp"   // single TU, debuggable
#include <cstdio>
#include <cstring>
namespace c = strata::kernels::cpu;
int main() {
    ggml_cpu_init();
    uint8_t wblk[18];
    const uint16_t d = 0x3400;
    std::memcpy(wblk, &d, 2);
    for (int i = 0; i < 16; ++i) wblk[2 + i] = (uint8_t) i;
    uint8_t yblk[34];
    const uint16_t dy = 0x3c00;
    std::memcpy(yblk, &dy, 2);
    for (int i = 0; i < 32; ++i) yblk[2 + i] = (int8_t)(i - 16);
    // manual single-block computation, mirroring the kernel's steps with prints
    const __m128i m4 = _mm_set1_epi8(0x0f);
    const __m128i lut16 = _mm_loadu_si128((const __m128i*) (const void*) kvalues_iq4nl);
    const __m256i lut = _mm256_broadcastsi128_si256(lut16);
    const __m128i q4 = _mm_loadu_si128((const __m128i*) (const void*) (wblk + 2));
    const __m128i lo = _mm_and_si128(q4, m4);
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(q4, 4), m4);
    const __m256i idx = _mm256_set_m128i(hi, lo);
    const __m256i g = _mm256_shuffle_epi8(lut, idx);
    const __m256i ax = _mm256_abs_epi8(g);
    const __m256i gs = _mm256_sign_epi8(g, g);
    alignas(32) int8_t gb[32], ab[32], sb[32];
    _mm256_store_si256((__m256i*) gb, g); _mm256_store_si256((__m256i*) ab, ax); _mm256_store_si256((__m256i*) sb, gs);
    std::printf("g   : "); for (int i=0;i<32;i++) std::printf("%4d", gb[i]); std::printf("\n");
    std::printf("ax  : "); for (int i=0;i<32;i++) std::printf("%4d", (int)(uint8_t)ab[i]); std::printf("\n");
    std::printf("gs  : "); for (int i=0;i<32;i++) std::printf("%4d", sb[i]); std::printf("\n");
    const __m256i yv = _mm256_loadu_si256((const __m256i*) (const void*) (yblk + 2));
    const __m256i sy = _mm256_sign_epi8(yv, gs);
    const __m256i p16 = _mm256_maddubs_epi16(ax, sy);
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i p32 = _mm256_madd_epi16(p16, ones);
    alignas(32) int16_t p16b[16]; alignas(32) int32_t p32b[8]; alignas(32) int8_t syb[32];
    _mm256_store_si256((__m256i*) syb, sy); _mm256_store_si256((__m256i*) p16b, p16); _mm256_store_si256((__m256i*) p32b, p32);
    std::printf("sy  : "); for (int i=0;i<32;i++) std::printf("%4d", syb[i]); std::printf("\n");
    std::printf("p32 : "); for (int i=0;i<8;i++) std::printf("%8d", p32b[i]); std::printf("\n");
    long long sum = 0; for (int i=0;i<8;i++) sum += p32b[i];
    std::printf("sum of p32 = %lld, times d 0.25 = %.6f\n", sum, (double) sum * 0.25);
    float mine = -1e30f;
    float* mp[1] = { &mine };
    const void* hp[1] = { yblk };
    c::iq4nl_rows_multi(wblk, 18, 32, hp, 1, mp, 0, 1);
    std::printf("kernel mine %.6f\n", mine);
    // scalar
    double sref = 0;
    for (int j = 0; j < 16; ++j) {
        sref += (double) (int8_t) yblk[2 + j] * kvalues_iq4nl[wblk[2 + j] & 0xf];
        sref += (double) (int8_t) yblk[2 + 16 + j] * kvalues_iq4nl[wblk[2 + j] >> 4];
    }
    std::printf("scalar ref %.6f\n", sref * 0.25);
    return 0;
}
