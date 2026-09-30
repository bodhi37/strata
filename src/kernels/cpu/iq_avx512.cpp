// src/kernels/cpu/iq_avx512.cpp - plan v0.3 P6: the i-quant expert rows in 512-bit lanes, several tokens at once.
//
// ggml-cpu's x86 dot products for these formats are AVX2 and single-token: every token re-decodes the weights
// and a sign is applied with five instructions per 32 values.  Here 64 values are decoded once (8 or 16 grid
// lookups, one 64-bit sign mask, one vector of scales) and every token applies them with a masked subtract, a
// `maddubs` and a `madd`.  The arithmetic is ggml's (ggml-cpu/quants.c, the `_generic` references): integer sums
// per block, times d_x * d_y * the format's constant - only the order of the float additions differs.
//
// Formats: IQ2_XXS (16), IQ2_XS (17), IQ3_XXS (18), IQ3_S (21), IQ2_S (22).  IQ1_M stays on ggml-cpu.
#include "strata/kernels/cpu/iq_avx512.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>

#include <cmath>
#include <cstring>

namespace strata::kernels::cpu {
namespace {

inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }
inline uint32_t u32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
inline uint16_t u16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
inline uint64_t u64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }

// lanes 0-7 -> s0, 8-15 -> s1, 16-23 -> s2, 24-31 -> s3 (int16 lanes of a maddubs result: two values each)
inline __m512i scales4(int s0, int s1, int s2, int s3) {
    const __m512i idx = _mm512_set_epi16(3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,
                                         0, 0, 0, 0, 0, 0, 0, 0);
    const uint64_t packed = (uint64_t) (uint16_t) s0 | ((uint64_t) (uint16_t) s1 << 16) |
                            ((uint64_t) (uint16_t) s2 << 32) | ((uint64_t) (uint16_t) s3 << 48);
    return _mm512_permutexvar_epi16(idx, _mm512_set1_epi64((long long) packed));
}

// ---- per format: 64 values (chunk j of a 256-value block) -> grid magnitudes, sign mask, scales
template <int TY> struct Fmt;

template <> struct Fmt<16> {   // IQ2_XXS: d, qs[32] u16
    static constexpr int bytes = 66;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        const uint32_t a0 = u32(q), a1 = u32(q + 4), b0 = u32(q + 8), b1 = u32(q + 12);
        g = _mm512_set_epi64((long long) iq2xxs_grid[b0 >> 24], (long long) iq2xxs_grid[(b0 >> 16) & 255],
                             (long long) iq2xxs_grid[(b0 >> 8) & 255], (long long) iq2xxs_grid[b0 & 255],
                             (long long) iq2xxs_grid[a0 >> 24], (long long) iq2xxs_grid[(a0 >> 16) & 255],
                             (long long) iq2xxs_grid[(a0 >> 8) & 255], (long long) iq2xxs_grid[a0 & 255]);
        const uint64_t s = (uint64_t) ksigns_iq2xs[a1 & 127] | ((uint64_t) ksigns_iq2xs[(a1 >> 7) & 127] << 8) |
                           ((uint64_t) ksigns_iq2xs[(a1 >> 14) & 127] << 16) | ((uint64_t) ksigns_iq2xs[(a1 >> 21) & 127] << 24) |
                           ((uint64_t) ksigns_iq2xs[b1 & 127] << 32) | ((uint64_t) ksigns_iq2xs[(b1 >> 7) & 127] << 40) |
                           ((uint64_t) ksigns_iq2xs[(b1 >> 14) & 127] << 48) | ((uint64_t) ksigns_iq2xs[(b1 >> 21) & 127] << 56);
        m = _cvtu64_mask64(s);
        const int sa = 2 * (int) (a1 >> 28) + 1, sb = 2 * (int) (b1 >> 28) + 1;
        sc = scales4(sa, sa, sb, sb);
    }
};

template <> struct Fmt<17> {   // IQ2_XS: d, qs[32] u16 (9-bit grid index + 7-bit sign index), scales[8]
    static constexpr int bytes = 74;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        uint16_t v[8];
        std::memcpy(v, q, 16);
        g = _mm512_set_epi64((long long) iq2xs_grid[v[7] & 511], (long long) iq2xs_grid[v[6] & 511],
                             (long long) iq2xs_grid[v[5] & 511], (long long) iq2xs_grid[v[4] & 511],
                             (long long) iq2xs_grid[v[3] & 511], (long long) iq2xs_grid[v[2] & 511],
                             (long long) iq2xs_grid[v[1] & 511], (long long) iq2xs_grid[v[0] & 511]);
        uint64_t s = 0;
        for (int l = 0; l < 8; ++l) s |= (uint64_t) ksigns_iq2xs[v[l] >> 9] << (8 * l);
        m = _cvtu64_mask64(s);
        const uint8_t s0 = b[66 + 2 * j], s1 = b[66 + 2 * j + 1];
        sc = scales4(2 * (s0 & 15) + 1, 2 * (s0 >> 4) + 1, 2 * (s1 & 15) + 1, 2 * (s1 >> 4) + 1);
    }
};

template <> struct Fmt<22> {   // IQ2_S: d, qs[64] (32 grid bytes, 32 sign bytes), qh[8], scales[8]
    static constexpr int bytes = 82;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* qs = b + 2 + 8 * j;
        const uint8_t h0 = b[66 + 2 * j], h1 = b[66 + 2 * j + 1];
        g = _mm512_set_epi64((long long) iq2s_grid[qs[7] | ((h1 << 2) & 0x300)], (long long) iq2s_grid[qs[6] | ((h1 << 4) & 0x300)],
                             (long long) iq2s_grid[qs[5] | ((h1 << 6) & 0x300)], (long long) iq2s_grid[qs[4] | ((h1 << 8) & 0x300)],
                             (long long) iq2s_grid[qs[3] | ((h0 << 2) & 0x300)], (long long) iq2s_grid[qs[2] | ((h0 << 4) & 0x300)],
                             (long long) iq2s_grid[qs[1] | ((h0 << 6) & 0x300)], (long long) iq2s_grid[qs[0] | ((h0 << 8) & 0x300)]);
        m = _cvtu64_mask64(u64(b + 2 + 32 + 8 * j));
        const uint8_t s0 = b[74 + 2 * j], s1 = b[74 + 2 * j + 1];
        sc = scales4(2 * (s0 & 15) + 1, 2 * (s0 >> 4) + 1, 2 * (s1 & 15) + 1, 2 * (s1 >> 4) + 1);
    }
};

template <> struct Fmt<18> {   // IQ3_XXS: d, qs[64] grid bytes, 8 x u32 (4 x 7-bit sign index + 4-bit scale)
    static constexpr int bytes = 98;
    static constexpr float K = 0.25f;
    // R10 note: a vpgatherdd variant of this decode was tried and REVERTED - neutral at nt<=3 and 15% SLOWER
    // at nt=5 (register pressure against the five token accumulators).  Zen 5's gather does not beat 16
    // scalar L1 lookups here.  Measured dead end; do not retry without a profiler saying otherwise.
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        g = _mm512_set_epi32((int) iq3xxs_grid[q[15]], (int) iq3xxs_grid[q[14]], (int) iq3xxs_grid[q[13]], (int) iq3xxs_grid[q[12]],
                             (int) iq3xxs_grid[q[11]], (int) iq3xxs_grid[q[10]], (int) iq3xxs_grid[q[9]], (int) iq3xxs_grid[q[8]],
                             (int) iq3xxs_grid[q[7]], (int) iq3xxs_grid[q[6]], (int) iq3xxs_grid[q[5]], (int) iq3xxs_grid[q[4]],
                             (int) iq3xxs_grid[q[3]], (int) iq3xxs_grid[q[2]], (int) iq3xxs_grid[q[1]], (int) iq3xxs_grid[q[0]]);
        const uint32_t a = u32(b + 2 + 64 + 8 * j), c = u32(b + 2 + 64 + 8 * j + 4);
        const uint64_t s = (uint64_t) ksigns_iq2xs[a & 127] | ((uint64_t) ksigns_iq2xs[(a >> 7) & 127] << 8) |
                           ((uint64_t) ksigns_iq2xs[(a >> 14) & 127] << 16) | ((uint64_t) ksigns_iq2xs[(a >> 21) & 127] << 24) |
                           ((uint64_t) ksigns_iq2xs[c & 127] << 32) | ((uint64_t) ksigns_iq2xs[(c >> 7) & 127] << 40) |
                           ((uint64_t) ksigns_iq2xs[(c >> 14) & 127] << 48) | ((uint64_t) ksigns_iq2xs[(c >> 21) & 127] << 56);
        m = _cvtu64_mask64(s);
        const int sa = 2 * (int) (a >> 28) + 1, sb = 2 * (int) (c >> 28) + 1;
        sc = scales4(sa, sa, sb, sb);
    }
};

template <> struct Fmt<21> {   // IQ3_S: d, qs[64], qh[8], signs[32], scales[4]
    static constexpr int bytes = 110;
    static constexpr float K = 1.0f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        const uint32_t h0 = b[66 + 2 * j], h1 = b[66 + 2 * j + 1];
#define G3(k, h, kk) (int) iq3s_grid[q[k] | (((h >> kk) & 1) << 8)]
        g = _mm512_set_epi32(G3(15, h1, 7), G3(14, h1, 6), G3(13, h1, 5), G3(12, h1, 4),
                             G3(11, h1, 3), G3(10, h1, 2), G3(9, h1, 1), G3(8, h1, 0),
                             G3(7, h0, 7), G3(6, h0, 6), G3(5, h0, 5), G3(4, h0, 4),
                             G3(3, h0, 3), G3(2, h0, 2), G3(1, h0, 1), G3(0, h0, 0));
#undef G3
        m = _cvtu64_mask64(u64(b + 74 + 8 * j));
        const uint8_t s = b[106 + j];
        const int sa = 2 * (s & 15) + 1, sb = 2 * (s >> 4) + 1;
        sc = scales4(sa, sa, sb, sb);
    }
};

template <int TY, int NT>
inline void row_dot(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    __m512 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm512_setzero_ps();
    const __m512i zero = _mm512_setzero_si512();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * Fmt<TY>::bytes;
        __m512i acci[NT];
        for (int t = 0; t < NT; ++t) acci[t] = _mm512_setzero_si512();
        for (int j = 0; j < 4; ++j) {
            __m512i g, sc;
            __mmask64 m;
            Fmt<TY>::decode(blk, j, g, m, sc);
            for (int t = 0; t < NT; ++t) {
                const __m512i yv = _mm512_loadu_si512((const void*) (y[t][i].qs + 64 * j));
                const __m512i ys = _mm512_mask_sub_epi8(yv, m, zero, yv);
                acci[t] = _mm512_add_epi32(acci[t], _mm512_madd_epi16(_mm512_maddubs_epi16(g, ys), sc));
            }
        }
        const float dx = h2f(u16(blk)) * Fmt<TY>::K;
        for (int t = 0; t < NT; ++t)
            accf[t] = _mm512_fmadd_ps(_mm512_set1_ps(dx * y[t][i].d), _mm512_cvtepi32_ps(acci[t]), accf[t]);
    }
    for (int t = 0; t < NT; ++t) res[t] = _mm512_reduce_add_ps(accf[t]);
}

template <int TY, int NT>
void gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, float* const* ff,
             int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    const int nb = n / QK_K;
    float g[NT], u[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot<TY, NT>(blob + (size_t) r * gu_row, nb, y, g);
        row_dot<TY, NT>(blob + up_off + (size_t) r * gu_row, nb, y, u);
        for (int t = 0; t < NT; ++t) ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
    }
}

template <int TY, int NT>
void dot_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot<TY, NT>(w + (size_t) r * row_bytes, n / QK_K, y, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}

template <int TY>
void gu_rows_nt(int nt, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                float* const* ff, int r0, int r1) {
    switch (nt) {
        case 1: gu_rows<TY, 1>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 2: gu_rows<TY, 2>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 3: gu_rows<TY, 3>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 4: gu_rows<TY, 4>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 5: gu_rows<TY, 5>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 6: gu_rows<TY, 6>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 7: gu_rows<TY, 7>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        default: gu_rows<TY, 8>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
    }
}

template <int TY>
void dot_rows_nt(int nt, const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0,
                 int r1) {
    switch (nt) {
        case 1: dot_rows<TY, 1>(w, row_bytes, n, act, out, r0, r1); break;
        case 2: dot_rows<TY, 2>(w, row_bytes, n, act, out, r0, r1); break;
        case 3: dot_rows<TY, 3>(w, row_bytes, n, act, out, r0, r1); break;
        case 4: dot_rows<TY, 4>(w, row_bytes, n, act, out, r0, r1); break;
        default: for (int t0 = 0; t0 < nt; t0 += 4) {
            const int k = nt - t0 < 4 ? nt - t0 : 4;
            dot_rows_nt<TY>(k, w, row_bytes, n, act + t0, out + t0, r0, r1);
        }
    }
}

}  // namespace

// ---- R10: multi-token IQ4_NL rows (the down projection of every native pack on this box) ----
//
// ggml-cpu's AVX2 kernel re-runs the nibble LUT for EVERY token: measured on this machine, one expert's
// down projection costs 87 us at nt=2 and 212 us at nt=5 - the decode is re-done per token, which is the
// whole difference.  Here one 32-value block decodes ONCE into |g| and sign(g), and every token costs one
// load, one vpsignb, one maddubs, one madd.  The per-block integer sums are exact (the same values ggml
// computes), so the only difference from ggml's kernel is the order of the float additions - the same
// class of difference the AVX-512 gate/up kernels already carry (measured rel ~1e-7, see native_expert_parity).
void iq4nl_rows_multi(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt, float* const* out,
                      int r0, int r1) {
    const int nb = n / 32;                 // block_q8_0 / block_iq4_nl: 32 values per block
    const __m128i m4 = _mm_set1_epi8(0x0f);
    const __m128i lut16 = _mm_loadu_si128((const __m128i*) (const void*) kvalues_iq4nl);
    const __m256i lut = _mm256_broadcastsi128_si256(lut16);
    const __m256i ones = _mm256_set1_epi16(1);
    for (int r = r0; r < r1; ++r) {
        const uint8_t* wrow = w + (size_t) r * row_bytes;
        __m256 accv[8];
        for (int t = 0; t < nt; ++t) accv[t] = _mm256_setzero_ps();
        for (int ib = 0; ib < nb; ++ib) {
            const uint8_t* wb = wrow + (size_t) ib * 18;   // block_iq4_nl: fp16 d + 16 packed nibbles
            const __m128i q4 = _mm_loadu_si128((const __m128i*) (const void*) (wb + 2));
            const __m128i lo = _mm_and_si128(q4, m4);
            const __m128i hi = _mm_and_si128(_mm_srli_epi16(q4, 4), m4);
            const __m256i g = _mm256_shuffle_epi8(lut, _mm256_set_m128i(hi, lo));   // 32 int8 magnitudes
            // R10 FIX: vpsignb(a,b) = a*sign(b), so sign(g,g) is |g|, NOT the sign vector - the first version
            // precomputed "gs = sign(g,g)" and then sy = sign(yv, gs) reduced to yv (sign of |g| is +1 or 0),
            // i.e. THE WEIGHT SIGNS WERE NEVER APPLIED (sum |w|*y instead of sum w*y; caught by
            // scratch/iq4nl_fuzz.cpp after the engine produced repetitive output - the vs-float parity at
            // 1.3e-2 could not see it under the activation noise).  The correct form needs no precomputed
            // sign vector at all: ggml's own mul_add_epi8 is maddubs(|x|, y*sign(x)).
            const __m256i ax = _mm256_abs_epi8(g);
            const float dx = h2f(u16(wb));
            for (int t = 0; t < nt; ++t) {
                const uint8_t* yb = (const uint8_t*) hq[t] + (size_t) ib * 34;   // block_q8_0: fp16 d + qs[32]
                const __m256i yv = _mm256_loadu_si256((const __m256i*) (const void*) (yb + 2));
                const __m256i sy = _mm256_sign_epi8(yv, g);                      // y * sign(g)
                const __m256i p16 = _mm256_maddubs_epi16(ax, sy);
                const __m256i p32 = _mm256_madd_epi16(p16, ones);
                const float dy = h2f(u16(yb));
                accv[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * dy), _mm256_cvtepi32_ps(p32), accv[t]);
            }
        }
        for (int t = 0; t < nt; ++t) {
            const __m128 s4 = _mm_add_ps(_mm256_castps256_ps128(accv[t]), _mm256_extractf128_ps(accv[t], 1));
            const __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
            out[t][r] = _mm_cvtss_f32(_mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1)));
        }
    }
}


// ---- IQ4_XS gate/up rows (Orca: gu type 23, Q8_K activations) ----
//
// block_iq4_xs: fp16 d + u16 scales_h + u8 scales_l[4] + u8 qs[128] = 136 B per 256 values
// (ggml-common.h).  8 sub-blocks of 32 values; sub-block ib's 6-bit scale is
//   nib = (scales_l[ib/2] >> (4*(ib&1))) & 0xF, high = (scales_h >> (4*(ib/2) + 2*(ib&1))) & 0x3,
//   ls = (nib | (high << 4)) - 32
// the same integers ggml_vec_dot_iq4_xs_q8_K's AVX2 path computes.  ggml re-runs the nibble
// LUT per token; here each sub-block decodes ONCE into int8 grid values and every token costs
// one load, one sign, one maddubs, one madd - the same decode-once structure as
// iq4nl_rows_multi above.  Per-block integer sums are exact; only float addition order
// differs from ggml.
template <int NT>
inline void row_dot_iq4xs(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
    const __m128i m4 = _mm_set1_epi8(0x0f);
    const __m128i lut16 = _mm_loadu_si128((const __m128i*) (const void*) kvalues_iq4nl);
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * 136;
        const float dx = h2f(u16(blk));
        const uint16_t sh = u16(blk + 2);
        const uint8_t* sl = blk + 4;
        const uint8_t* qs = blk + 8;
        __m256i acci[NT];
        for (int t = 0; t < NT; ++t) acci[t] = _mm256_setzero_si256();
        for (int ib = 0; ib < 8; ++ib) {
            const __m128i q16 = _mm_loadu_si128((const __m128i*) (const void*) (qs + 16 * ib));
            const __m128i lo = _mm_and_si128(q16, m4);
            const __m128i hi = _mm_and_si128(_mm_srli_epi16(q16, 4), m4);
            const __m128i glo = _mm_shuffle_epi8(lut16, lo);
            const __m128i ghi = _mm_shuffle_epi8(lut16, hi);
            const __m256i g =
                _mm256_inserti128_si256(_mm256_castsi128_si256(glo), ghi, 1);
            const __m256i ax = _mm256_abs_epi8(g);
            const int p = ib >> 1, s = ib & 1;
            const int nib = (sl[p] >> (4 * s)) & 0xF;
            const int hb = (sh >> (4 * p + 2 * s)) & 0x3;
            const __m256i vls = _mm256_set1_epi16((short) ((nib | (hb << 4)) - 32));
            for (int t = 0; t < NT; ++t) {
                const __m256i yv =
                    _mm256_loadu_si256((const __m256i*) (const void*) (y[t][i].qs + 32 * ib));
                const __m256i sy = _mm256_sign_epi8(yv, g);
                const __m256i p16 = _mm256_maddubs_epi16(ax, sy);
                acci[t] = _mm256_add_epi32(acci[t], _mm256_madd_epi16(p16, vls));
            }
        }
        for (int t = 0; t < NT; ++t)
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d),
                                      _mm256_cvtepi32_ps(acci[t]), accf[t]);
    }
    for (int t = 0; t < NT; ++t) {
        const __m128 s4 = _mm_add_ps(_mm256_castps256_ps128(accf[t]), _mm256_extractf128_ps(accf[t], 1));
        const __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
        res[t] = _mm_cvtss_f32(_mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1)));
    }
}

template <int NT>
void gu_rows_iq4xs(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                   float* const* ff, int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    const int nb = n / 256;
    float g[NT], u[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot_iq4xs<NT>(blob + (size_t) r * gu_row, nb, y, g);
        row_dot_iq4xs<NT>(blob + up_off + (size_t) r * gu_row, nb, y, u);
        for (int t = 0; t < NT; ++t) ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
    }
}

void iq4xs_gu_rows_nt(int nt, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                      float* const* ff, int r0, int r1) {
    switch (nt) {
        case 1: gu_rows_iq4xs<1>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 2: gu_rows_iq4xs<2>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 3: gu_rows_iq4xs<3>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 4: gu_rows_iq4xs<4>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 5: gu_rows_iq4xs<5>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 6: gu_rows_iq4xs<6>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 7: gu_rows_iq4xs<7>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 8: gu_rows_iq4xs<8>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        default: for (int t0 = 0; t0 < nt; t0 += 8) {
            const int k = nt - t0 < 8 ? nt - t0 : 8;
            iq4xs_gu_rows_nt(k, blob, gu_row, up_off, n, act + t0, ff + t0, r0, r1);
        }
    }
}


bool iq512_supported(int type) noexcept {
    return type == 16 || type == 17 || type == 18 || type == 21 || type == 22 || type == 23;
}

void iq512_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                   float* const* ff, int r0, int r1) {
    if (type == 23) {   // IQ4_XS (Orca gu): decode-once multi-token kernel above, Q8_K activations
        iq4xs_gu_rows_nt(nt, blob, gu_row, up_off, n, act, ff, r0, r1);
        return;
    }
    switch (type) {
        case 16: gu_rows_nt<16>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 17: gu_rows_nt<17>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 18: gu_rows_nt<18>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 21: gu_rows_nt<21>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 22: gu_rows_nt<22>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        default: break;
    }
}

void iq512_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                int r0, int r1) {
    switch (type) {
        case 16: dot_rows_nt<16>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 17: dot_rows_nt<17>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 18: dot_rows_nt<18>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 21: dot_rows_nt<21>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 22: dot_rows_nt<22>(nt, w, row_bytes, n, act, out, r0, r1); break;
        default: break;
    }
}

}  // namespace strata::kernels::cpu
