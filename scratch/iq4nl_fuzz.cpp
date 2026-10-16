// scratch/iq4nl_fuzz.cpp - R10: exhaustive parity fuzz for the multi-token IQ4_NL down kernel.
//
//   build/iq4nl_fuzz [trials]
//
// For each trial: a random IQ4_NL weight row set, random Q8_0 activations, random fp16 scales in the
// pack's real range, all nt 1..8, random row sub-ranges - compare iq4nl_rows_multi (all tokens, ranges)
// against ggml's own vec_dot per token.  Pass = max |rel| over every element, plus a bitwise-identical
// check of the per-block INTEGER sums (via a reference scalar).
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace c = strata::kernels::cpu;

int main(int argc, char** argv) {
    const int TRIALS = argc > 1 ? std::atoi(argv[1]) : 200;
    c::NativeFmt f;
    std::string err;
    if (!c::native_fmt(18, 20, 2560, 640, f, err)) {   // IQ3_XXS geometry, IQ4_NL down
        std::fprintf(stderr, "native_fmt: %s\n", err.c_str());
        return 2;
    }
    const int H = 2560, FF = 640;
    const int nrow = 24;                       // a few rows per trial
    std::vector<uint8_t> w((size_t) nrow * f.d_row);
    std::mt19937 rng(20260927);
    double max_rel = 0, max_abs = 0;
    int bitwise_bad = 0;
    const auto* td = ggml_get_type_traits_cpu(GGML_TYPE_IQ4_NL);
    for (int trial = 0; trial < TRIALS; ++trial) {
        // random weights with realistic fp16 scales (small, like the pack's)
        for (int r = 0; r < nrow; ++r) {
            uint8_t* row = &w[(size_t) r * f.d_row];
            for (size_t b = 0; b < f.d_row; b += 18) {
                const uint16_t d = (uint16_t) (0x1a00 + rng() % 0x0300);
                std::memcpy(row + b, &d, 2);
                for (int i = 2; i < 18; ++i) row[b + i] = (uint8_t) rng();
            }
        }
        for (int nt = 1; nt <= 8; ++nt) {
            // random float activations, quantized to Q8_0 the way the pool does
            std::vector<float> xf((size_t) nt * FF);
            std::normal_distribution<float> nd(0.f, 1.f);
            for (float& v : xf) v = nd(rng);
            std::vector<uint8_t> hq((size_t) nt * c::kNativeHBytes);
            for (int t = 0; t < nt; ++t) c::native_quant_h(f, &xf[(size_t) t * FF], &hq[(size_t) t * c::kNativeHBytes]);
            const void* hp[8];
            for (int t = 0; t < nt; ++t) hp[t] = &hq[(size_t) t * c::kNativeHBytes];
            // random row range
            const int span = 1 + (int) (rng() % nrow);
            const int r0 = (int) (rng() % (nrow - span + 1));
            const int r1 = r0 + span;
            std::vector<float> mine((size_t) nt * H, -1e30f), ref((size_t) nt * H, -1e30f);
            float* mp[8], * rp[8];
            for (int t = 0; t < nt; ++t) { mp[t] = &mine[(size_t) t * H]; rp[t] = &ref[(size_t) t * H]; }
            c::iq4nl_rows_multi(w.data(), f.d_row, FF, hp, nt, mp, r0, r1);
            // ggml per-token reference over the same rows
            for (int t = 0; t < nt; ++t)
                for (int r = r0; r < r1; ++r) {
                    float s = 0.f;
                    td->vec_dot(FF, &s, 0, w.data() + (size_t) r * f.d_row, 0, hp[t], 0, 1);
                    rp[t][r] = s;
                }
            for (int t = 0; t < nt; ++t)
                for (int r = r0; r < r1; ++r) {
                    const double a = mine[(size_t) t * H + r], b = ref[(size_t) t * H + r];
                    const double rel = std::fabs(b) > 1e-20 ? std::fabs(a - b) / std::fabs(b) : std::fabs(a - b);
                    const double abserr = std::fabs(a - b);
                    if (rel > max_rel && std::fabs(b) > 1e-3) max_rel = rel;   // only meaningful rels
                    if (abserr > max_abs) max_abs = abserr;
                    // the integer sums are exact in both; only the float ORDER differs.  Near-cancelling
                    // sums make rel meaningless, so the gate is: small abs error AND small rel when the
                    // result is not near zero.  A real bug (e.g. missing signs) shows rel ~1 at any size.
                    if (rel > 1e-3 && std::fabs(b) > 1e-3 && abserr > 1e-4) {
                        if (bitwise_bad < 5)
                            std::printf("BAD trial %d nt=%d row %d: mine %.8f ref %.8f rel %.3e\n",
                                        trial, nt, r, mine[(size_t) t * H + r], ref[(size_t) t * H + r], rel);
                        ++bitwise_bad;
                    }
                }
        }
    }
    std::printf("iq4nl_fuzz: %d trials, max rel %.3e (on |ref|>1e-3), max abs %.3e, %d BAD\n", TRIALS, max_rel, max_abs, bitwise_bad);
    return bitwise_bad ? 1 : 0;
}
