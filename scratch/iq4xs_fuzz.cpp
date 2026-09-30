// scratch/iq4xs_fuzz.cpp - parity fuzz for the multi-token IQ4_XS gate/up kernel.
//
//   <build>/iq4xs_fuzz [trials]
//
// For each trial: random IQ4_XS gate/up rows (Orca geometry: n_embd 2560, type 23),
// random Q8_K activations via native_quant_act, all nt 1..8, random row sub-ranges -
// compare iq4xs_gu_rows_nt (all tokens, ranges, with SiLU) against ggml's own vec_dot
// per token per row. Pass = small rel/abs error (float addition order only).
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
    if (!c::native_fmt(23, 20, 2560, 640, f, err)) {   // Orca: IQ4_XS gate/up, IQ4_NL down
        std::fprintf(stderr, "native_fmt: %s\n", err.c_str());
        return 2;
    }
    std::printf("gu_row=%zu up_off=%zu (expect 1360 / 870400)\n", f.gu_row, f.up_off);
    const int H = 2560;
    const int nrow = 16;                       // a few expert rows per trial
    std::vector<uint8_t> blob((size_t) nrow * f.gu_row + f.up_off + (size_t) nrow * f.gu_row);
    std::mt19937 rng(20260930);
    double max_rel = 0, max_abs = 0;
    int bad = 0;
    const auto* tg = ggml_get_type_traits_cpu((ggml_type) 23);
    if (!tg || !tg->vec_dot) { std::fprintf(stderr, "no ggml vec_dot for 23\n"); return 2; }
    for (int trial = 0; trial < TRIALS; ++trial) {
        // random bytes everywhere, then realistic small d per 256-block for gate and up rows
        for (size_t b = 0; b < blob.size(); ++b) blob[b] = (uint8_t) rng();
        for (int r = 0; r < nrow; ++r) {
            // realistic small d per 256-block for gate and up rows
            for (int base = 0; base < 2; ++base) {
                uint8_t* blk0 = blob.data() + (base ? f.up_off : 0) + (size_t) r * f.gu_row;
                for (int ibl = 0; ibl < H / 256; ++ibl) {
                    const uint16_t d = (uint16_t) (0x1c00 + rng() % 0x0400);
                    std::memcpy(blk0 + (size_t) ibl * 136, &d, 2);
                }
            }
        }
        for (int nt = 1; nt <= 8; ++nt) {
            std::vector<float> xf((size_t) nt * H);
            std::normal_distribution<float> nd(0.f, 1.f);
            for (float& v : xf) v = nd(rng);
            std::vector<uint8_t> aq((size_t) nt * c::kNativeActBytes);
            for (int t = 0; t < nt; ++t) c::native_quant_act(f, &xf[(size_t) t * H], &aq[(size_t) t * c::kNativeActBytes]);
            const void* ap[8];
            for (int t = 0; t < nt; ++t) ap[t] = &aq[(size_t) t * c::kNativeActBytes];
            const int span = 1 + (int) (rng() % nrow);
            const int r0 = (int) (rng() % (nrow - span + 1));
            const int r1 = r0 + span;
            std::vector<float> mine((size_t) nt * H, -1e30f), ref((size_t) nt * H, -1e30f);
            float* mp[8], * rp[8];
            for (int t = 0; t < nt; ++t) { mp[t] = &mine[(size_t) t * H]; rp[t] = &ref[(size_t) t * H]; }
            c::iq4xs_gu_rows_nt(nt, blob.data(), f.gu_row, f.up_off, H, ap, mp, r0, r1);
            for (int t = 0; t < nt; ++t)
                for (int r = r0; r < r1; ++r) {
                    float g = 0.f, u = 0.f;
                    tg->vec_dot(H, &g, 0, blob.data() + (size_t) r * f.gu_row, 0, ap[t], 0, 1);
                    tg->vec_dot(H, &u, 0, blob.data() + f.up_off + (size_t) r * f.gu_row, 0, ap[t], 0, 1);
                    rp[t][r] = (g / (1.f + std::exp(-g))) * u;
                }
            for (int t = 0; t < nt; ++t)
                for (int r = r0; r < r1; ++r) {
                    const double a = mine[(size_t) t * H + r], b = ref[(size_t) t * H + r];
                    const double rel = std::fabs(b) > 1e-20 ? std::fabs(a - b) / std::fabs(b) : std::fabs(a - b);
                    const double abserr = std::fabs(a - b);
                    if (rel > max_rel && std::fabs(b) > 1e-3) max_rel = rel;
                    if (abserr > max_abs) max_abs = abserr;
                    if (rel > 1e-3 && std::fabs(b) > 1e-3 && abserr > 1e-4) {
                        if (bad < 5)
                            std::printf("BAD trial %d nt=%d row %d: mine %.8f ref %.8f rel %.3e\n",
                                        trial, nt, r, mine[(size_t) t * H + r], ref[(size_t) t * H + r], rel);
                        ++bad;
                    }
                }
        }
    }
    std::printf("iq4xs_fuzz: %d trials, max rel %.3e (on |ref|>1e-3), max abs %.3e, %d BAD\n", TRIALS, max_rel, max_abs, bad);
    return bad ? 1 : 0;
}
