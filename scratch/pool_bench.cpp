// scratch/pool_bench.cpp - R10: isolate the CPU expert pool's real throughput from the engine.
//
// Reproduces the decode window's exact per-layer pattern: L layers, each with E distinct experts, each
// expert carrying NT tokens, run through ExpertPool::run_split_multi_native with the native (GGUF) formats
// of the real quants.  Reports ms/layer and effective GB/s, against the single-thread kernel rate, so the
// pool's scheduling overhead (wake stagger, task granularity) is separated from the kernel's speed.
//
//   build/pool_bench [gu_type] [d_type] [nt] [experts_per_layer] [layers] [pool_workers] [gap_ms] [pass0_frac]
//
// Defaults: 18 20 (IQ3_XXS gate/up + IQ4_NL down) 5 tokens, 17 experts/layer, 48 layers, 16 workers,
// gap 2.0 ms (the engine's GPU-attention + router time between layers), pass0_frac 0.7 (70% of the
// layer's experts are resident; the rest arrive after a further gap, modelling the two-pass dispatch).
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <algorithm>
#include <vector>

namespace c = strata::kernels::cpu;

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    const int gu_type = argc > 1 ? std::atoi(argv[1]) : 18;
    const int d_type = argc > 2 ? std::atoi(argv[2]) : 20;
    const int NT = argc > 3 ? std::atoi(argv[3]) : 5;
    const int E = argc > 4 ? std::atoi(argv[4]) : 17;
    const int L = argc > 5 ? std::atoi(argv[5]) : 48;
    const int W = argc > 6 ? std::atoi(argv[6]) : 16;
    const double gap_ms = argc > 7 ? std::atof(argv[7]) : 2.0;
    const double pass0_frac = argc > 8 ? std::atof(argv[8]) : 0.7;
    const double pass1_gap_ms = gap_ms * 0.75;   // the two-pass dispatch's read wait, folded into the gap
    const int64_t H = 2560, FF = 640;

    c::NativeFmt f;
    std::string err;
    if (!c::native_fmt(gu_type, d_type, H, FF, f, err)) {
        std::fprintf(stderr, "native_fmt: %s\n", err.c_str());
        return 2;
    }
    std::printf("fmt gu=%d d=%d blob=%.3f MB (gu_row=%zu d_row=%zu up_off=%zu down_off=%zu)\n",
                gu_type, d_type, f.bytes / 1e6, f.gu_row, f.d_row, f.up_off, f.down_off);

    const int NB = std::max(E * L, 64);           // one distinct blob per (layer, expert): DRAM-resident
    std::vector<uint8_t> blobs((size_t) NB * f.bytes);
    {
        std::mt19937 rng(1234);
        for (int b = 0; b < NB; ++b) {
            uint8_t* p = &blobs[(size_t) b * f.bytes];
            for (size_t i = 0; i < f.bytes; ++i) p[i] = (uint8_t) rng();
            // plausible fp16 scales at the head of every weight block, so the dot products neither
            // denormalise nor blow up: same trick expert_multi_test uses.
            for (size_t off = 0; off + 2 <= f.bytes; ) {
                p[off] = 0x00; p[off + 1] = 0x1c;   // ~0.004-0.012 fp16
                off += (gu_type == 18) ? 98 : (gu_type == 21) ? 110 : (gu_type == 23) ? 138 : 18;
            }
        }
    }

    // activations: NT distinct tokens, quantized once (the engine quantizes per token per layer; cheap)
    std::vector<float> x((size_t) NT * H);
    {
        std::mt19937 rng(99);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (float& v : x) v = nd(rng);
    }
    std::vector<uint8_t> nact((size_t) NT * c::kNativeActBytes);
    for (int t = 0; t < NT; ++t) c::native_quant_act(f, &x[(size_t) t * H], &nact[(size_t) t * c::kNativeActBytes]);
    // one DISTINCT output row per (expert, token): the engine writes h_ymiss_[tb*k*H + i*H], so experts never
    // share cache lines.  (An earlier version shared one row per token across experts - 17 cores
    // false-sharing the same 5 rows measured a phantom 2.8x parallelism loss.)
    std::vector<float> out((size_t) E * NT * H, 0.f);

    // ---- single-thread kernel rates (the pool's per-claim ceiling) ----
    {
        float* ffp[8];
        float ff[8][1024];
        for (int t = 0; t < NT; ++t) ffp[t] = ff[t];
        const void* a[8];
        for (int t = 0; t < NT; ++t) a[t] = &nact[(size_t) t * c::kNativeActBytes];
        c::native_gu_rows(f, &blobs[0], a, NT, ffp, 0, (int) FF);
        const int it = 30;
        const double t0 = now_ms();
        for (int i = 0; i < it; ++i) c::native_gu_rows(f, &blobs[0], a, NT, ffp, 0, (int) FF);
        const double gu_us = (now_ms() - t0) / it * 1000;
        std::vector<uint8_t> hq((size_t) NT * c::kNativeHBytes);
        for (int t = 0; t < NT; ++t) c::native_quant_h(f, ff[t], &hq[(size_t) t * c::kNativeHBytes]);
        const void* hp[8];
        for (int t = 0; t < NT; ++t) hp[t] = &hq[(size_t) t * c::kNativeHBytes];
        float* op[8];
        float oo[8][4096];
        for (int t = 0; t < NT; ++t) op[t] = oo[t];
        const double t1 = now_ms();
        for (int i = 0; i < it; ++i) c::native_down_rows(f, &blobs[0], hp, NT, op, 0, (int) H);
        const double d_us = (now_ms() - t1) / it * 1000;
        std::printf("single-thread: gu %.0f us (%.2f GB/s)  down %.0f us (%.2f GB/s)  blob %.0f us (%.2f GB/s)\n",
                    gu_us, 2.0 * f.up_off / gu_us / 1e3, d_us, (double) (f.bytes - f.down_off) / d_us / 1e3,
                    gu_us + d_us, (double) f.bytes / (gu_us + d_us) / 1e3);
    }

    // ---- the pool, engine pattern ----
    c::ExpertPool pool(W, true, true);
    std::vector<c::ExpertJobMulti> jobs((size_t) E);
    const void* acts[8];
    for (int t = 0; t < NT; ++t) acts[t] = &nact[(size_t) t * c::kNativeActBytes];
    // a fixed pseudo-random (layer, expert) -> blob map, so the access pattern is scattered like the LRU
    // tier's slots instead of sequential like a fresh vector
    std::vector<int> blob_of((size_t) L * E);
    {
        std::mt19937 rng(7);
        for (int i = 0; i < L * E; ++i) blob_of[(size_t) i] = i % NB;
        std::shuffle(blob_of.begin(), blob_of.end(), rng);
    }

    const int REPS = 3;
    auto gap = [](double ms) { if (ms > 0) std::this_thread::sleep_for(std::chrono::microseconds((int64_t) (ms * 1000))); };
    const int E0 = (int) (E * pass0_frac);      // resident experts (pass 0)
    for (int rep = 0; rep < REPS; ++rep) {
        const double t0 = now_ms();
        for (int l = 0; l < L; ++l) {
            gap(gap_ms);                        // GPU attention + router, pool idle
            int idx = 0;
            for (int e = 0; e < E0; ++e, ++idx) {
                c::ExpertJobMulti& j = jobs[(size_t) idx];
                j.blob = &blobs[(size_t) blob_of[(size_t) (l * E + idx)] * f.bytes];
                j.nt = NT;
                for (int t = 0; t < NT; ++t) { j.nact[t] = acts[t]; j.out[t] = &out[(size_t) (idx * NT + t) * H]; }
            }
            if (E0 > 0) pool.run_split_multi_native(f, jobs.data(), E0);
            gap(pass1_gap_ms);                 // the read wait the resident work hides
            const int E1 = E - E0;
            for (int e = 0; e < E1; ++e, ++idx) {
                c::ExpertJobMulti& j = jobs[(size_t) idx];
                j.blob = &blobs[(size_t) blob_of[(size_t) (l * E + idx)] * f.bytes];
                j.nt = NT;
                for (int t = 0; t < NT; ++t) { j.nact[t] = acts[t]; j.out[t] = &out[(size_t) (idx * NT + t) * H]; }
            }
            if (E1 > 0) pool.run_split_multi_native(f, jobs.data() + E0, E1);
        }
        const double ms = now_ms() - t0;
        const double bytes = (double) L * E * f.bytes;
        const double per_layer = ms / L;
        std::printf("rep %d: %d layers x %d experts x nt=%d (gap %.1fms, pass0 %d/%d): %.2f ms/layer, "
                    "blob %.2f GB/s, window %.0f ms / %.2f GB, pure-pool %.0f%%\n",
                    rep, L, E, NT, gap_ms, E0, E, per_layer,
                    bytes / ms / 1e6, ms, bytes / 1e9, 100.0 * (1.0 - (gap_ms + pass1_gap_ms) / per_layer));
        std::printf("     pool phases per window: gu %.1f ms  q %.1f ms  down %.1f ms  (drain %.1f ms)\n",
                    pool.ms_multi_gu / L * L / L, pool.ms_multi_q / L, pool.ms_multi_down / L,
                    (pool.ms_multi_gu + pool.ms_multi_q + pool.ms_multi_down) / L);
    }
    double wg, wq, wd, wsum = 0, drain = 0, wpark = 0;
    // totals are lifetime; read them for the report
    return 0;
}
