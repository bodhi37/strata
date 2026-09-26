// src/core/expert_source.cpp - the adapter.  See the header for the three clauses of the contract.
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include "strata/core/pinned.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <immintrin.h>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strata::core {

// ================================ THE FILE-BACKED SOURCE ================================

FileExpertSource::~FileExpertSource() { close(); }

bool FileExpertSource::open(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, std::string& err) {
    close();
    if (n_layers <= 0 || n_expert <= 0) { err = "FileExpertSource: the geometry is empty"; return false; }
    n_expert_ = n_expert;
    blobs_ = n_layers * n_expert;
    const uint64_t want = (uint64_t) blobs_ * (uint64_t) strata::kernels::cpu::BLOB;
    const std::string path = pack_dir + "/experts.bin";

#if defined(_WIN32)
    // UTF-8 -> UTF-16: the pack may live under a path with non-ASCII characters, and `CreateFileA` would
    // silently mangle it into a file-not-found.
    const int wide = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::vector<wchar_t> wpath((size_t) (wide > 0 ? wide : 1));
    if (wide > 0) MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wide);
    // **`FILE_FLAG_RANDOM_ACCESS` WAS HERE AND IT COST 14x.**
    //
    // The design depends on the OS page cache holding the whole 34 GB expert set, because this machine has
    // 64 GB of DDR5 and `L9` measured the CPU path at 44.14 GB/s from DRAM.  `FILE_FLAG_RANDOM_ACCESS` tells
    // the cache manager the opposite: it disables read-ahead AND it lets the manager drop the pages again
    // quickly, on the assumption that a large randomly-accessed file will not be re-read.  Measured, on
    // `strata generate --max-new 24`: **1.93 GB/s** - disk speed, 344 ms/token, and it never warmed up over 25
    // tokens, because the pages were being evicted as fast as they were faulted in.
    //
    // The correct flag is NO flag.  The access pattern IS random (10 of 512 experts per layer, a different 10
    // each layer), but every byte read is read again on the next token, so retention is the whole game.
    HANDLE f = CreateFileW(wpath.data(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) {
        err = "FileExpertSource: cannot open " + path;
        return false;
    }
    LARGE_INTEGER sz{};
    if (!GetFileSizeEx(f, &sz)) {
        CloseHandle(f);
        err = "FileExpertSource: cannot size " + path;
        return false;
    }
    if ((uint64_t) sz.QuadPart != want) {
        char buf[400];
        std::snprintf(buf, sizeof buf,
                      "FileExpertSource: %s is %llu B but %lld layers x %lld experts x %d B is %llu B - this "
                      "is not the pack this geometry came from",
                      path.c_str(), (unsigned long long) sz.QuadPart, (long long) n_layers,
                      (long long) n_expert, (int) strata::kernels::cpu::BLOB, (unsigned long long) want);
        CloseHandle(f);
        err = buf;
        return false;
    }
    HANDLE m = CreateFileMappingW(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (m == nullptr) {
        CloseHandle(f);
        err = "FileExpertSource: CreateFileMapping failed on " + path;
        return false;
    }
    void* view = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    if (view == nullptr) {
        CloseHandle(m);
        CloseHandle(f);
        err = "FileExpertSource: MapViewOfFile failed on " + path;
        return false;
    }
    file_ = f;
    mapping_ = m;
    base_ = (const uint8_t*) view;
#else
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) { err = "FileExpertSource: cannot open " + path; return false; }
    struct stat st{};
    if (fstat(fd, &st) != 0) { ::close(fd); err = "FileExpertSource: cannot stat " + path; return false; }
    if ((uint64_t) st.st_size != want) {
        char buf[400];
        std::snprintf(buf, sizeof buf,
                      "FileExpertSource: %s is %llu B but %lld layers x %lld experts x %d B is %llu B - this "
                      "is not the pack this geometry came from",
                      path.c_str(), (unsigned long long) st.st_size, (long long) n_layers, (long long) n_expert,
                      (int) strata::kernels::cpu::BLOB, (unsigned long long) want);
        ::close(fd);
        err = buf;
        return false;
    }
    void* view = mmap(nullptr, (size_t) want, PROT_READ, MAP_SHARED, fd, 0);
    if (view == MAP_FAILED) { ::close(fd); err = "FileExpertSource: mmap failed on " + path; return false; }
    fd_ = fd;
    base_ = (const uint8_t*) view;
#endif
    return true;
}

void FileExpertSource::close() {
#if defined(_WIN32)
    if (base_ != nullptr) UnmapViewOfFile((LPCVOID) base_);
    if (mapping_ != nullptr) CloseHandle((HANDLE) mapping_);
    if (file_ != nullptr) CloseHandle((HANDLE) file_);
    mapping_ = nullptr;
    file_ = nullptr;
#else
    if (base_ != nullptr) munmap((void*) base_, (size_t) blobs_ * (size_t) strata::kernels::cpu::BLOB);
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
#endif
    base_ = nullptr;
    blobs_ = 0;
    reads_ = 0;
}

const uint8_t* FileExpertSource::blob(int64_t layer, int64_t expert) {
    if (base_ == nullptr) return nullptr;
    if (layer < 0 || expert < 0) return nullptr;
    // **`expert >= n_expert_` IS CHECKED SEPARATELY, AND THE FLAT INDEX ALONE DOES NOT CATCH IT.**  The blob
    // index is `layer * n_expert + expert`, so `blob(0, 512)` has flat index 512 - which is in range, and is
    // `blob(1, 0)`.  A router id one past the end of a layer would then read the NEXT LAYER's first expert:
    // finite, correctly sized, and wrong.  Layer and expert are separate axes and are validated as such.
    if (expert >= n_expert_) return nullptr;
    const int64_t i = layer * n_expert_ + expert;
    if (i >= blobs_) return nullptr;
    ++reads_;
    return base_ + (size_t) i * strata::kernels::cpu::BLOB;
}

// ================================ THE ADAPTER ================================

void expert_pool_dispatch(void* user, const float* x_f, const int32_t* ids, const float* weights, int64_t n_embd,
                          int64_t k, float* out) {
    (void) weights;   // clause 2: `moe_combine` applies it on the device.  Not an oversight.
    ExpertDispatch& d = *(ExpertDispatch*) user;
    if (d.failed) return;   // a previous layer already failed; do not make it worse

    using namespace strata::kernels::cpu;
    // Clause 3: the blob's internal offsets are compile-time constants, so a mismatched geometry does not
    // produce a wrong answer - it produces a walk off the end of the blob into the next expert's bytes, which
    // is finite and plausible.  Refuse, name the number, and let the driver report it.
    if (n_embd != H) {
        d.failed = true;
        d.fail = "the expert kernel is compiled for a 2560-wide activation";
        d.fail_layer = d.layers;
        return;
    }
    if (expert_layout().native) {
        // plan v0.3 P6: a native pack runs its experts in verify windows only (the driver guarantees it)
        d.failed = true;
        d.fail = "the single-token expert path does not take a native (IQ) pack";
        d.fail_layer = d.layers;
        return;
    }
    if (k > (int64_t) d.jobs.size()) d.jobs.resize((size_t) k);

    d.src->begin_layer(d.layers, ids, k);
    // R7: this path has nothing to interleave - one token, one batch - so it takes the old single-phase shape.
    d.src->wait_layer();

    // Clause 1: rebuilt from `x_f` on EVERY call.  `x_f` is mapped pinned memory whose address never changes,
    // so anything cached against it would be layer 0's activation reused 48 times.
    act_quant_q8_1(x_f, H, d.act);

    // ---- R4.2c: THE POOL'S HALF OF THE SPLIT.  **IT DOES NOT DECIDE ANYTHING - `Launch` ALREADY DID.**
    //
    // The decision has to be made on THIS layer's ids, and `Launch` is the only callback that runs before the
    // pool while the ids are known (the doorbell publishes them when the ring fires).  So `expert_hit_run`
    // decides, and this consumes `d.is_hit`.  The first version decided here instead, which meant `Launch`
    // computed the PREVIOUS layer's experts into this layer's rows: C1 went from mean KL 9.69e-02 to 1.03e+00.
    //
    // `njobs` indexes the JOB ARRAY and `i` indexes the OUTPUT - they are the same only when nothing is a hit.
    const bool graph_hits = d.host_res != nullptr;
    const bool use_hits = graph_hits || (d.hits_ready() && d.decided);
    int64_t njobs = 0;

    for (int64_t i = 0; i < k; ++i) {
        const int64_t e = ids[i];
        if (e < 0 || e >= d.n_expert) {
            d.failed = true;
            d.fail = "a routed expert id is out of range";
            d.fail_layer = d.layers;
            d.fail_expert = e;
            return;
        }
        const uint8_t* b = d.src->blob(d.layers, e);
        if (b == nullptr) {
            // The one failure the loop cannot see.  Leaving `out` at its previous contents would feed the NEXT
            // layer a stale expert vector, which `moe_combine` would weight and add - the token would still be
            // finite and would still be wrong, 48 layers deep.
            d.failed = true;
            d.fail = "the expert source could not produce a blob";
            d.fail_layer = d.layers;
            d.fail_expert = e;
            ++d.missing;
            return;
        }
        // A hit's row was zeroed by `Launch` and belongs to the GPU; the pool must not touch it.
        if (use_hits && (graph_hits ? d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e] >= 0
                                    : d.is_hit[(size_t) i] != 0)) {
            if (graph_hits) ++d.cache_hits;
            // The GPU owns this row and `hit_out` is zeroed, so the CPU's contribution is zero - but
            // `y_miss` is a REUSED pinned buffer, so the row must be written, not merely skipped.
            std::memset(out + (size_t) i * (size_t) n_embd, 0, (size_t) n_embd * sizeof(float));
            continue;
        }
        if (graph_hits) ++d.cache_refused;   // token graph: a miss (nothing is admitted during a token)

        // `njobs` indexes the JOB ARRAY and `i` indexes the OUTPUT - they are the same only when nothing is a
        // hit, and using one for the other is how a hit's row would get two experts summed into it.
        ExpertJob& j = d.jobs[(size_t) njobs++];
        j.blob = b;
        j.act = &d.act;             // SHARED across the batch: one conversion serves all ten experts
        j.out = out + (size_t) i * (size_t) n_embd;
        j.weight = 1.0f;            // clause 2: a diagnostic field, NOT the router weight
        j.slot = (int) i;
    }

    // Plan v0.3 P4: rows of every expert across all threads (bitwise the same as `run`).
    if (d.split_rows) d.pool->run_split(d.jobs.data(), (int) njobs);
    else d.pool->run(d.jobs.data(), (int) njobs);
    ++d.layers;
    d.experts += k;
}

void expert_pool_dispatch_multi(ExpertDispatch& d, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k,
                                float* out) {
    using namespace strata::kernels::cpu;
    if (d.failed) return;
    if (n_tok < 1 || n_tok > MAXT) {
        d.failed = true;
        d.fail = "a verify window has more tokens than the multi-token expert kernel takes";
        d.fail_layer = d.layers;
        return;
    }
    if ((int64_t) d.act_multi.size() < n_tok) d.act_multi.resize((size_t) MAXT);
    const ExpertLayout& lay = expert_layout();
    const bool native = lay.native;
    if (native && d.nact_multi.size() < (size_t) MAXT * kNativeActBytes) d.nact_multi.resize((size_t) MAXT * kNativeActBytes);
    if (d.job_of.size() != (size_t) d.n_expert) d.job_of.assign((size_t) d.n_expert, (int16_t) -1);
    if (d.jobs_multi.size() < (size_t) (n_tok * k)) d.jobs_multi.resize((size_t) (MAXT * k));
    const auto c0 = std::chrono::steady_clock::now();
    d.src->begin_layer(d.layers, ids, n_tok * k);
    if (!d.usage.empty())
        for (int64_t i = 0; i < n_tok * k; ++i)
            if (ids[i] >= 0 && ids[i] < d.n_expert) d.usage[(size_t) d.layers * (size_t) d.n_expert + (size_t) ids[i]] += 1.0f;
    // ---- plan v0.3 P6: the GPU's share, decided and published FIRST so the GPU starts while the CPU works.
    // Distinct experts in routing order; resident ones and the last pcie_num/256 of the missed ones go to the GPU.
    const int64_t n = n_tok * k;
    int32_t kind[128];                     // per entry: -1 CPU, 0 VRAM, 1 PCIe
    if (d.plan != nullptr && n <= 128 && n <= d.plan->cap) {
        int64_t distinct[128], first_of[128];
        int nd = 0, nmiss = 0;
        for (int64_t i = 0; i < n; ++i) {
            first_of[i] = i;
            for (int64_t j = 0; j < i; ++j)
                if (ids[j] == ids[i]) { first_of[i] = first_of[j]; break; }
            if (first_of[i] == i) {
                distinct[nd++] = i;
                const int32_t e = ids[i];
                if (e >= 0 && e < d.n_expert && d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e] < 0) ++nmiss;
            }
        }
        const bool pcie_ok = d.pcie_num > 0 && d.src->pcie_ready();
        const int m = pcie_ok ? (nmiss * d.pcie_num) >> 8 : 0;
        int miss_rank = 0, groups = 0, entries = 0, fetches = 0;
        GpuPlanSink& P = *d.plan;
        const uint8_t* dma_src[64];
        int64_t pcie_i0[64];
        for (int q = 0; q < nd; ++q) {
            const int64_t i0 = distinct[q];
            const int32_t e = ids[i0];
            int kd = -1;
            unsigned long long ptr = 0;
            if (e >= 0 && e < d.n_expert) {
                const int32_t slot = d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e];
                if (slot >= 0) {
                    kd = 0;
                    ptr = (unsigned long long) (d.cache_base + (d.cache_slot_off ? (size_t) d.cache_slot_off[slot]
                                                                                 : (size_t) slot * (size_t) d.cache_blob));
                } else {
                    if (miss_rank >= nmiss - m && fetches < P.staging_cap && fetches < 64) {
                        const uint8_t* src = d.src->blob(d.layers, e);
                        if (src != nullptr && d.src->pinned(d.layers, e)) {
                            kd = 1;
                            dma_src[fetches] = src;
                            pcie_i0[fetches] = i0;
                            ++fetches;
                        }
                    }
                    ++miss_rank;
                }
            }
            for (int64_t i = i0; i < n; ++i)
                if (first_of[i] == i0) kind[i] = kd;
            if (kd != 0) continue;                 // the VRAM groups first; the PCIe groups below
            P.ptr[groups] = ptr;
            P.start[groups] = entries;
            for (int64_t i = i0; i < n; ++i)
                if (first_of[i] == i0) {
                    P.dst[entries] = (int32_t) i;
                    P.tok[entries] = (int32_t) (i / k);
                    ++entries;
                }
            ++groups;
        }
        P.start[groups] = entries;
        const uint64_t bb = lay.blob_bytes(d.layers);
        for (int q = 0; q < fetches; ++q) {       // the PCIe groups: staging slot q, entries after the VRAM ones
            const int64_t i0 = pcie_i0[q];
            P.ptr2[q] = P.pcie_mode != 0 ? (unsigned long long) d.src->device_alias(d.layers, ids[i0])
                                 : P.staging + (unsigned long long) q * (unsigned long long) bb;
            P.start2[q] = entries;
            for (int64_t i = i0; i < n; ++i)
                if (first_of[i] == i0) {
                    P.dst[entries] = (int32_t) i;
                    P.tok[entries] = (int32_t) (i / k);
                    ++entries;
                }
            ++d.pcie_experts;
        }
        P.start2[fetches] = entries;
        P.counts[0] = groups;
        P.counts[1] = entries;
        P.counts[2] = fetches;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (P.publish) P.publish(P.ctx);
        if (P.fetch) P.fetch(P.ctx, dma_src, P.pcie_mode != 0 ? 0 : fetches, (size_t) bb);   // the copy engine, beside the CPU's work
    } else {
        for (int64_t i = 0; i < n; ++i) {
            const int32_t e = ids[i];
            kind[i] = (e >= 0 && e < d.n_expert && d.host_res != nullptr &&
                       d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e] >= 0) ? 0 : -1;
        }
    }
    const auto c1 = std::chrono::steady_clock::now();
    if (native && lay.fmt[(size_t) d.layers].gu_type == 42)   // a native Q2_0 pack: the Q2_0 kernels' activations
        for (int64_t t = 0; t < n_tok; ++t) act_quant_any(x_f + (size_t) t * H, H, d.act_multi[(size_t) t]);
    else if (native)
        for (int64_t t = 0; t < n_tok; ++t)
            native_quant_act(lay.fmt[(size_t) d.layers], x_f + (size_t) t * H, d.nact_multi.data() + (size_t) t * kNativeActBytes);
    else
        for (int64_t t = 0; t < n_tok; ++t) act_quant_q8_1(x_f + (size_t) t * H, H, d.act_multi[(size_t) t]);
    const auto c2 = std::chrono::steady_clock::now();
    // ---- R7: TWO PASSES, SO THE DRIVE'S LATENCY HIDES BEHIND THE CPU'S OWN WORK.
    //
    // The layer's routed experts split into two sets that need nothing from each other:
    //
    //   * the ones whose bytes are ALREADY available - the host hot tier, or the mapping - and
    //   * the ones `begin_layer` has put on the wire and that are still landing.
    //
    // Measured with the one-pass form on the IQ3_XXS pack, a 20 GiB host tier and `--spec 4`: of a 272 ms
    // window, **176 ms was `begin_layer`'s blocking wait** while the CPU had **57 ms** of resident-expert work
    // queued behind it, and the GPU sat idle through both.  Running pass 1 first puts that 57 ms inside the
    // wait instead of after it.  Entries are consistent about which pass they belong to because `ring_pending`
    // is a property of `(layer, expert)`, so an expert's rows never straddle the two pool calls.
    auto dispatch_pass = [&](int pass) {
        int njobs = 0;
        for (int64_t t = 0; t < n_tok; ++t)
            for (int64_t j = 0; j < k; ++j) {
                const int64_t i = t * k + j;
                const int64_t e = ids[i];
                float* row = out + (size_t) i * H;
                if (pass == 0 && (e < 0 || e >= d.n_expert)) {
                    d.failed = true;
                    d.fail = "a routed expert id is out of range";
                    d.fail_layer = d.layers;
                    d.fail_expert = e;
                    return false;
                }
                if (kind[i] >= 0) {             // the GPU computes this entry (a VRAM hit or a PCIe read)
                    if (pass == 0) {
                        if (kind[i] == 0) ++d.cache_hits;
                        std::memset(row, 0, (size_t) H * sizeof(float));
                    }
                    continue;
                }
                if (d.src->ring_pending(d.layers, e) != (pass == 1)) continue;
                ++d.cache_refused;
                int16_t& jo = d.job_of[(size_t) e];
                if (jo < 0) {
                    const uint8_t* b = d.src->blob(d.layers, e);
                    if (b == nullptr) {
                        d.failed = true;
                        d.fail = "the expert source could not produce a blob";
                        d.fail_layer = d.layers;
                        d.fail_expert = e;
                        ++d.missing;
                        return false;
                    }
                    jo = (int16_t) njobs++;
                    ExpertJobMulti& nj = d.jobs_multi[(size_t) jo];
                    nj.blob = b;
                    nj.nt = 0;
                }
                ExpertJobMulti& jb = d.jobs_multi[(size_t) jo];
                jb.act[jb.nt] = &d.act_multi[(size_t) t];
                jb.nact[jb.nt] = native ? d.nact_multi.data() + (size_t) t * kNativeActBytes : nullptr;
                jb.out[jb.nt] = row;
                ++jb.nt;
                ++d.multi_entries;
            }
        if (njobs > 0) {
            if (native) d.pool->run_split_multi_native(lay.fmt[(size_t) d.layers], d.jobs_multi.data(), njobs);
            else d.pool->run_split_multi(d.jobs_multi.data(), njobs);
        }
        return true;
    };
    if (!dispatch_pass(0)) return;
    const auto c2b = std::chrono::steady_clock::now();
    d.src->wait_layer();          // the reads submitted by `begin_layer` are awaited HERE, after the residents
    const auto c2c = std::chrono::steady_clock::now();
    if (!dispatch_pass(1)) return;
    const auto c3 = std::chrono::steady_clock::now();
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    d.ms_plan += ms(c0, c1);
    d.ms_actq += ms(c1, c2);
    d.ms_jobs += ms(c2, c3);
    d.ms_run += ms(c2, c2b);            // pass 0: the resident experts, computed while the reads are in flight
    d.ms_wait += ms(c2b, c2c);          // the part of the drive's latency the resident work did NOT cover
    for (int64_t i = 0; i < n_tok * k; ++i) {
        const int64_t e = ids[i];
        if (e >= 0 && e < d.n_expert) d.job_of[(size_t) e] = -1;
    }
    d.multi_misses += d.cache_refused;
    if (!d.usage_total.empty())
        for (int64_t i = 0; i < n_tok * k; ++i)
            if (ids[i] >= 0 && ids[i] < d.n_expert) ++d.usage_total[(size_t) d.layers * (size_t) d.n_expert + (size_t) ids[i]];
    ++d.layers;
    d.experts += n_tok * k;
}

void expert_hit_run(void* user, void* stream, HitPhase phase, const int32_t* ids, int64_t k) {
    ExpertDispatch& d = *(ExpertDispatch*) user;
    if (d.failed) return;
    cudaStream_t cs = (cudaStream_t) stream;

    if (phase == HitPhase::Launch) {
        d.decided = false;
        d.hit_pending = false;
        if (!d.hits_ready() || ids == nullptr || k <= 0) return;
        if ((int64_t) d.is_hit.size() < k) d.is_hit.resize((size_t) k);

        // ================================ THE DECISION, ONCE, ON THIS LAYER'S IDS ================================
        //
        // Every routed expert is asked of the cache.  Resident -> the GPU computes it.  Not resident -> it is
        // admitted and filled if there is room (which makes it a hit on THIS call, because the fill and the
        // kernel are on one stream in that order), and otherwise it stays a miss for the CPU.
        d.n_hits = 0;
        for (int64_t i = 0; i < k; ++i) {
            const int64_t e = ids[i];
            d.is_hit[(size_t) i] = 0;
            if (e < 0 || e >= d.n_expert) continue;   // out of range: the pool refuses it, with a message
            int32_t slot = d.cache->slot_of(d.layers, e);
            if (slot == kNotResident) {
                const int32_t cand = d.cache->admit(d.layers, e);
                if (cand == kNotResident) {
                    ++d.cache_refused;
                    continue;
                }
                // `blob` is asked ONLY for an expert about to be filled, so the source's read counter stays a
                // count of distinct experts moved rather than of looks.
                const uint8_t* b = d.src->blob(d.layers, e);
                std::string ferr;
                if (b == nullptr || !d.cache->fill_slot(cand, b, cs, ferr, (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(d.layers))) {
                    d.failed = true;
                    d.fail = "the expert cache could not fill a slot";
                    d.fail_layer = d.layers;
                    d.fail_expert = e;
                    return;
                }
                ++d.cache_admitted;
                slot = cand;
            } else {
                ++d.cache_hits;
            }
            d.is_hit[(size_t) i] = 1;
            d.h_slot[(size_t) d.n_hits] = slot;
            d.h_dst[(size_t) d.n_hits] = (int32_t) i;
            ++d.n_hits;
        }
        d.decided = true;
        if (d.n_hits <= 0) return;   // nothing resident yet: no GPU work, and nothing for `Combine` to add

        const size_t list_bytes = (size_t) d.n_hits * sizeof(int32_t);
        // `hit_out` is ZEROED rather than overwritten: the kernel writes only the rows this layer's hits own,
        // so a row that was a hit last layer and a miss this one would still hold last layer's expert and
        // `add_inplace` would sum it in.  Finite, plausible, wrong.
        if (cudaMemsetAsync(d.hit_out, 0, (size_t) d.parts_elems * sizeof(float), cs) != cudaSuccess ||
            cudaMemcpyAsync(d.d_slot, d.h_slot.data(), list_bytes, cudaMemcpyHostToDevice, cs) != cudaSuccess ||
            cudaMemcpyAsync(d.d_dst, d.h_dst.data(), list_bytes, cudaMemcpyHostToDevice, cs) != cudaSuccess) {
            d.hit_fail = "the hit list could not be staged";
            d.failed = true;
            d.fail = d.hit_fail;
            return;
        }
        // The activation is quantized HERE rather than reused from `s.moe.x_q8_0`, which `post[l-1]` wrote from
        // the PREVIOUS layer's `mixed`.  `pre[l]` has since overwritten `mixed`, so that buffer is a layer stale
        // - and a stale activation produces a perfectly finite expert for the wrong input.
        // **R4.2h: THE SCALED QUANTIZER, SO A HIT REPRODUCES A MISS.**  The CPU pool quantizes this same
        // activation with `act_quant_q8_1` and multiplies by the fp32 `ActQ::scale`; `quantize_q8_0` writes
        // an fp16 `d` instead, and `bench/micro/act_quant_parity.cu` measured **80 of 80 chunks differing by
        // up to 4.761e-04 relative**.  `quantize_q8_0_scaled` adopts the CPU's rule and scale, and the kernel
        // takes the fp32 array.  Falling back to the old path would silently reintroduce the divergence, so
        // the scales are required here rather than optional.
        if (d.x_q8_0_hit_scale == nullptr) {
            d.failed = true;
            d.fail = "the hit path has no fp32 activation scales (R4.2h)";
            return;
        }
        strata::kernels::quantize_q8_0_scaled(d.mixed, d.x_q8_0_hit, d.x_q8_0_hit_scale, strata::kernels::cpu::H,
                                              cs);
        if (d.hit_cpu_order)
            strata::kernels::moe_hit_grouped_s2_cpu_order(d.cache_base, d.d_slot, d.d_dst, d.n_hits,
                d.cache_blob, d.x_q8_0_hit, d.hit_scratch, d.hit_out, cs, d.x_q8_0_hit_scale);
        else
            strata::kernels::moe_hit_grouped_s2(d.cache_base, d.d_slot, d.d_dst, d.n_hits, d.cache_blob,
                d.x_q8_0_hit, d.hit_scratch, d.hit_out, cs, d.x_q8_0_hit_scale);
        d.hit_pending = true;
        if (d.hit_done != nullptr) cudaEventRecord((cudaEvent_t) d.hit_done, cs);
        // The A/B arm: ONE driver entry here, and nothing else changes.  If the work was waiting for the host
        // to enter the driver, this is what lets it start while the pool runs.
        if (d.hit_poke && d.hit_done != nullptr) (void) cudaEventQuery((cudaEvent_t) d.hit_done);
        return;
    }

    // Combine: `parts += hit_out`, stream-ordered after the misses were copied into `parts`.
    if (!d.hit_pending) return;
    d.hit_pending = false;
    // Did the GPU get the hit work done while the CPU was in the pool?  This query is itself a driver entry,
    // so it is the LAST chance to observe a late start: a NOT-READY here means the work had not finished by the
    // time the pool returned, and with no poke in front of it that can only be because it began after.
    if (d.hit_done != nullptr) {
        if (cudaEventQuery((cudaEvent_t) d.hit_done) == cudaSuccess) ++d.hit_ready;
        else ++d.hit_late;
    }
    strata::kernels::add_inplace(d.parts_out, d.hit_out, d.parts_elems, cs);
}

// ================================ THE RESIDENT ARENA (R2.1) ================================

// Plan v0.3 P6: the arena from the model's shard 1.  Each layer's gate, up and down tensors hold the 512 experts
// one after another; they are read in chunks and each expert's slice lands at its place in the blob
// [gate rows | up rows | down rows] - the layout tools/iq_pack.py would have written to experts.bin.
LoadStats load_experts_gguf(const std::string& gguf, uint8_t* dst, const strata::kernels::cpu::ExpertLayout& lay,
                            int threads) {
    LoadStats st;
    st.layers = (uint64_t) lay.n_layers;
    const auto t0 = std::chrono::steady_clock::now();
    std::atomic<int64_t> next{0};
    std::atomic<bool> bad{false};
    // a layer's experts may sit in another shard of the model (native_experts.txt v3): a name beside `gguf`
    const size_t cut = gguf.find_last_of("/\\");
    const std::string dir = cut == std::string::npos ? std::string() : gguf.substr(0, cut + 1);
    auto file_of = [&](int64_t l) -> std::string {
        if (lay.gguf_file.empty() || lay.gguf_file[(size_t) l].empty()) return gguf;
        return dir + lay.gguf_file[(size_t) l];
    };
    auto worker = [&]() {
        std::ifstream f;
        std::string open_name;
        std::vector<uint8_t> buf;
        for (;;) {
            const int64_t l = next.fetch_add(1);
            if (l >= lay.n_layers || bad) break;
            const std::string name = file_of(l);
            if (name != open_name) {
                f.close();
                f.clear();
                f.open(name, std::ios::binary);
                if (!f) { bad = true; return; }
                open_name = name;
            }
            const auto& fm = lay.fmt[(size_t) l];
            const uint64_t blob = lay.bytes[(size_t) l];
            const uint64_t per[3] = {fm.up_off, fm.up_off, blob - fm.down_off};
            const uint64_t at[3] = {0, fm.up_off, fm.down_off};
            for (int r = 0; r < 3; ++r) {
                const uint64_t src = lay.gguf_off[(size_t) (3 * l + r)];
                const uint64_t total = per[r] * (uint64_t) lay.n_expert;
                const uint64_t chunk = per[r] * 16;           // 16 experts per read
                buf.resize((size_t) chunk);
                for (uint64_t done = 0; done < total; done += chunk) {
                    const uint64_t n = std::min<uint64_t>(chunk, total - done);
                    f.seekg((std::streamoff) (src + done));
                    f.read((char*) buf.data(), (std::streamsize) n);
                    if ((uint64_t) f.gcount() != n) { bad = true; return; }
                    for (uint64_t k = 0; k < n / per[r]; ++k) {
                        const uint64_t e = done / per[r] + k;
                        std::memcpy(dst + lay.blob_offset(l, (int64_t) e) + at[r], buf.data() + k * per[r], (size_t) per[r]);
                    }
                }
            }
        }
    };
    std::vector<std::thread> pool;
    for (int i = 1; i < threads; ++i) pool.emplace_back(worker);
    worker();
    for (auto& t : pool) t.join();
    if (bad) {
        st.seconds = -1.0;
        return st;
    }
    st.bytes = lay.total;
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return st;
}

ArenaExpertSource::~ArenaExpertSource() { close(); }

bool ArenaExpertSource::open(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, int threads,
                             std::string& err) {
    close();
    const std::string path = pack_dir + "/experts.bin";
    // plan v0.3 P6: the layout (canonical, or a native pack's per-layer blobs) was loaded by the driver
    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
    if (lay.n_layers != n_layers || lay.n_expert != n_expert) {
        err = "ArenaExpertSource: the expert layout was loaded for a different geometry";
        return false;
    }
    const int64_t blob = (int64_t) lay.max_blob;
    const uint64_t want = lay.total;

    // plan v0.3 P6: no experts.bin in a native pack -> the experts come straight from the GGUF
    const bool from_gguf = !std::ifstream(path, std::ios::binary) && lay.native && !lay.gguf_off.empty() && !gguf_.empty();
    // SIZE CHECK BEFORE THE ALLOCATION, not after.  A wrong pack should name the two numbers rather than spend
    // 34 GB and a minute of loading first.
    if (!from_gguf) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) { err = "ArenaExpertSource: cannot open " + path; return false; }
        const uint64_t got = (uint64_t) f.tellg();
        if (got != want) {
            char buf[400];
            std::snprintf(buf, sizeof buf,
                          "ArenaExpertSource: %s is %llu B but %lld layers x %lld experts (blobs up to %lld B) "
                          "make %llu B - this is not the pack this geometry came from",
                          path.c_str(), (unsigned long long) got, (long long) n_layers, (long long) n_expert,
                          (long long) blob, (unsigned long long) want);
            err = buf;
            return false;
        }
    }

    // one layer per registration slice, so no expert straddles two registrations.  The arena is one blob
    // longer than the file: a copy of a whole VRAM slot (the largest blob) may then start at any expert.
    std::vector<uint64_t> bounds, loff, lbytes;
    for (int64_t l = 0; l < n_layers; ++l) {
        bounds.push_back(lay.layer_offset(l));
        loff.push_back(lay.layer_offset(l));
        lbytes.push_back(lay.blob_bytes(l) * (uint64_t) n_expert);
    }
    bounds.push_back(want);
    // Small-RAM machines: a native pack's experts (tens of GiB) as a file-backed mmap arena.  The first run
    // materializes `<pack_dir>/experts-native.bin` from the GGUF shards; later runs page in lazily and the
    // kernel page cache holds whatever fits - the same contract as FileExpertSource's mmap of experts.bin,
    // for a pack whose blobs are not the canonical Q2_0 size.
    if (from_gguf && file_backing_) {
#if defined(_WIN32)
        err = "ArenaExpertSource: file-backed native experts need the POSIX mmap path";
        return false;
#else
        const std::string fb = pack_dir + "/experts-native.bin";
        const uint64_t cap = want + (uint64_t) blob;
        bool preexisting = false;
        {
            std::ifstream f(fb, std::ios::binary | std::ios::ate);
            // a run killed mid-materialization leaves the full-size but INCOMPLETE file (ftruncate up front),
            // so trust it only with the completion marker written after the load finished
            if (f) {
                std::ifstream done(fb + ".done");
                preexisting = ((uint64_t) f.tellg() == cap) && done.good();
            }
        }
        file_fd_ = ::open(fb.c_str(), O_RDWR | O_CREAT, 0644);
        if (file_fd_ < 0) { err = "ArenaExpertSource: cannot open " + fb; return false; }
        if (!preexisting && ::ftruncate(file_fd_, (off_t) cap) != 0) {
            ::close(file_fd_); file_fd_ = -1;
            err = "ArenaExpertSource: cannot size " + fb;
            return false;
        }
        void* m = ::mmap(nullptr, cap, PROT_READ | PROT_WRITE, MAP_SHARED, file_fd_, 0);
        if (m == MAP_FAILED) {
            ::close(file_fd_); file_fd_ = -1;
            err = "ArenaExpertSource: mmap of " + fb + " failed (" + std::to_string(cap) + " B)";
            return false;
        }
        file_map_ = m;
        file_map_bytes_ = cap;
        base_ = (const uint8_t*) m;
        arena_ = nullptr;
        pinned_bytes_ = 0;
        slice_bytes_ = 0;
        dev_slice_.clear();
        note_ = "file-backed mmap " + fb +
                (preexisting ? " (pre-existing; lazy page-in)" : " (materializing from the GGUF shards)");
        LoadStats st;
        if (preexisting) {
            st.bytes = want;
            st.layers = (uint64_t) n_layers;
        } else {
            st = load_experts_gguf(gguf_, (uint8_t*) m, lay, threads);
        }
        if (st.bytes != want) {
            close();
            err = "ArenaExpertSource: the load read " + std::to_string(st.bytes) + " B of " + std::to_string(want);
            return false;
        }
        blobs_ = n_layers * n_expert;
        n_expert_ = n_expert;
        reads_ = 0;
        gib_per_s_ = st.gib_per_second();
        if (!preexisting) {
            std::ofstream m(fb + ".done");
            m << "ok";
        }
        return true;
#endif
    }
    PinnedArena* a = new PinnedArena(want + (uint64_t) blob, bounds);
    if (!a->valid()) {
        delete a;
        err = "ArenaExpertSource: the arena could not be reserved (" + std::to_string(want) + " B)";
        return false;
    }
    const LoadStats st = from_gguf ? load_experts_gguf(gguf_, a->data(), lay, threads)
                                   : load_experts_ranges(path, a->data(), loff, lbytes, threads, /*chunk=*/8u << 20);
    if (st.bytes != want) {
        delete a;
        err = "ArenaExpertSource: the load read " + std::to_string(st.bytes) + " B of " + std::to_string(want);
        return false;
    }
    arena_ = a;
    base_ = a->data();
    pinned_bytes_ = a->registered_bytes;
    // plan v0.3 P6: device aliases of the mapped registration, for the PCIe share of the misses
    dev_slice_.clear();
    slice_bytes_ = a->slice_bytes;
    if (a->registered_bytes > 0) {
        std::vector<uint64_t> starts = a->slice_bytes > 0 ? a->slice_starts : std::vector<uint64_t>{0};
        for (uint64_t off : starts) {
            void* d = nullptr;
            if (cudaHostGetDevicePointer(&d, (void*) (base_ + off), 0) != cudaSuccess) {
                (void) cudaGetLastError();
                dev_slice_.clear();
                break;
            }
            dev_slice_.push_back((const uint8_t*) d);
        }
    }
    blobs_ = n_layers * n_expert;
    n_expert_ = n_expert;
    reads_ = 0;
    note_ = a->note;
    gib_per_s_ = st.gib_per_second();
    return true;
}

void ArenaExpertSource::close() {
    if (pf_started_) {
        pf_stop_.store(true, std::memory_order_release);
        pf_epoch_.fetch_add(1, std::memory_order_release);
        for (auto& t : pf_workers_) t.join();
        pf_workers_.clear();
        pf_started_ = false;
        pf_stop_.store(false, std::memory_order_release);
    }
    ring_.clear();
    ring_of_.clear();
    ring_layer_ = -1;
    if (hot_arena_ != nullptr) {
#if !defined(_WIN32)
        if (hot_locked_ && hot_cap_ > 0) ::munlock(hot_arena_, (size_t) hot_cap_);
        ::munmap(hot_arena_, (size_t) hot_cap_);
#endif
        hot_arena_ = nullptr;
        hot_cap_ = hot_used_ = 0;
        hot_count_ = 0;
        hot_locked_ = false;
        hot_slot_.clear();
        dc_slot_bytes_ = 0;
        dc_slots_ = 0;
        dc_head_ = dc_tail_ = -1;
        dc_prev_.clear(); dc_next_.clear(); dc_idx_.clear(); dc_epoch_.clear();
        dc_free_.clear(); dc_admit_list_.clear();
        dc_admits_ = dc_evicts_ = dc_fallbacks_ = 0;
    }
    if (file_map_ != nullptr) {
#if !defined(_WIN32)
        if (file_map_bytes_ > 0) ::munmap(file_map_, file_map_bytes_);
#endif
        file_map_ = nullptr;
        file_map_bytes_ = 0;
    }
    if (file_fd_ >= 0) {
#if !defined(_WIN32)
        ::close(file_fd_);
#endif
        file_fd_ = -1;
    }
    if (arena_ != nullptr) {
        delete (PinnedArena*) arena_;
        arena_ = nullptr;
    }
    base_ = nullptr;
    blobs_ = 0;
    n_expert_ = 0;
}

bool ArenaExpertSource::pinned(int64_t layer, int64_t expert) const {
    if (base_ == nullptr || layer < 0 || layer >= strata::kernels::cpu::expert_layout().n_layers ||
        expert < 0 || expert >= n_expert_)
        return false;
    // R9b: a TIER-RESIDENT blob's host pointer is inside the cudaHostRegistered tier - safe for async DMA.
    // The in-flight ring blobs are NOT (their bytes land in the tier only when `wait_layer` commits), and the
    // file mapping never is.  `blob()` answers tier residents first, so this predicate agrees with the pointer
    // `blob()` returns for exactly the blobs that can be staged to the GPU.
    if (file_map_ != nullptr) {
        if (!hot_dev_ok_ || hot_slot_.empty()) return false;
        const int64_t idx = layer * n_expert_ + expert;
        return idx >= 0 && idx < blobs_ && hot_slot_[(size_t) idx] >= 0;
    }
    const auto& lay = strata::kernels::cpu::expert_layout();
    return lay.blob_offset(layer, expert) + lay.blob_bytes(layer) <= pinned_bytes_;
}

const uint8_t* ArenaExpertSource::device_alias(int64_t layer, int64_t expert) const {
    // R9b: the registered tier's device alias (pcie-mode direct/kernel).  Only the single-slice form is
    // wired; multi-slice leaves `hot_dev_base_` null and DMA mode (the default for native packs) instead.
    if (file_map_ != nullptr) {
        if (!hot_dev_ok_ || hot_dev_base_ == nullptr || hot_slot_.empty()) return nullptr;
        if (layer < 0 || expert < 0 || expert >= n_expert_) return nullptr;
        const int64_t idx = layer * n_expert_ + expert;
        if (idx < 0 || idx >= blobs_ || hot_slot_[(size_t) idx] < 0) return nullptr;
        return hot_dev_base_ + hot_slot_[(size_t) idx];
    }
    if (dev_slice_.empty() || !pinned(layer, expert)) return nullptr;
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (slice_bytes_ == 0) return dev_slice_[0] + lay.blob_offset(layer, expert);
    // one registration slice per layer
    if ((size_t) layer >= dev_slice_.size()) return nullptr;
    return dev_slice_[(size_t) layer] + (uint64_t) expert * lay.blob_bytes(layer);
}

// ================================ THE PINNED HOT TIER (small-RAM machines) ================================
//
// See the header.  Two things make this pay rather than duplicate the mapping:
//
//   * the tier is ANONYMOUS and `mlock`ed, so it is not a reclaim candidate.  The header above says the mapped
//     arena loses exactly that property ("file-backed pages are the ones the OS drops from the standby list"),
//     and measured 42.8 GB/s reading anonymous memory against ~19 GB/s through the mapping;
//   * each blob is copied with ONE whole-blob read out of the mapping.  A 2.6 MB blob touched a 4 KiB page at a
//     time costs ~340 us per fault on this class of drive (~45x the drive's own 3.8 ms for the whole blob).
//
// The order is the PROFILE's, so the tier holds the highest-frequency blobs first and what still misses is the
// tail of the distribution.  After each range is copied its file pages are released with POSIX_FADV_DONTNEED:
// they will never be read from the file again (this tier serves them), and on a 30 GiB box the 10-14 GiB that
// frees is what lets the COLD expert set stay cached for prefill's full pass over it.
void ArenaExpertSource::pin_hot(const std::vector<std::pair<int32_t, int32_t>>& ranked, uint64_t bytes) {
    if (base_ == nullptr || file_map_ == nullptr || bytes == 0 || ranked.empty() || blobs_ <= 0 || hot_arena_ != nullptr)
        return;
    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
    if (lay.n_expert <= 0 || lay.n_layers <= 0) return;
    // R8: the tier is dynamic (LRU with profile-seeded contents) unless the A/B arm asks for the static form.
    dynamic_tier_ = (std::getenv("STRATA_STATIC_TIER") == nullptr);
#if !defined(_WIN32)
#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB (21 << 26)
#endif
    // R7: **MADV_HUGEPAGE, AND READING STRAIGHT INTO THE TIER.**  Two changes from the R5 form, both measured
    // on the reason the tier exists:
    //
    //   * TLB.  The whole point of this tier is that a decode step walks ~460 two-megabyte blobs at random.
    //     Through 4 KiB pages that is ~550,000 TLB entries of working set per window; through 2 MiB pages it is
    //     ~460.  `MAP_HUGETLB` cannot be used - this box has 0 reserved hugepages and reserving GiB of them
    //     would fight the kernel for the same RAM - but THP is `always` here with `defer+madvise` defrag, so
    //     `MADV_HUGEPAGE` on the anonymous range is what actually gets the 2 MiB pages.  The mapping is sized
    //     UP to a 2 MiB multiple and the blob starts are 256-byte aligned, so no blob straddles a boundary
    //     that matters.
    //   * I/O.  The R5 form memcpy'd out of the file MAPPING, so filling a 12-20 GiB tier faulted 4 KiB at a
    //     time and left 12-20 GiB of useless page cache behind.  It now `pread`s whole blobs into the tier
    //     directly: one syscall per blob at the drive's own rate, and nothing is left in the page cache for
    //     the kernel to evict the hot set in favour of.
    //
    // R8: the arena is carved into FIXED SLOTS of the largest blob (4 KiB-rounded).  Blob sizes are uniform
    // across this pack's layers to ~1% (measured: 5,318 disk blobs / 11.57 GB = 2.176 MB average against a
    // 2.18 MB max), so the per-slot waste is negligible, and fixed slots make admission, eviction and the LRU
    // O(1) array walks - no allocator, no fragmentation, and a freed slot holds any layer's next blob.
    const uint64_t slot = ((uint64_t) lay.max_blob + 4095u) / 4096u * 4096u;
    const uint64_t cap = bytes / slot * slot;
    if (cap == 0) { note_ += "; hot tier skipped (bytes below one slot)"; return; }
    void* m = ::mmap(nullptr, (size_t) cap, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) {
        note_ += "; hot tier NOT allocated (mmap of " + std::to_string(cap) + " B failed)";
        return;
    }
    hot_arena_ = (uint8_t*) m;
    hot_cap_ = cap;
    const bool huge = (::madvise(m, (size_t) cap, MADV_HUGEPAGE) == 0);
    dc_init((int64_t) slot);
    hot_slot_.assign((size_t) blobs_, -1);
    for (const auto& pr : ranked) {
        if (hot_count_ >= dc_slots_) break;   // the arena is full of whole slots
        const int32_t l = pr.first, e = pr.second;
        if (l < 0 || e < 0 || l >= lay.n_layers || e >= lay.n_expert) continue;
        const int64_t idx = (int64_t) l * n_expert_ + (int64_t) e;
        if (idx < 0 || idx >= blobs_ || hot_slot_[(size_t) idx] >= 0) continue;
        const uint64_t len = lay.blob_bytes(l);
        const uint64_t off = lay.blob_offset(l, e);
        if (len == 0 || off + len > file_map_bytes_ || len > dc_slot_bytes_) break;
        uint8_t* dst = hot_arena_ + (uint64_t) hot_count_ * dc_slot_bytes_;
        bool ok = false;
        if (file_fd_ >= 0) {
            uint64_t done = 0;
            while (done < len) {
                const ssize_t r = ::pread(file_fd_, dst + done, (size_t) (len - done), (off_t) (off + done));
                if (r <= 0) break;
                done += (uint64_t) r;
            }
            ok = (done == len);
        } else {
            std::memcpy(dst, base_ + off, (size_t) len);
            ok = true;
        }
        if (!ok) break;
        // R8: rank order IS the initial LRU order - rank 0 at the head (evicted last), the rank cutoff at
        // the tail, so a domain shift evicts the coldest profile blobs first.
        const int32_t s = (int32_t) hot_count_;
        dc_idx_[(size_t) s] = (int32_t) idx;
        dc_epoch_[(size_t) s] = 0;
        // R8.1: the profile seeds the frequency dimension too.  64 halves to ~1 after six untouched windows,
        // so on a genuinely different domain the cold profile blobs DO yield - but a one-shot miss flood
        // (count 1) can never flush them within a single request.
        dc_count_[(size_t) s] = 64;
        dc_prev_[(size_t) s] = dc_tail_;
        dc_next_[(size_t) s] = -1;
        if (dc_tail_ >= 0) dc_next_[(size_t) dc_tail_] = s; else dc_head_ = s;
        dc_tail_ = s;
        hot_slot_[(size_t) idx] = (int64_t) ((uint64_t) s * dc_slot_bytes_);
        ++hot_count_;
        // **AND GIVE THE FILE PAGES BACK.**  A `pread` of a blob leaves that blob in the page cache, so filling
        // a 20 GiB tier would otherwise leave 20 GiB of page cache behind - for pages this tier now serves from
        // anonymous memory and will never read from the file again.  On a 30 GiB box that page cache is the
        // difference between a hot tier that can be grown and one that cannot.
        if (file_fd_ >= 0) ::posix_fadvise(file_fd_, (off_t) off, (off_t) len, POSIX_FADV_DONTNEED);
    }
    for (int64_t s = hot_count_; s < dc_slots_; ++s) dc_free_.push_back((int32_t) s);
    if (hot_count_ == 0) {
        ::munmap(hot_arena_, (size_t) hot_cap_);
        hot_arena_ = nullptr; hot_cap_ = 0; hot_slot_.clear();
        note_ += "; hot tier empty";
        return;
    }
    hot_used_ = (uint64_t) hot_count_ * dc_slot_bytes_;
    // R8: lock AFTER the fill (the R7 order this kernel accepts).  Locking the untouched mapping BEFORE
    // filling fails with ENOMEM here (22 GiB of THP prefault under fragmentation) and the failure mode is
    // catastrophic: the tier runs reclaimable and the kernel evicts hot-tier pages mid-generation.  mlock()
    // sets VM_LOCKED on the whole VMA, so pages faulted LATER by an LRU admission inside the range are
    // locked too - admissions need nothing extra.
    hot_locked_ = (::mlock(hot_arena_, (size_t) hot_cap_) == 0);
    if (!dynamic_tier_ && hot_used_ < hot_cap_) {   // static A/B: the free tail has no future, give it back
        ::munmap(hot_arena_ + hot_used_, (size_t) (hot_cap_ - hot_used_));
        hot_cap_ = hot_used_;
        dc_slots_ = (int64_t) hot_count_;
    }
    if (!dynamic_tier_) hot_locked_ = (::mlock(hot_arena_, (size_t) hot_used_) == 0);
    // ---- R9b: REGISTER THE TIER WITH CUDA so the GPU's PCIe expert path can read it.
    //
    // With `--mmap-experts` the arena is a plain file mmap: `pinned_bytes_` is 0, `dev_slice_` empty, so the
    // verify plan builder's `pinned()`/`device_alias()` gates were permanently closed and EVERY routed expert
    // was computed on the CPU pool - measured ~330 ms of serial pool work per verify window on this box, which
    // is the decode wall all by itself.  The tier is anonymous, mlocked and 2 MiB-page-backed - exactly what
    // `cudaHostRegister` wants - and tier-resident blobs are 97%+ of routed traffic, so registering it turns
    // `fetch_dma` (default --pcie-mode auto -> DMA for native packs, --pcie-frac 0.55) back on: the copy
    // engine moves the layer's PCIe share into VRAM staging beside the CPU's own work and the grouped kernel
    // computes it on the GPU.  Whole-range registration first; 2 GiB slices as the fallback, because a
    // failure here must degrade to the CPU-only path, never to torn reads.
    if (hot_locked_ && hot_arena_ != nullptr && hot_cap_ > 0 && file_map_ != nullptr) {
        void* dev = nullptr;
        auto reg = [&](uint8_t* p, uint64_t n) {
            return cudaHostRegister(p, (size_t) n, cudaHostRegisterPortable | cudaHostRegisterMapped);
        };
        if (reg(hot_arena_, hot_cap_) == cudaSuccess) {
            if (cudaHostGetDevicePointer(&dev, hot_arena_, 0) == cudaSuccess) {
                hot_dev_ok_ = true;
                hot_dev_base_ = (const uint8_t*) dev;
            } else {
                (void) cudaGetLastError();
                cudaHostUnregister(hot_arena_);
            }
        } else {
            (void) cudaGetLastError();
            const uint64_t slice = (uint64_t) 2 << 30;
            hot_dev_ok_ = true;
            std::vector<void*> parts;
            for (uint64_t off = 0; off < hot_cap_ && hot_dev_ok_; off += slice) {
                const uint64_t n = std::min<uint64_t>(slice, hot_cap_ - off);
                if (reg(hot_arena_ + off, n) != cudaSuccess) { hot_dev_ok_ = false; break; }
                void* d = nullptr;
                if (cudaHostGetDevicePointer(&d, hot_arena_ + off, 0) != cudaSuccess) { hot_dev_ok_ = false; break; }
                parts.push_back(d);
            }
            if (hot_dev_ok_ && parts.size() == 1) hot_dev_base_ = (const uint8_t*) parts[0];
            if (!hot_dev_ok_) {
                for (size_t i = 0; i < parts.size(); ++i) cudaHostUnregister(hot_arena_ + (uint64_t) i * slice);
            } else if (parts.size() > 1) {
                // multi-slice: DMA (pcie-mode 0) works because it only needs `pinned()` == true; the
                // single-pointer `device_alias()` form (direct/kernel modes) stays off with a null base.
                hot_dev_base_ = nullptr;
            }
        }
    }
#endif
    char buf[420];
    std::snprintf(buf, sizeof buf,
                  "; hot tier %.2f GiB in %lld blobs from the profile%s%s (THP %s, slot %.2f MiB, %lld slots, %lld free)",
                  (double) hot_used_ / 1073741824.0, (long long) hot_count_,
                  hot_locked_ ? " (mlocked)" : " (mlock FAILED - reclaimable)",
                  dynamic_tier_ ? ", LRU adaptive" : ", static", huge ? "on" : "off",
                  (double) dc_slot_bytes_ / 1048576.0, (long long) dc_slots_,
                  (long long) dc_free_.size());
    note_ += buf;
    if (file_map_ != nullptr)
        note_ += hot_dev_ok_ ? (hot_dev_base_ ? "; tier cudaHostRegistered+device-alias (GPU PCIe expert path ON)"
                                              : "; tier cudaHostRegistered sliced (GPU PCIe DMA path ON)")
                             : "; tier NOT registered (GPU PCIe expert path off)";
}

// ---- R8: the adaptive LRU tier ---------------------------------------------------------------------------

void ArenaExpertSource::dc_init(int64_t slot_bytes) {
    dc_slot_bytes_ = (uint64_t) slot_bytes;
    dc_slots_ = hot_cap_ > 0 ? (int64_t) (hot_cap_ / dc_slot_bytes_) : 0;
    dc_prev_.assign((size_t) dc_slots_, -1);
    dc_next_.assign((size_t) dc_slots_, -1);
    dc_idx_.assign((size_t) dc_slots_, -1);
    dc_epoch_.assign((size_t) dc_slots_, 0);
    dc_count_.assign((size_t) dc_slots_, 0);
    dc_free_.clear();
    dc_head_ = dc_tail_ = -1;
    dc_admit_list_.clear();
    dc_admits_ = dc_evicts_ = dc_fallbacks_ = 0;
}

void ArenaExpertSource::dc_unlink(int32_t s) {
    const int32_t p = dc_prev_[(size_t) s], n = dc_next_[(size_t) s];
    if (p >= 0) dc_next_[(size_t) p] = n; else dc_head_ = n;
    if (n >= 0) dc_prev_[(size_t) n] = p; else dc_tail_ = p;
    dc_prev_[(size_t) s] = dc_next_[(size_t) s] = -1;
}

void ArenaExpertSource::dc_link_head(int32_t s) {
    dc_prev_[(size_t) s] = -1;
    dc_next_[(size_t) s] = dc_head_;
    if (dc_head_ >= 0) dc_prev_[(size_t) dc_head_] = s;
    dc_head_ = s;
    if (dc_tail_ < 0) dc_tail_ = s;
}

void ArenaExpertSource::dc_touch(int32_t s) {
    std::lock_guard<std::mutex> lk(dc_mu_);
    dc_touch_locked(s);
}

void ArenaExpertSource::dc_touch_locked(int32_t s) {
    if (s < 0 || s >= (int32_t) dc_slots_) return;
    if (dc_head_ != s) {
        dc_unlink(s);
        dc_link_head(s);
    }
    dc_epoch_[(size_t) s] = window_epoch_;
    // R8.1: frequency.  A one-shot miss (a diverse generation's novel expert) stays cheap to evict; a blob
    // the workload keeps re-referencing gets expensive to evict.  Capped so a long-lived favourite cannot
    // become untouchable.
    if (dc_count_[(size_t) s] < 4095) ++dc_count_[(size_t) s];
}

int32_t ArenaExpertSource::dc_alloc(int64_t idx) {
    std::lock_guard<std::mutex> lk(dc_mu_);
    return dc_alloc_locked(idx);
}

int32_t ArenaExpertSource::dc_alloc_locked(int64_t idx) {
    int32_t s;
    if (!dc_free_.empty()) {
        s = dc_free_.back();
        dc_free_.pop_back();
    } else {
        // R8.1: EVICT BY FREQUENCY, WITH LRU AS THE TIE-BREAK.  The pure-LRU form thrashes on a diverse
        // long generation: every miss is a one-shot (measured: 9,754 misses, 0% window-to-window repeat),
        // so the flood admitted and evicted 15,914 blobs in one request and decode fell to 7.6 tok/s while
        // the resident profile content - which WOULD have been hit - was flushed.  The victim is now the
        // lowest-hit-count slot within a bounded walk from the LRU tail (LRU position breaks ties), so a
        // one-shot flood recycles among itself and never pushes out content the workload actually re-uses.
        s = -1;
        int32_t best = -1;
        uint16_t best_count = 0xffff;
        int32_t v = dc_tail_;
        for (int scanned = 0; v >= 0 && scanned < 512; ++scanned, v = dc_prev_[(size_t) v]) {
            if (dc_epoch_[(size_t) v] == window_epoch_) continue;   // this window's own bytes
            const uint16_t c = dc_count_[(size_t) v];
            if (c < best_count) { best_count = c; best = v; if (c == 0) break; }
        }
        if (best < 0) return -1;   // every slot is in use by this window: fall back to the legacy ring
        s = best;
        const int64_t vidx = dc_idx_[(size_t) s];
        if (vidx >= 0 && vidx < (int64_t) hot_slot_.size()) hot_slot_[(size_t) vidx] = -1;
        ++dc_evicts_;
        // NO explicit unlink here: `dc_touch_locked` below unlinks the victim (it is still in the list) and
        // relinks it at the head.  Unlinking twice corrupts the intrusive list - the second unlink reads the
        // -1 sentinels and sets head = tail = -1, which silently collapses the LRU to one slot and every
        // later allocation evicts nothing (measured: 394 admissions against 16,671 ring fallbacks).
    }
    dc_idx_[(size_t) s] = (int32_t) idx;
    dc_touch_locked(s);
    dc_count_[(size_t) s] = 1;   // admitted: one shot of credit until the workload proves otherwise
    return s;
}

// R8b: **PREFILL ADMISSIONS.**  See the header.  `bytes` are the staging buffer's landed copy of
// `(layer, expert)`; a resident blob just gets its frequency touch, a miss takes a slot (evicting the
// lowest-value victim) and memcpy's in.  Called from prefill's consume step, host thread, serialized against
// the decode path - but NOT against prefill's own reader threads touching resident blobs, hence the lock.
bool ArenaExpertSource::dc_stage_admit(int64_t layer, int64_t expert, const uint8_t* bytes) {
    if (!dynamic_tier_ || dc_slots_ <= 0 || hot_slot_.empty() || bytes == nullptr) return false;
    if (layer < 0 || expert < 0 || expert >= n_expert_) return false;
    const int64_t idx = layer * n_expert_ + expert;
    if (idx < 0 || idx >= blobs_) return false;
    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
    const uint64_t len = lay.blob_bytes(layer);
    if (len == 0 || len > dc_slot_bytes_) return false;
    std::lock_guard<std::mutex> lk(dc_mu_);
    if (hot_slot_[(size_t) idx] >= 0) {          // resident: this prompt's evidence of reuse
        dc_touch_locked((int32_t) ((uint64_t) hot_slot_[(size_t) idx] / dc_slot_bytes_));
        return true;
    }
    const int32_t s = dc_alloc_locked(idx);
    if (s < 0) return false;
    std::memcpy(hot_arena_ + (uint64_t) s * dc_slot_bytes_, bytes, (size_t) len);
    hot_slot_[(size_t) idx] = (int64_t) ((uint64_t) s * dc_slot_bytes_);
    ++dc_stage_admits_;
    return true;
}

const uint8_t* ArenaExpertSource::blob(int64_t layer, int64_t expert) {
    if (base_ == nullptr) return nullptr;
    if (layer < 0 || expert < 0 || expert >= n_expert_) return nullptr;
    const int64_t idx = layer * n_expert_ + expert;
    if (idx < 0 || idx >= blobs_) return nullptr;
    ++reads_;
    ++blob_stats_.requests;
    // R8: the hot tier is checked FIRST.  An admission is committed here in `wait_layer`, before the pass-1
    // `blob()` calls run - and a committed blob must NOT fall through to the ring check, because its ring
    // slot (when it has one at all) is the legacy staging buffer, not the memory its bytes were read into.
    if (!hot_slot_.empty() && hot_slot_[(size_t) idx] >= 0) {
        ++hot_lookups_;
        ++hot_hits_;
        ++blob_stats_.hot;
        if (dc_slots_ > 0) dc_touch((int32_t) ((uint64_t) hot_slot_[(size_t) idx] / dc_slot_bytes_));
        return hot_arena_ + hot_slot_[(size_t) idx];
    }
    // R5-fetch: the ring serves THIS layer's pread misses (see begin_layer); `ring_layer_` pins the validity
    // to the layer the reads were issued for, so a later `blob()` (the profile fill, prefill's staging) can
    // never see a previous layer's bytes.
    if (layer == ring_layer_ && (int64_t) ring_of_.size() == blobs_ && ring_of_[(size_t) idx] >= 0) {
        ++blob_stats_.ring;
        return ring_[(size_t) ring_of_[(size_t) idx]].data();
    }
    // R7: the fall-through is a fault on the file mapping.  If the page is in the page cache it is a DRAM read;
    // if not, the kernel goes to the drive for it, 4 KiB at a time.  This is the path the R5 pread ring exists
    // to avoid, and counting it is how a run says whether it avoided it.
    ++blob_stats_.map;
    return base_ + strata::kernels::cpu::expert_layout().blob_offset(layer, expert);
}

ArenaExpertSource::BlobStats ArenaExpertSource::take_blob_stats() {
    BlobStats s = blob_stats_;
    blob_stats_ = BlobStats{};
    return s;
}

// R8: the stats-free core of `read_blob`, safe to call from several staging readers at once.  A tier-resident
// blob memcpy's out of the locked arena (read-only); anything else is one whole-blob `pread` out of the file.
int64_t ArenaExpertSource::read_blob_raw(int64_t layer, int64_t expert, void* dst) {
    if (base_ == nullptr || layer < 0 || expert < 0 || expert >= n_expert_) return -1;
    const int64_t idx = layer * n_expert_ + expert;
    if (idx < 0 || idx >= blobs_) return -1;
    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
    const uint64_t len = lay.blob_bytes(layer);
    if (len == 0) return -1;
    {   // R8b: the lookup, the copy and the touch hold `dc_mu_` TOGETHER, because the host thread's
        // `dc_stage_admit` can evict this slot between the lookup and the copy - a memcpy out of a freed
        // slot would hand the staging buffer torn bytes that reach the GPU as a plausible, wrong expert.
        std::lock_guard<std::mutex> lk(dc_mu_);
        if (!hot_slot_.empty() && hot_slot_[(size_t) idx] >= 0) {
            std::memcpy(dst, hot_arena_ + hot_slot_[(size_t) idx], (size_t) len);
            dc_touch_locked((int32_t) ((uint64_t) hot_slot_[(size_t) idx] / dc_slot_bytes_));
            return (int64_t) len;
        }
    }
    if (file_fd_ < 0 || file_map_ == nullptr) return -1;
    const uint64_t off = lay.blob_offset(layer, expert);
    if (off + len > file_map_bytes_) return -1;
    uint8_t* p = (uint8_t*) dst;
    uint64_t done = 0;
    while (done < len) {
        const ssize_t r = ::pread(file_fd_, p + done, (size_t) (len - done), (off_t) (off + done));
        if (r <= 0) return -1;
        done += (uint64_t) r;
    }
    // R8.2: same page-cache argument as the ring reader - the staging buffer holds the bytes, the cache
    // copy only fuels reclaim churn (see pf_worker).
    ::posix_fadvise(file_fd_, (off_t) off, (off_t) len, POSIX_FADV_DONTNEED);
    return (int64_t) len;
}

// R5-fetch: read one whole blob into `dst`.  The hot tier memcpy's out of the locked arena; the file-backed
// arena issues ONE sequential `pread`, which the drive serves at its full rate - against ~340 us of latency
// per 4 KiB page fault when the pool walks the mapping instead (see `pin_hot`'s comment for the measurement).
int64_t ArenaExpertSource::read_blob(int64_t layer, int64_t expert, void* dst) {
    if (base_ == nullptr || layer < 0 || expert < 0 || expert >= n_expert_) return -1;
    const int64_t idx = layer * n_expert_ + expert;
    if (idx < 0 || idx >= blobs_) return -1;
    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
    const uint64_t len = lay.blob_bytes(layer);
    if (len == 0) return -1;
    if (!hot_slot_.empty() && hot_slot_[(size_t) idx] >= 0) {
        std::memcpy(dst, hot_arena_ + hot_slot_[(size_t) idx], (size_t) len);
        ++prefetched_;
        return (int64_t) len;
    }
    const int64_t r = read_blob_raw(layer, expert, dst);
    if (r < 0) return -1;
    ++prefetched_;
    ++blob_stats_.disk;
    blob_stats_.disk_bytes += (int64_t) len;
    return r;
}

// R5-fetch: the persistent reader threads.  The protocol is ExpertPool's (pool.cpp): a worker counts itself
// parked BEFORE its first wait so the host's publish below can never outrun a thread that is still leaving
// the previous batch's claim loop - a stolen or double-claimed job would make `pf_done_` overshoot the batch
// size and the host would wait forever.  R5b: the park is a BLOCKING condvar wait, not a spin - the readers
// idle most of each layer (the pool computes), and spinning threads on the SMT siblings measurably slow the
// compute workers down.
void ArenaExpertSource::pf_worker() {
    uint32_t seen = 0;
    for (;;) {
        {
            std::unique_lock<std::mutex> lk(pf_mu_);
            pf_parked_.fetch_add(1, std::memory_order_acq_rel);   // arrive at the park before the first wait
            pf_cv_.wait(lk, [&] { return pf_epoch_.load(std::memory_order_acquire) != seen ||
                                         pf_stop_.load(std::memory_order_relaxed); });
            pf_parked_.fetch_sub(1, std::memory_order_acq_rel);   // leaving the park
        }
        if (pf_stop_.load(std::memory_order_acquire)) return;
        seen = pf_epoch_.load(std::memory_order_relaxed);
        for (;;) {
            const uint32_t i = pf_head_.fetch_add(1, std::memory_order_relaxed);
            if (i >= pf_njobs_.load(std::memory_order_acquire)) break;
            const PfJob& j = pf_jobs_[(size_t) i];
            uint64_t done = 0;
            while (done < j.len) {
                const ssize_t r = ::pread(file_fd_, j.dst + done, (size_t) (j.len - done), (off_t) (j.off + done));
                if (r <= 0) break;
                done += (uint64_t) r;
            }
            // R8.2: **DROP THE PAGE-CACHE COPY IMMEDIATELY.**  A whole-blob `pread` leaves ~530 cached pages
            // behind, and on a 30 GiB box that already runs a 22 GiB mlocked tier there is no cache left to
            // hold them: the kernel allocates, clears, copies and then RECLAIMS every page of every miss.
            // perf during a diverse 300-token generation measured ~39% of the engine's CPU inside kernel
            // page-management on exactly this cycle.  The bytes the ring needed are in the ring; the cache
            // copy serves nobody (decode re-uses go through the LRU tier, prefill staging re-reads nothing).
            ::posix_fadvise(file_fd_, (off_t) j.off, (off_t) j.len, POSIX_FADV_DONTNEED);
            pf_done_.fetch_add(1, std::memory_order_release);
        }
    }
}

// R5-fetch.  Called with a layer's routing ids before the pool sets up its jobs.  The misses of the layer
// (what neither the hot tier nor a previous read can answer) are read WHOLE into the ring with `pf_workers_`
// concurrent preads - the drive streams them back-to-back at its sequential rate while the host finishes
// dispatching, instead of the drain faulting 4 KiB at a time and waiting on each page.  The ring is one
// layer deep: the dispatch contract has the pool block until this layer's drain is done, so the slots are
// free the next time this runs.  Blobs beyond the ring (more unique misses than slots - not reachable with
// this model's k=10 (+shared) and MAXT=4, but never say never) fall back to one WILLNEED each.
void ArenaExpertSource::begin_layer(int64_t layer, const int32_t* ids, int64_t k) {
    if (base_ == nullptr || ids == nullptr || k <= 0) return;
    if (layer < 0 || layer >= strata::kernels::cpu::expert_layout().n_layers) return;
    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
    if ((int64_t) pf_seen_.size() != n_expert_) pf_seen_.assign((size_t) n_expert_, 0);
    else std::fill(pf_seen_.begin(), pf_seen_.end(), (uint8_t) 0);
    if (!pf_started_ && file_fd_ >= 0 && file_map_ != nullptr) {
        // R7: the predictor is on by default; `STRATA_NO_PREDICT=1` is the A/B arm.
        // R8: off by default (see pf_window_start); STRATA_PREDICT=1 turns it back on.
        prefetch_predict_ = (std::getenv("STRATA_PREDICT") != nullptr);
        ring_.assign(kRingSlots, {});
        for (auto& r : ring_) r.resize((size_t) lay.max_blob + 512);
        ring_of_.assign((size_t) blobs_, -1);
        pf_jobs_.resize((size_t) kRingSlots * (size_t) kMaxChunks);
        for (int i = 0; i < 10; ++i) pf_workers_.emplace_back([this] { pf_worker(); });
        pf_started_ = true;
    }
    if (!pf_started_) {
        // resident arena or no file: nothing to fetch, but keep the dedupe+stats behaviour
        return;
    }
    // R7: a new pass starts at layer 0, and the previous pass's ring is not valid any more.  Without this the
    // `layer == ring_layer_` gate in `blob()` would serve a stale blob to anything that asks for the last
    // decode layer's index (the prefill suffix of a KEEP request is the caller that can reach it).
    if (layer == 0) {
        ring_layer_ = -1;
        pf_window_start();
    }
    ring_layer_ = layer;
    ring_waiting_ = false;
    dc_admit_list_.clear();
    int n = 0, njobs = 0;                  // n = blobs claimed, njobs = sub-reads published
    int64_t miss_bytes = 0;
    ++blob_stats_.calls;
    for (int64_t i = 0; i < k && n < kRingSlots; ++i) {
        const int64_t e = ids[i];
        if (e < 0 || e >= n_expert_ || pf_seen_[(size_t) e]) continue;
        pf_seen_[(size_t) e] = 1;
        ++blob_stats_.entries;
        const int64_t idx = layer * n_expert_ + e;
        if (!hot_slot_.empty() && hot_slot_[(size_t) idx] >= 0) {
            // R8: a resident expert must not keep a stale ring claim from an earlier window - with the
            // dynamic tier an expert CAN become resident between windows, and a stale `ring_of_` would both
            // misroute it to pass 1 and (once the slot is reused by another layer) serve the WRONG BYTES.
            ring_of_[(size_t) idx] = -1;
            ++blob_stats_.hot_skips;
            continue;
        }
        const uint64_t len = lay.blob_bytes(layer), off = lay.blob_offset(layer, e);
        if (len == 0 || off + len > file_map_bytes_) continue;
        const int slot = n++;
        // R8: **THE RING READ LANDS DIRECTLY IN AN LRU SLOT.**  A miss is admitted to the tier before the
        // read is issued, so the commit in `wait_layer` is one pointer publish - not a 2.2 MB copy per miss
        // per layer.  The LRU epoch is stamped at allocation, so a later claim in the SAME begin_layer call
        // can never evict an earlier one, and no victim can be a blob this window is still using.
        uint8_t* dst = ring_[(size_t) slot].data();
        if (dc_slots_ > 0) {
            const int32_t dc = dc_alloc(idx);
            if (dc >= 0) {
                dst = hot_arena_ + (uint64_t) dc * dc_slot_bytes_;
                dc_admit_list_.push_back(((int64_t) dc << 32) | (int64_t) (uint32_t) idx);
            } else {
                ++dc_fallbacks_;   // every slot touched this window: legacy ring, no admission
            }
        }
        // kSplit 1 must be exactly one read of the whole blob - `kChunk` only caps the SPLIT case.  Getting
        // this wrong left the engine issuing five 512 KiB reads per blob, which measured 8.5 tok/s against
        // 10.7 for the whole-blob form.
        const uint64_t step = kSplit <= 1 ? len
                              : std::min<uint64_t>(kChunk, (len + (uint64_t) kSplit - 1) / (uint64_t) kSplit);
        for (uint64_t at = 0; at < len && njobs < kRingSlots * kMaxChunks; at += step) {
            const uint64_t n = std::min<uint64_t>(step, len - at);
            pf_jobs_[(size_t) njobs++] = {dst + at, off + at, n};
        }
        ring_of_[(size_t) idx] = slot;
        pf_record_miss(idx);
        miss_bytes += (int64_t) len;
    }
    if (njobs == 0) return;
    // The rest: entries that are NOT resident and did NOT get a ring slot - i.e. a layer whose distinct
    // non-resident experts exceed `kRingSlots`.  They get one whole-blob WILLNEED each.
    //
    // **R7: THE HOT-TIER GUARD BELOW IS THE FIX FOR A 6.4x READ AMPLIFICATION.**  `pf_seen_[e]` is set to 1
    // for EVERY deduped id above, hot ones included, so this loop walked all of them; the only filter was
    // `ring_of_[idx] >= 0`, and a HOT expert never got a ring slot, so its test passed and the engine asked
    // the kernel to read it off the SSD into the page cache.  Measured on the IQ3_XXS pack, `--hot-ram-gib
    // 12`, one 150-token generation: 2,880 begin_layer calls, 67,748 ids seen, 52,741 hot-skipped, **96,231
    // blobs fetched** - against the 15,007 that were actually missed.  The 81,224 extra blobs were 176.8 GB
    // of readahead for experts already resident in RAM, and they were the whole of the decode time (209.4 GB
    // in 62.5 s = 3.35 GB/s, exactly the window time).  A resident expert must not be re-read from the drive,
    // whatever the reason.
    for (int64_t i = 0; i < k; ++i) {
        const int64_t e = ids[i];
        if (e < 0 || e >= n_expert_ || !pf_seen_[(size_t) e]) continue;
        pf_seen_[(size_t) e] = 2;       // mark as handled; ring_of_ decides who is actually served
        const int64_t idx = layer * n_expert_ + e;
        if (!hot_slot_.empty() && hot_slot_[(size_t) idx] >= 0) continue;   // RESIDENT: nothing to read, ever
        if (ring_of_[(size_t) idx] >= 0) continue;
        const uint64_t len = lay.blob_bytes(layer), off = lay.blob_offset(layer, e);
        if (len == 0 || off + len > file_map_bytes_) continue;
        ::posix_fadvise(file_fd_, (off_t) off, (off_t) len, POSIX_FADV_WILLNEED);
        ++prefetched_;
        ++blob_stats_.disk;
        blob_stats_.disk_bytes += (int64_t) len;
        pf_record_miss(idx);
    }
    // publish the batch; the WAIT is now `wait_layer`, so the caller can put the hot tier's own expert work in
    // front of it.  The parked-count barrier (ExpertPool's protocol) guarantees every reader thread is parked
    // before head_/done_/njobs_ are reset, so no thread can touch the previous batch's state while it is
    // republished.  `pf_njobs_` is published together with `pf_pending_`, and `ring_pending` reads that flag,
    // so no entry can be observed as "still coming" after its bytes have landed.
    while (pf_parked_.load(std::memory_order_acquire) != (uint32_t) pf_workers_.size()) _mm_pause();
    pf_head_.store(0, std::memory_order_relaxed);
    pf_done_.store(0, std::memory_order_relaxed);
    pf_jobs_active_ = njobs;
    pf_njobs_.store((uint32_t) njobs, std::memory_order_release);
    pf_epoch_.fetch_add(1, std::memory_order_release);
    { std::lock_guard<std::mutex> lk(pf_mu_); pf_cv_.notify_all(); }
    blob_stats_.disk += n;
    blob_stats_.disk_bytes += miss_bytes;
    ring_waiting_ = true;
}

// R7: warm the page cache for a layer's whole expert set.  Prefill calls this once per layer, with the
// experts it is about to stage, before the first staging read.  See the header for why: prefill is QD1 and
// measured 0.67 GB/s on a 33k-token prompt, i.e. it was latency-bound exactly like decode was.
void ArenaExpertSource::prefetch(int64_t layer, const int32_t* experts, int64_t n) {
    if (file_fd_ < 0 || file_map_ == nullptr || experts == nullptr || n <= 0) return;
    if (layer < 0 || layer >= strata::kernels::cpu::expert_layout().n_layers) return;
    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
    const uint64_t len = lay.blob_bytes(layer);
    if (len == 0) return;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t e = experts[i];
        if (e < 0 || e >= n_expert_) continue;
        const int64_t idx = layer * n_expert_ + e;
        if (!hot_slot_.empty() && hot_slot_[(size_t) idx] >= 0) continue;   // the tier answers it from RAM
        const uint64_t off = lay.blob_offset(layer, e);
        if (off + len > file_map_bytes_) continue;
        ::posix_fadvise(file_fd_, (off_t) off, (off_t) len, POSIX_FADV_WILLNEED);
        ++prefetched_;
    }
}

// R7: the blocking half of `begin_layer`.  Same wait the old single-phase form did inline.
// R7: a new verify window.  Score the previous window as a predictor of this one, then (optionally) ask the
// kernel to start reading the predicted set NOW, so the per-layer `pread` finds it in the page cache instead of
// paying the drive's QD1 latency 48 times.
//
// WHY THIS IS THE RIGHT LEVER.  Measured on this box's SNV2S1000G: one 2.08 MB random read costs **2.82 ms**
// (740 MB/s) at QD1, and the same read at QD2+ costs 0.06 ms once the page is resident.  The engine needs ~1-2
// such reads per layer and there are 48 layers, so the drive's LATENCY - not its bandwidth - sets the floor:
// 48 x 2.9 ms = 139 ms per window.  `POSIX_FADV_WILLNEED` on the predicted blobs moves that work to the start
// of the window where it can run at full queue depth beside the CPU, and the 106-odd syscalls cost microseconds.
void ArenaExpertSource::pf_window_start() {
    ++window_epoch_;
    // R8.1: decay the frequency counts.  Half-life = one window: a blob hit last window keeps most of its
    // credit, a blob untouched for eight windows has none, and the tier follows the workload as it moves.
    // ~10,840 shifts per window - noise.
    for (size_t i = 0; i < dc_count_.size(); ++i) dc_count_[i] >>= 1;
    // R8: the fadvise predictor is OFF by default now.  With the LRU tier, a blob that misses again is
    // ADMITTED (the ring reads it into a tier slot), so warming its page cache with WILLNEED only duplicates
    // the read - and on a cold page cache each call costs ~2 ms of synchronous kernel work, which at ~1,600
    // misses per adapting window was seconds per window of pure syscall.  STRATA_PREDICT=1 restores the R7
    // behaviour for A/B.
    if (window_epoch_ > 1 && prefetch_predict_ && file_fd_ >= 0) {
        for (const int32_t idx : last_misses_) {
            if (idx < 0 || idx >= blobs_) continue;
            const int64_t l = idx / n_expert_, e = idx % n_expert_;
            if (!hot_slot_.empty() && hot_slot_[(size_t) idx] >= 0) continue;
            const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
            const uint64_t off = lay.blob_offset(l, e), len = lay.blob_bytes(l);
            if (len == 0 || off + len > file_map_bytes_) continue;
            ::posix_fadvise(file_fd_, (off_t) off, (off_t) len, POSIX_FADV_WILLNEED);
            ++blob_stats_.win_prefetched;
        }
    }
    last_misses_.swap(cur_misses_);
    cur_misses_.clear();
}

void ArenaExpertSource::pf_record_miss(int64_t idx) {
    if (idx < 0 || idx >= blobs_) return;
    if (miss_epoch_.size() != (size_t) blobs_) miss_epoch_.assign((size_t) blobs_, 0);
    if (miss_epoch_[(size_t) idx] == window_epoch_) return;              // already recorded this window
    if (miss_epoch_[(size_t) idx] == window_epoch_ - 1) ++blob_stats_.win_repeat;
    miss_epoch_[(size_t) idx] = window_epoch_;
    cur_misses_.push_back((int32_t) idx);
    ++blob_stats_.win_misses;
}

void ArenaExpertSource::wait_layer() {
    if (ring_waiting_) {
        const int n = pf_jobs_active_;
        while (pf_done_.load(std::memory_order_acquire) != (uint32_t) n) _mm_pause();
        prefetched_ += n;
        ring_waiting_ = false;
    }
    // R8: commit this layer's admissions.  The reads are confirmed landed, so publishing the slot offsets
    // now turns every one of them into a RAM hit for the rest of the session - and every `blob()` after
    // this point (the pass-1 dispatch) serves the tier copy, never the ring.
    if (!dc_admit_list_.empty()) {
        for (const int64_t v : dc_admit_list_) {
            const int32_t s = (int32_t) ((uint64_t) v >> 32);
            const int64_t idx = (int64_t) (int32_t) ((uint64_t) v & 0xffffffffu);
            if (s < 0 || s >= (int32_t) dc_slots_ || idx < 0 || idx >= (int64_t) hot_slot_.size()) continue;
            hot_slot_[(size_t) idx] = (int64_t) ((uint64_t) s * dc_slot_bytes_);
            ++dc_admits_;
        }
        dc_admit_list_.clear();
    }
}

// R7: is `(layer, expert)`'s blob one of the reads this layer submitted to the ring?
//
// **THIS MUST NOT TEST `ring_waiting_`.**  That flag means "wait_layer has not run yet", and it is cleared
// *between* dispatch pass 0 and pass 1 - so a version of this predicate that checked it made every pass-1
// entry look like a pass-0 entry, pass 1 skipped all of them, and **~19% of this model's expert
// contributions were silently dropped**: measured on the IQ3_XXS pack, 97 windows, 78,855 routed ids,
// 63,930 blob() calls - exactly the hot-tier count, i.e. not one miss was ever dispatched.  The model still
// produced fluent English, which is precisely why it had to be found in the counters rather than in the
// output; it is also the likely source of the capability eval's arithmetic misses.  A blob is "pending"
// when it was claimed for THIS layer, whatever stage of the two-pass handoff it is at.
bool ArenaExpertSource::ring_pending(int64_t layer, int64_t expert) const {
    if (layer != ring_layer_) return false;
    if (expert < 0 || expert >= n_expert_) return false;
    const int64_t idx = layer * n_expert_ + expert;
    if (idx < 0 || idx >= blobs_ || (int64_t) ring_of_.size() != blobs_) return false;
    return ring_of_[(size_t) idx] >= 0;
}

}  // namespace strata::core
