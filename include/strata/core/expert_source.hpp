// include/strata/core/expert_source.hpp - P2.S3/P2.S5: the expert pool's adapter to the host loop.
//
// `session_loop` publishes, per layer, the NORMED activation `x_f`, the ten routed expert ids and their router
// weights, and expects `k x n_embd` floats back.  This is the piece that turns those four things into an
// `ExpertPool::run` call.  Nothing here is clever, and that is the point: the pool, the kernel and the loop are
// each already verified, so this file has exactly one job - get the CONTRACT between them right.
//
// THE CONTRACT, and each clause is a way to be wrong:
//
//   1. **`x_f` IS THE SAME PINNED BUFFER EVERY LAYER.**  Its ADDRESS never changes, so the `ActQ` cannot be
//      cached by pointer - a cache keyed on `x_f` would quantize layer 0 and reuse it for all 47 remaining
//      layers, which is a finite, plausible, completely wrong token.  There is no cache here at all: the
//      conversion is one 2560-element pass against a 0.3 ms/layer budget, and a correct answer is worth more
//      than the microseconds.
//   2. **THE POOL DOES NOT APPLY THE ROUTER WEIGHT.**  `moe_combine` (`src/core/layer.cpp:479`) sums
//      `w[i] * parts[i]` on the device.  `ExpertJob::weight` is a diagnostic field; setting it here would
//      apply the weight twice, which is invisible in a single layer and compounds over 48.
//   3. **THE KERNEL'S GEOMETRY IS FIXED BY THE ARTIFACT** (`H = 2560`, `FF = 640`, `BLOB = 1,382,400`).  A
//      different `n_embd` or a different expert width is REFUSED rather than mis-indexed, because the blob's
//      internal offsets are compile-time constants and reading a 640-wide expert as a 2560-wide one walks off
//      the end into the next expert's bytes without faulting.
#pragma once

#include "strata/core/expert_cache.hpp"
#include "strata/core/hit_hook.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <new>
#include <cstdint>
#include <cstddef>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <string>
#include <thread>
#include <vector>

namespace strata::kernels::cpu {
struct ExpertLayout;
}

namespace strata::core {

/// R20: a stateless allocator that hands out storage aligned to `kAlign` bytes.  O_DIRECT requires the BUFFER
/// address to be aligned too, not just the file offset and length, so the read-ring slots have to come from
/// somewhere that guarantees it - `std::vector<uint8_t>`'s allocator does not, which is why every decode miss
/// used to be read into a bounce buffer and memcpy'd into place.  512 is the strictest alignment this needs to
/// satisfy (the device's logical block size); the probe in `expert_source.cpp` only relaxes the offset/length
/// test to it once a real direct read has succeeded, and a destination this aligned qualifies either way.
template <class T>
struct AlignedBuf {
    static constexpr std::size_t kAlign = 512;
    using value_type = T;

    AlignedBuf() = default;
    template <class U> AlignedBuf(const AlignedBuf<U>&) noexcept {}

    T* allocate(std::size_t n) {
        void* p = nullptr;
        const std::size_t bytes = n * sizeof(T);
        // posix_memalign wants the alignment to be a power-of-two multiple of sizeof(void*): 512 is.
        if (bytes == 0) return nullptr;
        if (posix_memalign(&p, kAlign, bytes) != 0) throw std::bad_alloc();
        return static_cast<T*>(p);
    }
    void deallocate(T* p, std::size_t) noexcept { std::free(p); }

    template <class U> struct rebind { using other = AlignedBuf<U>; };
};

template <class A, class B>
bool operator==(const AlignedBuf<A>&, const AlignedBuf<B>&) noexcept { return true; }
template <class A, class B>
bool operator!=(const AlignedBuf<A>&, const AlignedBuf<B>&) noexcept { return false; }
class PeerExperts;   // multi-GPU: the second GPU's expert tier (peer_experts.hpp)
class RemoteExperts;
struct LoadStats;

namespace detail {

/// Sentinel used by the pure complement planner for a blob that remains in the mmap fallback.
inline constexpr uint64_t kNoCacheComplement = ~uint64_t{0};

/// Required cgroup-v2 usage counters for the conservative cache-reclaim allowance.
struct CgroupMemoryStat {
    uint64_t current = 0;
    uint64_t inactive_file = 0;
    uint64_t file_dirty = 0;
    uint64_t file_writeback = 0;
    bool valid = false;
};

/// Calculate additional bytes under a finite cgroup limit after reclaiming only clean inactive file cache.
/// Returns false when the required memory.stat counters were unavailable.
bool cgroup_available_bytes(uint64_t limit, const CgroupMemoryStat& stat, uint64_t& bytes);

/// #633: the RAM this process can get.  `available`: MemAvailable (Windows: the available physical memory), lowered
/// to the room under the tightest cgroup limit; `cgroup_limit`: that tightest limit itself (v2 memory.max of the group
/// and its ancestors, or v1 memory.limit_in_bytes), ~0 when there is none - what a container can never exceed.
struct HostMemory {
    uint64_t available = 0;
    uint64_t cgroup_limit = ~uint64_t{0};
};

/// Linux reads `meminfo`, `self_cgroup` and the cgroup tree under `cgroup_root` (the parameters are for tests; the
/// defaults are the real files).  cgroup v2 as before (an unreadable limit of a group that has one fails); cgroup v1's
/// memory controller (`<root>/memory/<path>`: memory.limit_in_bytes - memory.usage_in_bytes); no cgroup line at all is
/// MemAvailable alone.  False when the RAM cannot be determined.
bool host_available_memory(HostMemory& m, const std::string& meminfo = "/proc/meminfo",
                           const std::string& self_cgroup = "/proc/self/cgroup",
                           const std::string& cgroup_root = "/sys/fs/cgroup");

/// Build compact offsets for experts absent from both the primary GPU cache and an optional second GPU tier.
/// Kept CPU-only so selection and byte accounting can be tested without initializing a GPU.
bool make_cache_complement_plan(
    int64_t n_layers, int64_t n_expert, const std::vector<uint64_t>& layer_blob_bytes,
    const std::vector<std::pair<int32_t, int32_t>>& primary_gpu_pairs,
    const std::vector<std::pair<int32_t, int32_t>>& additional_gpu_pairs,
    std::vector<uint64_t>& offsets, uint64_t& bytes, std::string& err);

/// Resolve one blob through the compact copy when present, otherwise preserve its exact mapped-file fallback.
const uint8_t* cache_complement_blob_or_fallback(
    size_t index, const std::vector<uint64_t>& offsets, const uint8_t* complement_host,
    const uint8_t* mapped_fallback);

/// The resident RAM mode: which GPU-cache slots' experts are kept in RAM too.  The prompt path lends the cache's
/// LAST slots (from `lend_from` on; a short prompt lends only the last few), and a lent slot's expert is streamed
/// from RAM during the prompt and copied back into its slot after it.  `base_bytes` (every expert no slot holds)
/// must fit `budget`; slots are then added from the end down to `lend_from` while they still fit.  Returns the
/// first slot kept in RAM (`slot_bytes.size()` = none), or -1 when `base_bytes` alone exceeds `budget`.
int64_t choose_resident_keep_from(const std::vector<uint64_t>& slot_bytes, uint64_t base_bytes, uint64_t budget,
                                  int64_t lend_from);

/// The adaptive tier swapped `in` into a GPU slot and `out` out of it: `out` takes `in`'s place in the compact copy
/// (the caller copies out's bytes there).  False, and nothing changed, unless `in` is in the copy and `out` is not.
bool exchange_cache_complement(std::vector<uint64_t>& offsets, size_t in, size_t out);

}  // namespace detail



/// Where one routed expert's bytes come from.
///
/// Phase 2 has NO cache (`phase-2-correct-engine.md`: hit rate `h = 0`), so the only implementation is a
/// file-backed reader.  The interface exists anyway because Phase 3 replaces exactly this object with the VRAM
/// cache, and because a test can supply an in-memory source without a 34 GB artifact.
class ExpertSource {
public:
    virtual ~ExpertSource() = default;

    /// The expert-layout blob for `(layer, expert)`, or nullptr if it cannot be produced.
    ///
    /// The pointer only has to stay valid until the next `blob()` call: with `h = 0` every expert is computed
    /// immediately and nothing is retained.  A CACHING source must return pointers into the cache, not into a
    /// reused staging buffer - otherwise the pool would read the next expert's bytes while computing this one.
    virtual const uint8_t* blob(int64_t layer, int64_t expert) = 0;

    /// Blobs touched, for the driver to report.  A source that does not count returns 0.
    virtual int64_t reads() const { return 0; }

    /// Plan v0.3 P5: whether `blob(layer, expert)` lies in page-locked, CUDA-registered memory, so an
    /// asynchronous host-to-device copy can DMA it directly (no staging copy on the CPU).
    virtual bool pinned(int64_t layer, int64_t expert) const { (void) layer; (void) expert; return false; }

    /// Called once before the first expert of a layer.  A source that reads from disk wants to start the read
    /// here so it overlaps the quantisation, and a prefetching source in Phase 3 wants the ids.
    virtual void begin_layer(int64_t layer, const int32_t* ids, int64_t k) { (void) layer; (void) ids; (void) k; }
    /// R5-fetch: read the whole blob for `(layer, expert)` into `dst` (which the caller sizes to at least
    /// `blob_bytes(layer)`), by-passing the mapped pointer.  A file-backed arena answers with one sequential
    /// `pread` (the drive's own throughput) or a memcpy when the hot tier holds the blob; a resident arena
    /// memcpy's.  Returns the bytes read, or -1 when the source cannot (the caller then uses `blob()`).
    virtual int64_t read_blob(int64_t layer, int64_t expert, void* dst) { (void) expert; (void) dst; (void) layer; return -1; }
    /// R8: `read_blob` without the stats bookkeeping, so several staging readers can call it concurrently
    /// (prefill's reader pool).  Same contract; -1 when the source cannot produce the blob this way.
    virtual int64_t read_blob_raw(int64_t layer, int64_t expert, void* dst) {
        (void) layer; (void) expert; (void) dst; return -1;
    }
    /// R8b: prefill admission hook (see `ArenaExpertSource::dc_stage_admit`).  A no-op for sources without
    /// an adaptive tier.
    virtual bool stage_admit(int64_t layer, int64_t expert, const uint8_t* bytes) {
        (void) layer; (void) expert; (void) bytes; return false;
    }
    /// **R7: SPLIT THE WAIT FROM THE READ SO IT CAN BE HIDDEN.**  `begin_layer` used to submit the layer's
    /// whole-blob reads AND block until they landed, which put the drive's latency on the critical path in front
    /// of the CPU's own drain - measured on the IQ3_XXS pack, 176 ms of a 272 ms window where the CPU had 57 ms
    /// of work it could have been doing.  A source that prefetches now returns from `begin_layer` with the reads
    /// IN FLIGHT, answers `ring_pending` for the entries that are still coming, and blocks in `wait_layer`.
    /// Callers that cannot interleave anything (`expert_pool_dispatch`'s single-token path) simply call
    /// `wait_layer` immediately, which is exactly the old behaviour.
    virtual void wait_layer() {}
    /// True when `(layer, expert)`'s bytes are being read and are NOT yet valid.  A source with no prefetch
    /// answers false for everything, so the split reduces to the old code path.
    virtual bool ring_pending(int64_t layer, int64_t expert) const { (void) layer; (void) expert; return false; }
    /// **R7: ASK THE KERNEL TO START READING THIS LAYER'S EXPERTS NOW.**  Prefill stages one blob at a time
    /// on the host thread, so - exactly like decode before the two-pass split - the drive sits at QD1 and a
    /// 33k-token prompt measured 498 s for 330 GB of reads (0.67 GB/s, the drive's QD1 rate).  Prefill knows
    /// the whole layer's expert set up front and processes it in file-offset order, so a WILLNEED over that
    /// list turns the QD1 stream into near-sequential readahead at the drive's full rate and the staging
    /// reads become page-cache hits.  A source without a file is a no-op.
    virtual void prefetch(int64_t layer, const int32_t* experts, int64_t n) { (void) layer; (void) experts; (void) n; }
    /// Plan v0.3 P6: the DEVICE address of a pinned, mapped blob (the GPU can read it over PCIe), or null.
    virtual const uint8_t* device_alias(int64_t layer, int64_t expert) const { (void) layer; (void) expert; return nullptr; }
    /// R9b: can this source feed the GPU's PCIe expert path at all?  The plan builder needs a cheap per-layer
    /// gate that does not depend on one arbitrary expert being resident (the old test,
    /// `device_alias(layer, 0) != nullptr`, flapped per layer with an LRU tier).
    virtual bool pcie_ready() const { return device_alias(0, 0) != nullptr; }
    /// A file-backed source: start reading this expert's pages now (it will be needed by the CPU); no-op elsewhere.
    virtual void prefetch(int64_t layer, int64_t expert) { (void) layer; (void) expert; }
    /// A file-backed source: this expert lives in VRAM, so its pages need not stay in RAM - hand them back to the
    /// kernel (a later read re-reads the file; no result depends on it).  Returns the bytes released; 0 elsewhere.
    virtual uint64_t release(int64_t layer, int64_t expert) { (void) layer; (void) expert; return 0; }
    /// Whether the verify window may give the GPU a PCIe share of this layer's misses at all (each expert is still
    /// checked with `pinned`).  The arena answers per layer through its expert 0; the resident RAM mode's compact
    /// copy has no expert 0 when the GPU cache holds it, so it answers for the whole copy.
    virtual bool pcie_layer(int64_t layer) const { return device_alias(layer, 0) != nullptr; }

    /// CS-T: whether `blob(layer, expert)` would be assembled into a short-lived buffer (a native pack read from its
    /// GGUF shards in place, where an expert's gate, up and down rows are three separate slices).  Such a pointer
    /// stays valid for the layer it was asked in and the next one or two; a consumer that keeps a blob longer (the
    /// prompt path's stager queues a whole chunk) copies it with `copy_blob` instead.
    virtual bool transient(int64_t layer, int64_t expert) const { (void) layer; (void) expert; return false; }
    /// The blob's bytes into `dst` (blob_bytes(layer) of them).  Safe from several threads for a source whose
    /// `transient` can be true.
    virtual bool copy_blob(int64_t layer, int64_t expert, uint8_t* dst);
    /// The `n` experts of `layer` the CPU is about to ask `blob` for, all at once: a source that reads a file may
    /// fetch them in parallel.  The bytes `blob` then returns are the same.  Default: nothing.
    virtual void prefetch(int64_t layer, const int64_t* experts, int64_t n) { (void) layer; (void) experts; (void) n; }
    /// CS-T: the experts of `layer` a predictor expects next - a source that reads a file may start reading their
    /// pages now, in the background.  Only warms: what `blob` returns is unchanged.  Default: nothing.
    virtual void warm(int64_t layer, const int64_t* experts, int64_t n) { (void) layer; (void) experts; (void) n; }
    /// Whether `warm` does anything (the predictor is not run otherwise).
    virtual bool warms() const { return false; }
};

/// CS-T, routing-aware prefetch of the file tier: when the CPU pool starts layer `l`, a worker thread applies layer
/// l+1's router (BF16, host copy) to layer l's MoE input - the residual stream changes little from one layer to the
/// next - takes each token's top `k` experts, drops the ones the GPU cache or the RAM copy holds, and asks the
/// source to `warm` the rest, so their pages are on the way while layer l computes.  A prediction only warms pages:
/// it never changes which experts are computed or how.
class RouterLookahead {
public:
    RouterLookahead() = default;
    ~RouterLookahead();
    RouterLookahead(const RouterLookahead&) = delete;
    RouterLookahead& operator=(const RouterLookahead&) = delete;
    /// `routers[l]`: layer l's ffn_gate_inp as BF16 bits, n_expert rows of n_embd.
    bool start(std::vector<std::vector<uint16_t>> routers, int64_t n_embd, int64_t n_expert, int k, ExpertSource* src,
               std::string& err);
    /// Layer `layer`'s MoE input for `n_tok` tokens (host floats): predict and warm layer + 1.  Never waits: a
    /// prediction still running for an earlier layer makes this one skip.
    void submit(int64_t layer, const float* x, int64_t n_tok, const int32_t* host_res);
    int64_t predicted() const { return predicted_.load(std::memory_order_relaxed); }
    int64_t skipped() const { return skipped_.load(std::memory_order_relaxed); }
    double busy_ms() const { return (double) busy_us_.load(std::memory_order_relaxed) / 1000.0; }

private:
    void run();
    std::vector<std::vector<uint16_t>> routers_;
    int64_t n_embd_ = 0, n_expert_ = 0;
    int k_ = 10;
    ExpertSource* src_ = nullptr;
    std::thread thread_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool quit_ = false, pending_ = false, busy_ = false;
    int64_t layer_ = -1, n_tok_ = 0;
    const int32_t* host_res_ = nullptr;
    std::vector<float> x_;
    std::atomic<int64_t> predicted_{0}, skipped_{0};
    std::atomic<uint64_t> busy_us_{0};
};

/// Plan v0.3 P6: what the GPU computes in a verify window's layer, written by the pool (mapped host memory) right
/// after the ring and published before the CPU starts its own share.  Groups of entries that share a blob; the
/// blob is a VRAM slot or a pinned host blob read over PCIe.
struct GpuPlanSink {
    int32_t* counts = nullptr;             ///< [0] groups, [1] entries
    int32_t* start = nullptr;              ///< cap + 1
    int32_t* dst = nullptr;                ///< cap: the entry's row of parts (token * k + j)
    int32_t* tok = nullptr;                ///< cap: the entry's token
    unsigned long long* ptr = nullptr;     ///< cap: the group's blob, device address
    /// Plan v0.3 P6 (DMA): the PCIe share is a second list of groups - `ptr2[q]` is staging slot q, `start2` indexes
    /// the same dst/tok entries - computed by the GPU after `fetch` has copied their blobs into staging with the
    /// copy engine.  counts[0] = VRAM groups, counts[1] = all entries, counts[2] = PCIe groups.
    unsigned long long* ptr2 = nullptr;
    int32_t* start2 = nullptr;
    unsigned long long staging = 0;
    int64_t staging_cap = 0;
    int64_t cap = 0;
    void (*publish)(void* ctx) = nullptr;
    /// Starts the DMA copies of `n` host blobs (pinned) into staging slots 0..n-1 and signals the GPU when they land.
    void (*fetch)(void* ctx, const uint8_t* const* src, int n, size_t bytes) = nullptr;
    void* ctx = nullptr;
    /// Plan v0.3 P6: how a PCIe group reaches the GPU.  0 = the copy engine stages it (DMA, `fetch`); 1 = the grouped
    /// kernel reads the mapped arena directly; 2 = a copy kernel stages it inside the graph.  For 1 and 2 `ptr2`
    /// holds the arena's device alias.
    int pcie_mode = 0;
};

/// The adapter's own state.  One per session, reused every layer so the token path allocates nothing (P2.T10).
struct ExpertDispatch {
    strata::kernels::cpu::ExpertPool* pool = nullptr;
    ExpertSource* src = nullptr;
    RouterLookahead* lookahead = nullptr;   ///< CS-T: warms the next layer's predicted file-tier experts
    RemoteExperts* remote[3] = {}; ///< optional CUDA1..3 tiers for otherwise CPU-served rows
    int remote_count = 0;
    int64_t n_expert = strata::kernels::cpu::NE;

    /// Counters, for the driver to report rather than for control flow.
    int64_t layers = 0;
    int64_t experts = 0;
    int64_t missing = 0;

    /// R21: --gpu-share.  0..256: the fraction (in 256ths) of a layer's DISTINCT experts - RAM-tier
    /// hits included, not just the VRAM-cache misses - routed to the GPU through the pinned arena's
    /// device alias, so the window's expert reads split across the CPU pool's cores and the PCIe path
    /// at the same time.  0 = the legacy policy (misses only, the last pcie_num/256 of them).
    int gpu_share = 0;
    int64_t pcie_residents = 0;   ///< R21: RAM-tier blobs this run sent over PCIe (vs the miss ones).

    // ================================ R4: THE VRAM TIER, MEASURED BEFORE IT IS USED =========================    //
    // **THE CACHE IS CONSULTED AND FILLED HERE, AND NOTHING IS COMPUTED FROM IT YET.**  That is deliberate and
    // it is `R4-design-note.md` §7 step 2: the *dispatch* is measured on its own before any kernel is written,
    // because a hit rate measured after the kernel exists cannot say whether a disappointing result was the
    // policy, the split or the kernel.
    //
    // So every expert is still computed by the CPU, exactly as before, and the run is numerically identical
    // with the cache on or off.  What changes is that the engine now reports **h on its own routing, on a real
    // workload** - which is the number `R4.1` asks for and which until now existed only from offline traces.
    //
    // The policy is compulsory-miss: the first time `(layer, expert)` is routed, if a slot is free it is
    // admitted and filled from the arena.  No eviction, because eviction policy is the measured question
    // (`R4.1`'s LFU-decay vs LRU sweep) and a placeholder would set the hit rate everything is sized against.
    ExpertCache* cache = nullptr;
    int64_t cache_hits = 0;      ///< lookups already resident
    int64_t cache_admitted = 0;  ///< lookups that took a slot
    int64_t cache_refused = 0;   ///< lookups with no slot free (the cache is full)
    void* cache_stream = nullptr;
    const char* cache_fail = nullptr;

    // ================================ R4.2c: THE HITS GO TO THE GPU =========================    //
    // **THE SPLIT IS BY ROUTER INDEX, AND THAT IS THE WHOLE TRICK.**  A layer routes ten experts; the resident
    // ones are computed on the GPU and the rest on the CPU, and both answers have to end up in `parts` at the
    // index the ROUTER gave them, because that is the order `moe_combine` weights against.  So the CPU's `out`
    // rows are zeroed for the hits, and each hit carries its routed index to the kernel as `dst`.
    //
    // Everything below is per-session state rather than per-call, so the token path allocates nothing (P2.T10).
    const uint8_t* cache_base = nullptr;   ///< the slot arena on the DEVICE
    int64_t cache_blob = 0;                ///< bytes per slot
    const uint64_t* cache_slot_off = nullptr;   ///< plan v0.3 P6: per-slot offsets when the slots differ in size
    void* hit_scratch = nullptr;           ///< `moe_hit_grouped_scratch_bytes(K, ...)`
    float* parts_out = nullptr;            ///< the graph's `parts` buffer, on the device
    /// Where the GPU's hits land, `K x n_embd`, DEVICE and separate from `parts_out` on purpose: see
    /// `HitPhase`.  Zeroed by the hit path each layer before the kernel writes it.
    float* hit_out = nullptr;
    int64_t parts_elems = 0;               ///< `K * n_embd`, the length of both buffers
    const float* mixed = nullptr;          ///< the layer's normed activation, for the hit kernel's Q8_0
    uint8_t* x_q8_0_hit = nullptr;         ///< `(n_embd/32) * 34` bytes, its own buffer
    /// **R4.2h: THE CPU's fp32 ACTIVATION SCALES, `(n_embd/32)` FLOATS.**  Without them the GPU's hits are
    /// computed with the `block_q8_0`'s fp16 `d` while the CPU's misses use the fp32 `ActQ::scale`
    /// (`cpu/expert.cpp:92`) - **4.761e-04 relative on 80 of 80 chunks**, measured with both real
    /// implementations linked in `bench/micro/act_quant_parity.cu`.  That disagreement is why turning the
    /// cache on changed the generated tokens.  Required whenever the hit path runs.
    float* x_q8_0_hit_scale = nullptr;
    int32_t* d_slot = nullptr;             ///< device, K entries
    int32_t* d_dst = nullptr;              ///< device, K entries
    std::vector<int32_t> h_slot, h_dst;    ///< host staging, sized at session setup
    /// **PER ROUTER INDEX, DECIDED IN `Launch` AND CONSUMED BY THE POOL.**  The two callbacks share it
    /// so the decision is made exactly once, on this layer's ids, and neither side can re-decide it.
    std::vector<uint8_t> is_hit;
    bool decided = false;
    /// Plan v0.3 P4 token graph: the STATIC residency table (`n_layers x n_expert`, slot or -1), the host's copy
    /// of what the device hit path reads.  When set, the pool leaves a resident expert's row at zero (the GPU
    /// computes it) without any `Launch` callback.
    const int32_t* host_res = nullptr;
    /// Plan v0.3 P4: split every expert by rows across the pool's threads (default on; A/B `--no-split-rows`).
    bool split_rows = true;

    // ================================ R4.2d: DID THE GPU ACTUALLY START? =========================    //
    // **THE HIT PATH IS 0.91 ms SLOWER AND THE ONLY EXPLANATION LEFT IS THAT IT NEVER OVERLAPS.**  The kernel
    // is 159 GB/s against the CPU's 35, the CPU's half of the drain falls 6 ms, and none of it reaches the
    // token - which is what it would look like if the enqueued hit kernels did not BEGIN until the host next
    // entered the driver, i.e. after `pool()` returned.  That is the third sighting of this driver behaving
    // that way (rounds 195/287 on the doorbell, round 36 on the head and sampler) and it decides whether R4 can
    // pay at all, so it gets measured rather than argued.
    //
    // `hit_done` is recorded on the stream straight after the hit kernel.  `Combine` - which runs after the
    // pool - queries it: SUCCESS means the GPU finished while the CPU was working, NOT-READY means it had not.
    // Same shape as the doorbell's `rings_mid_graph`, and for the same reason.
    void* hit_done = nullptr;      ///< cudaEvent_t, created at session setup
    int64_t hit_ready = 0;         ///< layers where the hit work was DONE by the time the pool returned
    int64_t hit_late = 0;          ///< layers where it was not
    /// The candidate fix, as an A/B arm: enter the driver once right after the hit launch.  If submission is
    /// lazy, this starts the GPU work before the pool instead of after it.
    bool hit_poke = false;
    bool hit_cpu_order = false;    ///< experimental CPU-order GPU expert arithmetic; opt-in only
    int64_t n_hits = 0;                    ///< this layer's hits
    /// Set by `Launch` and consumed by `Combine`, so a `Combine` with no `Launch` in front of it cannot
    /// add a stale buffer into `parts`.
    bool hit_pending = false;
    const char* hit_fail = nullptr;

    /// Whether the hit path is wired up.  All of it or none of it: a half-configured hit path would compute
    /// some experts twice and others not at all, which is a wrong token rather than an error.
    bool hits_ready() const {
        return cache != nullptr && cache_base != nullptr && parts_out != nullptr && hit_out != nullptr &&
               mixed != nullptr &&
               hit_scratch != nullptr && x_q8_0_hit != nullptr && x_q8_0_hit_scale != nullptr && d_slot != nullptr && d_dst != nullptr;
    }

    std::vector<strata::kernels::cpu::ExpertJob> jobs;
    strata::kernels::cpu::ActQ act;
    /// Plan v0.3 P6 verify window: one quantized activation per token, the multi-token jobs, and the expert ->
    /// job map (reset after every layer).
    std::vector<strata::kernels::cpu::ActQ> act_multi;
    /// Plan v0.3 P6: a native pack's per-token activations (MAXT x kNativeActBytes).
    std::vector<uint8_t> nact_multi;
    std::vector<strata::kernels::cpu::ExpertJobMulti> jobs_multi;
    std::vector<int16_t> job_of;
    /// Plan v0.3 P6: the verify window's GPU plan (VRAM hits + the PCIe share of the misses); `pcie_num`/256 of
    /// each layer's distinct missed experts (the last ones in routing order) are read by the GPU over PCIe.
    GpuPlanSink* plan = nullptr;
    int pcie_num = 0;
    int64_t pcie_experts = 0;      ///< distinct experts the GPU read over PCIe in verify windows
    double ms_plan = 0, ms_actq = 0, ms_jobs = 0, ms_run = 0, ms_wait = 0;   ///< verify-window dispatch sections
    /// #588: routed (token, expert) entries the GPU computed from outside its cache in verify windows: read over PCIe
    /// (--pcie-frac, kind 1) or on another GPU (kind 2).  In neither cache_hits nor cache_refused.
    int64_t offload_entries = 0;
    /// Plan v0.3 P6: decayed routing counts per (layer, expert) during decode (sized by the caller; empty = off),
    /// which the driver uses to swap the most-routed missing experts into the VRAM tier between rounds.
    std::vector<float> usage;
    /// R5: cumulative routed counts per (layer, expert), never decayed - what `--dump-profile` ranks into a
    /// STRP profile built from THIS model's actual routing on THIS machine.
    std::vector<uint64_t> usage_total;
    int64_t multi_misses = 0;      ///< distinct (layer, expert) pairs the CPU computed in verify windows
    int64_t multi_entries = 0;     ///< routed (token, expert) entries the CPU served in verify windows
    /// Multi-GPU: the second GPU's tier.  Its experts are computed there instead of on the CPU (kind 2).
    PeerExperts* peer = nullptr;
    int64_t peer_entries = 0;      ///< routed entries the peer served in verify windows
    /// Set when `dispatch` could not produce an answer.  The loop itself has no error channel, so this is
    /// where a source failure surfaces: the driver checks it after `session_loop` returns rather than the
    /// engine computing from a half-filled `parts`.
    ///
    /// **`failed` LATCHES AND `fail` IS ONLY THE MESSAGE.**  Clearing the message must not re-arm the adapter:
    /// a session that failed at layer 5 has already fed `moe_combine` whatever `parts` held, so every layer
    /// after it is built on a hole - and resuming into a plausible-looking token is the exact outcome this
    /// whole mechanism exists to prevent.  `failed` is what the short-circuit reads.
    bool failed = false;
    const char* fail = nullptr;
    int64_t fail_layer = -1;
    int64_t fail_expert = -1;
};

/// `strata::core::PoolFn`, exactly.  Silent on failure BY SIGNATURE - see `ExpertDispatch::fail`.
void expert_pool_dispatch(void* user, const float* x_f, const int32_t* ids, const float* weights, int64_t n_embd,
                          int64_t k, float* out);

/// Plan v0.3 P6: the pool for a verify window of `n_tok` tokens.  `x_f` is (n_tok, n_embd), `ids` (n_tok, k) and
/// `out` (n_tok * k, n_embd).  Each distinct missed expert is computed once for all the tokens routed to it;
/// resident experts' rows are zeroed (the GPU adds them).  Requires `host_res` (the token-graph residency).
void expert_pool_dispatch_multi(ExpertDispatch& d, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k,
                                float* out);

/// **THE HITS, LAUNCHED AFTER THE MISSES ARE STAGED AND BEFORE `post[l]`.**  Same shape as `PoolFn` and for the
/// same reason: `session_loop` owns the ORDER and this owns the work, so the loop needs to know nothing about
/// expert caches.  Returns immediately when there are no hits, which is every layer until the cache is warm.
///
/// It must run AFTER the host has copied the misses into `parts_dev` (it writes into the same buffer, on rows
/// the CPU zeroed) and BEFORE `post[l]` (which reads it).  Both are stream-ordered on the loop's own stream.
void expert_hit_run(void* user, void* stream, HitPhase phase, const int32_t* ids, int64_t k);
/// The pool half of the same decision; see `ExpertDispatch::is_hit`.

/// **PHASE 2'S ONLY SOURCE: `experts.bin`, memory-mapped, no cache.**
///
/// `experts.bin` is 33,973,862,400 B and `BLOB` is 1,382,400, so it holds exactly `48 x 512 = 24,576` blobs and
/// the index is `layer * 512 + expert` with NO padding.  A mapping is therefore the whole implementation: the
/// blob pointer is base plus a multiply, and the page fault that follows is the read.
///
/// **AND THAT IS NOT AS SLOW AS IT SOUNDS, WHICH IS THE POINT.**  The expert set is 34 GB and this machine has
/// 64 GB of DDR5, so a warm OS page cache holds ALL of it - the second token onward is a DRAM read at the
/// measured 44.14 GB/s, not a disk read.  The first token's 663.6 MB comes off the disk and is the cold-path
/// number; `benches` should report the two separately rather than blending them.
///
/// IT IS NOT AN LRU, AN LFU OR A PREFETCHER.  Phase 3 replaces this object with those.  Phase 2 is hit rate
/// `h = 0` on purpose, so that the CPU path and the host round trip are exercised on every layer of every
/// token - which is the only way they get debugged before speed matters.
class FileExpertSource : public ExpertSource {
public:
    FileExpertSource() = default;
    ~FileExpertSource() override;
    FileExpertSource(const FileExpertSource&) = delete;
    FileExpertSource& operator=(const FileExpertSource&) = delete;

    /// Maps `<pack_dir>/experts.bin` and checks its size against the loaded expert layout.  Canonical packs use
    /// `n_layers * n_expert * BLOB`; native packs use their variable per-layer blob sizes and offsets.
    ///
    /// The size check is not a formality: a short file would fault at the END of a long sequence, and an
    /// over-long one means the pack is not the one the geometry came from.  Refuses with the two numbers.
    ///
    /// CS-T: a native pack WITHOUT experts.bin, after `set_gguf`, maps the model's GGUF shards instead (every file
    /// native_experts.txt names, after `check_experts_gguf`), and assembles a blob from its three role slices when
    /// it is asked for: the SSD tier, read in place through the OS file cache, with no 30-77 GB experts.bin copy.
    /// With experts.bin present nothing changes.
    bool open(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, std::string& err);
    /// CS-T: the --native shard (native_experts.txt names the other files beside it); see `open`.
    void set_gguf(const std::string& native) { gguf_ = native; }
    /// Whether the experts are read from the GGUF shards in place (no experts.bin).
    bool gguf_mode() const { return !role_ptr_.empty(); }
    /// Pin a compact host mirror of experts absent from a fully filled static GPU cache. The mmap remains open
    /// as a fallback for later cache reloads. This is opt-in because the complement may still be a large allocation.
    ///
    /// The resident RAM mode (`--resident-experts`, `--resident-cpu-experts`):
    ///   - `pin`: page-locked and mapped (`cudaHostAlloc`), so the prompt path copies it by DMA and the verify
    ///     window may read a share of the misses over PCIe; when the driver refuses, ordinary memory locked in the
    ///     working set instead.  `pin = false` is ordinary pageable memory (the ROCm arm: large pinned allocations
    ///     can fail there, and it is what the HIP measurements used).
    ///   - `lend_from_slot` >= 0: the GPU-cache slots from there to the end are the prompt path's lend region; their
    ///     experts are kept in RAM too, from the last slot down, as far as `available RAM - headroom_bytes` allows
    ///     (a lent slot's expert is streamed during the prompt and copied back after it).
    ///   - the rest (the experts no slot holds) must fit that budget, or nothing is allocated and this returns false.
    ///
    /// CS-T, `budget_bytes` > 0 (`--resident-budget-gib`): only as many of those experts as fit `budget_bytes`, taken
    /// in `rank` order (the expert profile: the hottest after the GPU cache's), are copied; the rest stay on the
    /// mapped files (the SSD tier).  No lend region then (a lent slot's expert is read from the files).
    /// #467: `budget_bytes` = `kResidentWhatFits` is that path sized by the RAM alone (available minus the headroom
    /// and the #403 margin) - the soft --resident-experts mode's second try when the whole complement does not fit;
    /// false when not even one expert fits.  On Windows the mapped experts leave the working set before any reading.
    static constexpr uint64_t kResidentWhatFits = ~0ull;
    bool pin_cache_complement(
        const ExpertCache& cache, std::string& err, bool pin = true,
        const std::vector<std::pair<int32_t, int32_t>>& additional_gpu_pairs = {}, int64_t lend_from_slot = -1,
        uint64_t headroom_bytes = 8ull << 30, uint64_t budget_bytes = 0,
        const std::vector<std::pair<int32_t, int32_t>>* rank = nullptr);
    void close();

    bool mapped() const { return base_ != nullptr; }
    int64_t blobs() const { return blobs_; }
    uint64_t pinned_bytes() const { return complement_pinned_ ? complement_pin_limit_ : 0; }
    uint64_t resident_bytes() const { return complement_bytes_; }
    bool complement_pinned() const { return complement_pinned_; }
    bool complement_ready() const { return complement_ready_; }
    uint64_t locked_bytes() const { return complement_locked_; }
    /// Lend-region slots whose experts the compact copy holds (the last ones of the cache).
    int64_t resident_lent_slots() const { return complement_lent_slots_; }

    // ---- the resident RAM mode and the adaptive tier.  A swap puts `in` (held here) into a GPU slot and evicts
    // `out` (held only by that slot).  Before the slot is overwritten the caller copies it back into an exchange
    // buffer and calls `stage_exchange`: `out` is then read from that buffer, and `in` still from here (the CPU
    // computes both until the swap lands).  Once the slot copy has landed, `commit_exchanges` moves `out` into
    // `in`'s place, so the copy keeps holding exactly the experts the GPU does not - with no read of the file.
    /// Whether the compact copy holds `(layer, expert)`.
    bool has_resident(int64_t layer, int64_t expert) const;
    /// Host room for `n` evicted blobs (page-locked when possible).  Idempotent for the same or a smaller `n`.
    bool reserve_exchanges(int64_t n, std::string& err);
    int64_t exchange_capacity() const { return xstage_cap_; }
    uint8_t* exchange_buffer(int64_t q) const;
    /// Requires `has_resident(layer, in)`, `!has_resident(layer, out)` and `exchange_buffer(q)` holding out's blob.
    bool stage_exchange(int64_t layer, int64_t in, int64_t out, int64_t q);
    /// After the GPU copies of every staged swap have landed.  Returns how many exchanges were applied.
    int64_t commit_exchanges();
    int64_t exchanges() const { return exchanges_; }
    /// With the compact copy ready: blobs read from the mapped file since (what the plain mmap mode may read from
    /// the SSD).  0 in a steady resident mode; lend-region experts that did not fit the RAM count here.
    int64_t file_reads() const { return file_reads_.load(std::memory_order_relaxed); }
    /// CS-T per-tier counters: blobs served from the RAM copy, and the bytes read from the mapped files (the blobs
    /// `file_reads` counts, plus the prompt path's copies of them).
    int64_t ram_reads() const { return ram_reads_.load(std::memory_order_relaxed); }
    uint64_t file_read_bytes() const { return file_read_bytes_.load(std::memory_order_relaxed); }
    /// Of those, the bytes `blob` read (decode windows, adaptive swaps; the rest are the prompt path's copies), and
    /// the time spent reading the files, summed over threads.
    uint64_t file_blob_bytes() const { return file_blob_bytes_.load(std::memory_order_relaxed); }
    double file_ms() const { return (double) file_us_.load(std::memory_order_relaxed) / 1000.0; }
    /// Threads `prefetch` reads the GGUF with (STRATA_FETCH_THREADS, default 8).
    void set_fetch_threads(int n) { fetch_threads_ = n < 1 ? 1 : n; }
    /// #286 (Windows): read the experts straight from the drive (FILE_FLAG_NO_BUFFERING, overlapped) instead of
    /// through the mapped files - the GGUF in place, or a pack's experts.bin - when the file cache could not keep them
    /// beside `ram_bytes` (the RAM budget), cached now or not (their mapped pages would land in the working set); see
    /// experts_unbuffered.  The mapped reads' page faults are one small request each, and the pages they bring in
    /// take the RAM the budget was sized for.  `why` says what decided.
    /// #577: `ram_bytes` counts up to every expert, and only the experts outside it are what the cache must keep.
    bool set_unbuffered(uint64_t ram_bytes, std::string& why);
    /// #577: the same decision once the RAM copy is built (pin_cache_complement), from the RAM it really holds and the
    /// expert bytes outside it; switches either way (startup only, nothing reading).  Returns whether unbuffered.
    bool recheck_unbuffered(std::string& why);
    bool unbuffered() const { return !direct_.empty(); }
    /// Every expert's bytes (n_layers x n_expert blobs).
    uint64_t expert_bytes() const;
    /// #286, unbuffered: assembles the blobs of these pairs ahead of the `blob` calls that will ask for them (the
    /// GPU cache's fill from the profile) - one batch of reads instead of one blob at a time.  At most 64 pairs.
    void prefetch_pairs(const std::pair<int32_t, int32_t>* pairs, int64_t n);

    const uint8_t* blob(int64_t layer, int64_t expert) override;
    bool pinned(int64_t layer, int64_t expert) const override;
    const uint8_t* device_alias(int64_t layer, int64_t expert) const override;
    bool pcie_layer(int64_t layer) const override;
    bool transient(int64_t layer, int64_t expert) const override;
    bool copy_blob(int64_t layer, int64_t expert, uint8_t* dst) override;
    /// CS-T: advances the assembled blobs' age (see staged_blob).
    void begin_layer(int64_t layer, const int32_t* ids, int64_t k) override;
    /// CS-T: the GGUF in place assembles the missed experts on `fetch_threads_` threads.
    void prefetch(int64_t layer, const int64_t* experts, int64_t n) override;
    /// CS-T: the GGUF in place asks the OS for the predicted experts' pages (PrefetchVirtualMemory on Windows,
    /// madvise(WILLNEED) elsewhere), skipping the RAM copy's.
    void warm(int64_t layer, const int64_t* experts, int64_t n) override;
    /// Not when the reads are unbuffered: the warmed pages would be read through the file cache, a second time.
    bool warms() const override { return !role_ptr_.empty() && direct_.empty(); }
    /// Of the blobs the file tier read for the decode, how many had been warmed for their layer beforehand.
    int64_t warmed_hits() const { return warm_hits_.load(std::memory_order_relaxed); }
    int64_t warmed() const { return warm_count_.load(std::memory_order_relaxed); }

    /// Blobs touched, for the driver to report.  With `h = 0` this is `48 * k` per token and the number is only
    /// interesting once Phase 3 makes it not so.
    int64_t reads() const override { return reads_; }

private:
    const uint8_t* mapped_blob(int64_t layer, int64_t expert) const;
    /// The blob's bytes from the mapped file(s) - experts.bin, or the three GGUF role slices - into `dst`.
    bool copy_from_files(int64_t layer, int64_t expert, uint8_t* dst) const;
    bool open_gguf(std::string& err);
    const uint8_t* staged_blob(int64_t layer, int64_t expert);
    bool claim_stage(int64_t key, size_t& v, bool& fill);
    bool fill_stage(size_t v, int64_t layer, int64_t expert, uint8_t* dst);
    void publish_stage(size_t v, int64_t layer, bool ok, double us);
    struct Fill { size_t v; int64_t layer, e; uint8_t* dst; };
    /// The claimed buffers' blobs: one overlapped batch when unbuffered, else the fetch threads' mapped copies.
    void fill_many(const std::vector<Fill>& todo);
    /// #286: the blobs from the drive, unbuffered: every role's 4 KiB-aligned window is read at once (overlapped)
    /// into this thread's aligned buffer, then copied into place.  False when a read fails.
    bool read_direct(const Fill* fills, size_t n) const;
    bool open_direct(std::string& why);
    std::vector<std::string> paths_;          ///< the mapped files, as maps_
    std::vector<void*> direct_;               ///< #286: per file, an unbuffered overlapped handle (Windows)
    std::vector<int> role_file_;              ///< 3 x n_layers: index into maps_ / direct_
    /// blobs assembled in the stage buffers (`blob` hands those out): the GGUF in place, or any unbuffered source
    bool staged() const { return !role_ptr_.empty() || !direct_.empty(); }
    static constexpr uint64_t kNoComplement = detail::kNoCacheComplement;
    // ---- CS-T: the GGUF shards in place
    std::string gguf_;
    struct Map {
        const uint8_t* base = nullptr;
        uint64_t bytes = 0;
#if defined(_WIN32)
        void* file = nullptr;
        void* mapping = nullptr;
#else
        int fd = -1;
#endif
    };
    std::vector<Map> maps_;
    std::vector<const uint8_t*> role_ptr_;    ///< 3 x n_layers: gate / up / down of the layer's expert 0
    std::vector<uint64_t> role_bytes_;        ///< 3 x n_layers: bytes per expert of that role
    // the blobs assembled for `blob()`: a small pool of buffers, one per recent (layer, expert).  A buffer is
    // reused only once `kStageAge` layer changes have passed since its blob was last asked for, so a pointer holds
    // through the layer it was asked in and the next ones (the pool computes a layer's misses before the next).
    static constexpr uint64_t kStageAge = 3;
    std::mutex stage_mu_;
    std::vector<std::unique_ptr<uint8_t[]>> stage_buf_;
    std::vector<int64_t> stage_key_;
    std::vector<uint64_t> stage_epoch_, stage_used_;
    std::vector<char> stage_busy_;            ///< being filled (outside stage_mu_): never a victim
    std::condition_variable stage_cv_;
    int fetch_threads_ = 8;
    std::atomic<uint64_t> file_blob_bytes_{0}, file_us_{0};
    std::unique_ptr<std::atomic<uint32_t>[]> warm_stamp_;   ///< per (layer, expert): epoch_ + 1 when warmed
    std::atomic<int64_t> warm_hits_{0}, warm_count_{0};
    std::unordered_map<int64_t, size_t> stage_of_;
    uint64_t stage_blob_ = 0;
    uint64_t stage_seq_ = 0;
    uint64_t epoch_ = 0;
    int64_t last_layer_ = -1;
    bool stage_grew_ = false;
    std::atomic<int64_t> ram_reads_{0};
    std::atomic<uint64_t> file_read_bytes_{0};
    const uint8_t* base_ = nullptr;
    int64_t blobs_ = 0;
    int64_t n_layers_ = 0;
    int64_t n_expert_ = 0;
    uint64_t mapped_bytes_ = 0;
    std::vector<uint64_t> layer_offsets_, layer_blob_bytes_;
    void* complement_arena_ = nullptr;
    const uint8_t* complement_host_ = nullptr;
    const uint8_t* complement_device_ = nullptr;
    uint64_t complement_bytes_ = 0;
    std::vector<uint64_t> complement_offsets_;
    bool complement_pinned_ = false;
    bool complement_partial_ = false;         ///< CS-T: only the first complement_pin_limit_ bytes are registered
    uint64_t complement_pin_limit_ = 0;
    uint64_t complement_lock_off_ = 0;        ///< the working-set lock covers [lock_off, lock_off + locked)
    bool complement_ready_ = false;
    uint64_t complement_locked_ = 0;          ///< bytes held in the working set (pin refused)
    int64_t complement_lent_slots_ = 0;
    std::vector<const uint8_t*> override_;    ///< staged exchanges: an evicted expert read from its exchange buffer
    struct Exchange { size_t in, out; int64_t q; uint64_t bytes; };
    std::vector<Exchange> staged_;
    uint8_t* xstage_ = nullptr;               ///< exchange buffers, `xstage_cap_ x xstage_blob_`
    bool xstage_pinned_ = false;
    int64_t xstage_cap_ = 0;
    uint64_t xstage_blob_ = 0;
    int64_t exchanges_ = 0;
    std::atomic<int64_t> file_reads_{0};
    int64_t reads_ = 0;
#if defined(_WIN32)
    void* file_ = nullptr;
    void* mapping_ = nullptr;
#else
    int fd_ = -1;
#endif
};

// ================================ THE RESIDENT ARENA (R2.1) =========================//
// **THE MMAP ABOVE IS THE REVIEW'S FINDING C1 AND IT IS WORTH 2.2x.**  Read the comment on `FileExpertSource`
// about the page cache holding all 34 GB: that is true of the expert file ALONE, and it is not what this engine
// does.  The PLE/n-gram shard is a 26.8 GB file that the engine also maps, so 34 GB of experts plus 26.8 GB of
// n-gram is 60.8 GB of mapped, file-backed pages on a 63 GB machine - and file-backed pages are exactly the ones
// the OS drops from the standby list when it wants memory.  The expert stream then re-faults from disk.
//
// Measured, this round: the pool runs at **~19 GB/s in the engine against 42.8 GB/s in `bench/micro/cpu_s2.cpp`
// on the same machine, reading the same 34 GB**.  The micro does `std::fread` into a heap arena and reads
// ordinary (anonymous, resident) memory; the engine reads `MapViewOfFile`.  That is the whole difference, and it
// is why this class exists.
//
// It reads `experts.bin` into a `PinnedArena` once at startup, so the expert stream comes from anonymous memory
// the OS has no cheaper reason to evict.  `PinnedArena` also tries `cudaHostRegister`, which the GPU needs for
// Phase 3's cache fills and the CPU/PCIe miss split - but registration is best-effort and reported, not assumed.
class ArenaExpertSource : public ExpertSource {
public:
    ArenaExpertSource() = default;
    ~ArenaExpertSource() override;
    ArenaExpertSource(const ArenaExpertSource&) = delete;
    ArenaExpertSource& operator=(const ArenaExpertSource&) = delete;

    /// Allocates and loads `<pack_dir>/experts.bin`.  Prints nothing; the caller reports `note()` and the load
    /// rate, because those are the two numbers that say whether the arena is the one that was asked for.
    /// R5-prefetch: called with a layer's routing ids before the pool drains.  For every requested blob that
    /// the hot tier does not already hold, submit one whole-blob readahead (`posix_fadvise WILLNEED`) so the
    /// pages stream in at the drive's sequential rate while the GPU runs this layer's hits - instead of the
    /// pool faulting 4 KiB at a time from a cold drive inside the drain.
    void begin_layer(int64_t layer, const int32_t* ids, int64_t k) override;
    /// Blobs the prefetch has submitted since startup.
    int64_t prefetched() const { return prefetched_; }
    /// Small-RAM machines: back the native-pack arena with `<pack_dir>/experts-native.bin` (mmap, MAP_SHARED).
    /// The first run materializes the blobs from the GGUF shards into the file; later runs page in lazily
    /// and the kernel's page cache holds whatever fits.  No cudaHostRegister (the PCIe alias path is off).
    void set_file_backing(bool on) { file_backing_ = on; }
    /// **SMALL-RAM MACHINES: PIN THE HOT EXPERTS IN RAM.**  A file-backed arena cannot hold tens of GiB of
    /// experts on a 30 GiB box, and the docs above say exactly why the mapped form is the A/B arm: file-backed
    /// pages are the first thing the OS reclaims, so the expert stream keeps re-faulting from disk.  This gives
    /// the tier the machine CAN afford: an anonymous arena holding the `ranked` profile's hottest blobs,
    /// `mlock`ed so neither reclaim nor a prompt's full pass over the expert set can evict them.  Routing is
    /// heavily skewed (`--expert-cache-per-layer` measures 21.4% hits in 8 slots/layer, 70.4% in 64), so the
    /// top slice carries most of the traffic and the residual SSD reads are what is left.  Call after `open`.
    void pin_hot(const std::vector<std::pair<int32_t, int32_t>>& ranked, uint64_t bytes);
    /// Bytes actually pinned by `pin_hot`, and how many blobs that is.
    uint64_t hot_bytes() const { return hot_used_; }
    int64_t hot_blobs() const { return hot_count_; }
    /// R8: **MAKE THE TIER ADAPTIVE.**  The R5-R7 tier is a STATIC copy of the profile's top blobs, and a
    /// static tier tuned on one trace domain collapses when the conversation moves: measured on a fresh
    /// technical prompt after a tier built from the bench traces, only 33% of routed experts were resident
    /// and decode fell to 0.2 tok/s (1.5 GB/token of NVMe reads).  With this on, the tier is an LRU whose
    /// initial contents are the profile in rank order: a decode miss is admitted into the tier (the pread
    /// ring lands DIRECTLY in the slot, no extra copy), the least-recently-used blob is evicted to make
    /// room, and the static profile is just the starting LRU order.  R8b: prefill admits TOO (see
    /// `dc_stage_admit`) - the original "prefill must not admit" rule was wrong for agentic sessions, where
    /// the prompt's routed experts are precisely the working set every later turn and generation reuses.
    void set_dynamic_tier(bool on) { dynamic_tier_ = on; }
    bool dynamic_tier() const { return dynamic_tier_; }
    /// R8 diagnostics for the serve stats line.
    int64_t dc_admits() const { return dc_admits_; }
    int64_t dc_evicts() const { return dc_evicts_; }
    int64_t dc_fallbacks() const { return dc_fallbacks_; }
    /// R8b: **PREFILL ADMISSIONS.**  A prompt's routing is decided layer by layer as prefill runs, and the
    /// experts a prompt routes to are exactly the experts its generation will route to - so prefill's
    /// staging reads are admitted to the LRU.  A multi-turn session then prefills from RAM after its first
    /// turn and decode starts primed; measured, an unadmitted prompt sweeps ~20 GB of disk EVERY turn and
    /// decode pays the same misses again.  `dc_stage_admit` copies the landed staging bytes into a fresh or
    /// evicted LRU slot and commits it.  False when the tier cannot take it (no tier / no slot).
    bool dc_stage_admit(int64_t layer, int64_t expert, const uint8_t* bytes);
    int64_t dc_stage_admits() const { return dc_stage_admits_; }
    /// Whether the tier is doing anything: served / total `blob()` calls since startup.
    int64_t hot_hits() const { return hot_hits_; }
    int64_t hot_lookups() const { return hot_lookups_; }

    /// R22: **THE PRESSURE GOVERNOR (the elastic tier).**  `pin_hot`'s tier is `mlock`ed and
    /// `cudaHostRegister`ed - only reclaimable RAM frees itself; a hard-hosted session (24 GiB tier + KV pools
    /// + everything else on ~30 GiB) reaches zero headroom, thrashes the swap and dies inside CUDA
    /// (cublasCreate, 2026-10-04/05 twice).  `shed_pressure` returns WHOLE tier slices to the host: the
    /// cold-end slices' slots leave the tier (`hot_slot_` cleared under the tier lock - every consumer gate
    /// checks it, so the DMA/GPU path and the CPU pool stop using them), the slice's registration is undone
    /// (registration PINS the pages - it must be unregistered before they can be reclaimed) and the pages are
    /// dropped.  Decode on a shed slice pays a pread/miss instead of a DMA hit: slower but correct, and the
    /// tier's former free space is what keeps the engine (and the machine) alive.  Returns the bytes granted.
    uint64_t shed_pressure(uint64_t bytes_to_free);
    /// Bytes the governor has already shed (diagnostics / the note line).
    uint64_t hot_shed_bytes() const { return hot_shed_; }
    /// Slots the shed freed (refillable by the LRU admissions when pressure passes).
    int64_t shed_free_slots() const { return dc_free_.size(); }
    /// R22b: **THE GOVERNOR'S REGROW.**  Re-register + re-mlock the most-recently-refill-gravity shed slice
    /// (the arena's END slices refill first: the LIFO free list pops high slot ids), one per call, and only
    /// when the host has MemAvailable >= slice + 1024 MiB of its own (the engine reads /proc/meminfo itself).
    /// Returns the bytes regrown; 0 = refused (no headroom) or nothing to regrow.  The refilled slots' blobs
    /// are re-admitted by ordinary LRU traffic afterwards; a regrown slice does not resurrect content.
    uint64_t regrow_pressure(uint64_t bytes);

    /// R7: **WHERE THE EXPERT BYTES ACTUALLY CAME FROM.**  The old pair above counts only the lookups that
    /// reached the HOT-TIER CHECK, so a blob answered from the pread ring or straight out of the mapping was
    /// invisible - which is how "100% hot tier" coexisted with a disk-bound run.  These counters are inclusive
    /// over every blob the engine asked for, whichever path answered, and `disk_bytes` is the number that says
    /// whether the SSD is the wall.
    struct BlobStats {
        int64_t requests = 0;   ///< every blob() call that needed bytes for a routed expert
        int64_t hot = 0;        ///< answered from the pinned host tier (RAM, no I/O)
        int64_t ring = 0;       ///< answered from the pread ring (already read into RAM this layer)
        int64_t map = 0;        ///< fell through to the file mapping: a fault, page cache or disk
        int64_t disk = 0;       ///< whole-blob reads issued to the drive (ring jobs + staging preads)
        int64_t disk_bytes = 0; ///< bytes those reads moved
        // R7: the scheduling side of the same story.  `calls`/`entries` say how often begin_layer runs and
        // with how many ids; `hot_skips` says how many of those were already resident; `jobs` is what it
        // actually asked the drive for.  jobs >> (entries - hot_skips) means the ring is re-fetching.
        int64_t calls = 0, entries = 0, hot_skips = 0, already = 0;
        /// R7 prefetch study: of this run's disk misses, how many were ALSO a miss in the previous verify
        /// window.  That number is the entire case for prefetching: the per-layer read costs ~2.9 ms of exposed
        /// drive latency (measured, 2.08 MB at QD1 = 740 MB/s), so a prediction that covers even half of the
        /// next window's misses would remove half of a 140 ms per-window cost.
        int64_t win_misses = 0, win_repeat = 0, win_prefetched = 0, win_prefetch_used = 0;
    };
    BlobStats take_blob_stats();   ///< read and reset, so the caller reports per request
    /// R5-fetch: read the whole blob into `dst` - one sequential `pread` out of the file, or a memcpy when
    /// the hot tier holds it.  -1 when this source cannot (a resident arena has no file to read).
    int64_t read_blob(int64_t layer, int64_t expert, void* dst) override;
    int64_t read_blob_raw(int64_t layer, int64_t expert, void* dst) override;   ///< R8: stats-free, reader-safe
    void wait_layer() override;                          ///< R7: block until this layer's submitted reads land
    bool ring_pending(int64_t layer, int64_t expert) const override;   ///< R7: is this blob still in flight?
    void prefetch(int64_t layer, const int32_t* experts, int64_t n) override;   ///< R7: warm the page cache
    bool open(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, int threads, std::string& err,
              uint64_t max_pinned_bytes = 0, const std::string& shared_arena_file = {});
    /// Plan v0.3 P6: a native pack without experts.bin takes its experts from the model's GGUF: `native` is the
    /// --native shard, and native_experts.txt names the other shards beside it (per layer, or per role in v4).
    void set_gguf(const std::string& native) { gguf_ = native; }
    void close();

    bool mapped() const { return base_ != nullptr; }
    int64_t blobs() const { return blobs_; }
    const uint8_t* blob(int64_t layer, int64_t expert) override;
    int64_t reads() const { return reads_; }
    bool pinned(int64_t layer, int64_t expert) const override;
    const uint8_t* device_alias(int64_t layer, int64_t expert) const override;
    bool pcie_ready() const override { return file_map_ != nullptr ? hot_dev_ok_ : !dev_slice_.empty(); }
    void prefetch(int64_t layer, int64_t expert) override;
    uint64_t release(int64_t layer, int64_t expert) override;

    /// What backing was obtained and why, for the startup print.  "The engine adapts to the machine it is on" is
    /// only true if the engine says what it got.
    const std::string& note() const { return note_; }
    /// #633: set by `open` when the RAM available is less than the arena needs (a recommendation; it still loads)
    const std::string& ram_warning() const { return ram_warning_; }
    double load_gib_per_second() const { return gib_per_s_; }
    // Loader fix: the load, split.  `load_seconds()` is the wall clock of the load loop; the other two are
    // sums over the reader threads (see LoadStats), so on their own they say how much of that wall was spent
    // waiting for the disk and how much in memcpy + FNV-1a.
    double load_seconds() const { return load_seconds_; }
    double load_read_seconds() const { return load_read_s_; }
    double load_copy_seconds() const { return load_copy_s_; }

private:
    void* arena_ = nullptr;          ///< the PinnedArena, owned
    void* map_ = nullptr;            ///< STRATA_ARENA_MMAP: the arena file, mapped read-only (not the PinnedArena)
    uint64_t map_bytes_ = 0;
    std::vector<const uint8_t*> dev_slice_;   ///< device alias of each registered slice (or of the whole range)
    uint64_t slice_bytes_ = 0;
    const uint8_t* base_ = nullptr;
    int64_t blobs_ = 0;
    int64_t n_expert_ = 0;
    int64_t reads_ = 0;
    std::string note_;
    std::string ram_warning_;
    double gib_per_s_ = 0.0;
    double load_seconds_ = 0.0;
    double load_read_s_ = 0.0;
    double load_copy_s_ = 0.0;
    uint64_t pinned_bytes_ = 0;
    std::string gguf_;
    bool file_backing_ = false;
    void* file_map_ = nullptr;       ///< the mmap of experts-native.bin (file_backing_ path)
    uint64_t file_map_bytes_ = 0;
    int file_fd_ = -1;
    // R13: **O_DIRECT EXPERT READS.**  Every ring/staging/tier-fill read used to go through the page cache:
    // the kernel allocated ~530 cache pages per 2.18 MB blob, copied disk->cache->user, and the reader then
    // issued POSIX_FADV_DONTNEED to hand it all back (R8.2 measured ~39% of engine CPU inside kernel page
    // management on that cycle).  A second O_RDONLY|O_DIRECT fd serves the same bytes straight into the
    // destination buffer: no page-cache allocation, no DONTNEED, no double copy.  `file_fd_` stays for the
    // mmap fallthrough and the (page-cache) WILLNEED prefetchers.  STRATA_NO_ODIRECT=1 is the A/B arm.
    int direct_fd_ = -1;
    bool direct_ok_ = false;
    // R20: the alignment O_DIRECT actually enforces on THIS file, probed at open (512 on this NVMe/btrfs
    // pair, not the 4096 the code used to assume).  Blob offsets are 1024-aligned, so with a 512 requirement
    // every read lands directly in its destination and the 2.18 MB bounce memcpy disappears.
    uint64_t direct_align_ = 4096;
    uint8_t* hot_arena_ = nullptr;   ///< the anonymous pinned arena for the profile's hottest blobs
    uint64_t hot_cap_ = 0, hot_used_ = 0;
    int64_t hot_count_ = 0;
    int64_t hot_hits_ = 0, hot_lookups_ = 0;
    // ---- R8: the adaptive LRU tier.  The arena is carved into fixed slots of `dc_slot_bytes_` (the largest
    // blob, 4 KiB-rounded; blob sizes are uniform within ~1% so the waste is negligible), which makes
    // allocation, eviction and the LRU all O(1) array walks with no allocator and no fragmentation.  `blob()`
    // stays keyed on `hot_slot_` (blob index -> byte offset); the slot id is that offset divided by
    // `dc_slot_bytes_`, and the LRU is an intrusive list over slot ids.
    bool dynamic_tier_ = true;
    uint64_t dc_slot_bytes_ = 0;
    int64_t dc_slots_ = 0;
    std::vector<int32_t> dc_prev_, dc_next_;   ///< per slot: LRU neighbours (-1 = none)
    std::vector<int32_t> dc_idx_;              ///< per slot: blob index it holds, or -1 when free
    std::vector<uint32_t> dc_epoch_;           ///< per slot: window epoch of the last touch/admit
    std::vector<uint16_t> dc_count_;           ///< per slot: decayed hit count (frequency-aware eviction)
    std::vector<int32_t> dc_free_;             ///< free-slot stack
    int32_t dc_head_ = -1, dc_tail_ = -1;      ///< LRU: head = most recent
    std::vector<int64_t> dc_admit_list_;       ///< this layer's pending admissions (blob idx), committed in wait_layer
    int64_t dc_admits_ = 0, dc_evicts_ = 0, dc_fallbacks_ = 0;
    int64_t dc_stage_admits_ = 0;              ///< R8b: prefill admissions
    // ---- R22: the ELASTIC tier (the pressure governor).  `cudaHostRegister` pins pages (unswappable) and a
    // tier whose whole 24 GiB is registered+mlocked is what an over-subscribed box dies on (2026-10-04/05:
    // zram collapse, then cublasCreate).  Registration is in 2 GiB SLICES and `hot_reg_` tracks which slices
    // still have a live registration; `shed_pressure` (called by the server's governor thread) picks whole
    // slices from the arena's end (the coldest rank tail), unregisters + munlocks + drops their pages and
    // puts their slots on the free list for the LRU admissions to refill when headroom returns.
    std::vector<std::atomic<bool>> hot_reg_;    ///< per slice (~2 GiB = 12 slices per 24 GiB tier): the registration is live
    uint64_t hot_slice_slots_ = 0;              ///< whole slots per slice, computed in pin_hot
    uint64_t hot_shed_ = 0;                     ///< bytes handed back to the host so far
    /// R8b: guards the LRU structures when prefill's READER threads touch them (decode's mutators are all
    /// host-thread and never overlap prefill, but prefill touches run concurrently on the reader pool).
    ///
    /// R22 measured this lock as a possible prefill bottleneck and found it is NOT one - leave it alone.
    /// Every staging read does hold the mutex across a whole-blob (2.18 MB) memcpy, and `rd-wait` is 83% of
    /// MoE time, so narrowing the critical section looked like the obvious 2x.  It is not: a `shared_mutex`
    /// (`read_blob_raw` shared, `dc_stage_admit` unique) measured 2.2x WORSE, because the reader publishes its
    /// slot's `done` flag only AFTER `stage_admit`, which then waits for the exclusive side behind 16 readers
    /// holding shared across their copies - writer starvation on the very flag the consumer blocks on.  And
    /// moving just the memcpy out (touch kept under the lock, for the epoch stamp that prevents eviction) was
    /// indistinguishable from baseline at n=6 (median 192 vs 197 tok/s, both min~110/max~300 - see REPORT).
    std::mutex dc_mu_;
    void dc_init(int64_t slot_bytes);
    int32_t dc_alloc(int64_t idx);             ///< a free or evicted slot for blob `idx`, or -1
    void dc_touch(int32_t slot);               ///< move to LRU head, stamp the window epoch
    void dc_touch_locked(int32_t slot);
    int32_t dc_alloc_locked(int64_t idx);
    void dc_unlink(int32_t slot);
    void dc_link_head(int32_t slot);
    /// R7: inclusive source counters; see `BlobStats`.
    BlobStats blob_stats_{};
    bool hot_locked_ = false;
    // ---- R9b: THE GPU's PCIe EXPERT PATH READS THE TIER.  Upstream's 95 tok/s comes from the GPU computing
    // its share of every layer's experts (the default --pcie-frac 0.55) - but that path requires the blob's
    // host memory to be REGISTERED (cudaHostRegister), and the mmap arena variant never registered anything,
    // so `pinned()`/`device_alias()` were permanently false/null and decode ran 100% on the CPU pool
    // (~330 ms/window serial, the measured wall).  Registering the mlocked LRU tier turns both back on:
    // `fetch_dma` copies tier-resident blobs into VRAM staging at full PCIe rate beside the CPU's work, and
    // the grouped kernel computes them on the GPU.  Tier residents are exactly the right 97%+ to offload.
    bool hot_dev_ok_ = false;
    const uint8_t* hot_dev_base_ = nullptr;
    std::vector<int64_t> hot_slot_;  ///< per blob index: byte offset in hot_arena_, or -1 when not pinned
    int64_t prefetched_ = 0;         ///< R5-prefetch: whole-blob WILLNEED submissions since startup
    std::vector<uint8_t> pf_seen_;   ///< per-call dedupe scratch, sized n_expert_

    // ---- R5-fetch: the pread ring + reader pool -------------------------------------------------------------
    static constexpr int kRingSlots = 64;
    /// R7: **CHUNKED READS - BUILT, MEASURED, REJECTED.**  The hypothesis was that one 2.08 MB random read
    /// (2.8-2.9 ms at QD1) could be split into `kSplit` concurrent sub-reads to raise the queue depth, since
    /// the engine only has 1-2 misses per layer.  Measured on the IQ3_XXS pack, 20 GiB hot tier, `--spec 4
    /// --spec-min-p 0.8`: **kSplit 4 made it WORSE - 8.4 tok/s against 10.7 whole-blob, ring-wait 167 ms
    /// against 113.**  The micro-benchmark already said why: 266 KiB reads at QD8 move only 682 MB/s while
    /// 2.08 MiB reads at QD8 move 1000-1080 MB/s, so this drive's per-request overhead, not its queue depth,
    /// dominates at small sizes.  Left at 1 (whole blob) and kept as the A/B arm so the next agent does not
    /// re-derive it.
    // R10: runtime-tunable (STRATA_RSPLIT, default 1).  R7 measured kSplit 4 (512 KiB) WORSE (8.4 vs 10.7
    // tok/s) and left it at 1; kSplit 2 (two ~1.09 MiB halves, issued concurrently) was never tried and the
    // drive's size/latency curve (2 MiB QD8 ~1.0-1.1 GB/s, 266 KiB ~0.68) puts it at the sweet spot.
    static int kSplit() {
        static const int v = [] { const char* e = std::getenv("STRATA_RSPLIT"); return e ? std::max(1, std::atoi(e)) : 1; }();
        return v;
    }
    static constexpr uint64_t kChunk = 512 * 1024;
    static constexpr int kMaxChunks = 16;                  // (max_blob + kChunk - 1) / kChunk, bounded
    struct PfJob { uint8_t* dst; uint64_t off, len; };     ///< one sub-read of one blob
    /// R20: the decode read ring.  Slot buffers are allocated on the O_DIRECT alignment so `direct_read_span`
    /// can pread a missed blob STRAIGHT into its final home; with plain heap vectors every one of these reads
    /// took the bounce buffer plus a whole-blob memcpy.
    using RingBytes = std::vector<uint8_t, AlignedBuf<uint8_t>>;
    std::vector<RingBytes> ring_;            ///< per slot: one whole blob (largest layer's bytes)
    std::vector<int64_t> ring_of_;           ///< blob index -> ring slot for the CURRENT layer, else -1
    std::vector<PfJob> pf_jobs_;
    std::vector<std::thread> pf_workers_;
    std::atomic<uint32_t> pf_epoch_{0}, pf_done_{0}, pf_head_{0}, pf_njobs_{0}, pf_parked_{0};
    std::atomic<bool> pf_stop_{false};
    // R5b: parked readers BLOCK on this condvar instead of spinning - ten threads spinning PAUSE on a
    // 12-core part steal SMT issue slots from the twelve compute workers and measurably slow the drain.
    std::mutex pf_mu_;
    std::condition_variable pf_cv_;
    int64_t ring_layer_ = -1;
    bool pf_started_ = false;
    /// R7: the submitted batch is in flight until `wait_layer` has confirmed every job done.
    bool ring_waiting_ = false;
    int pf_jobs_active_ = 0;
    // ---- R7 prefetch study: is the previous verify window a good predictor of this one's misses?
    // `miss_epoch_[blob]` is the window number in which that blob was last a miss, so a miss whose epoch is
    // `window_epoch_ - 1` is a REPEAT.  `last_misses_` is the previous window's miss list, used both to score
    // the predictor and (when `prefetch_predict_`) to warm the page cache for the window that is starting.
    std::vector<uint32_t> miss_epoch_;
    std::vector<int32_t> last_misses_, cur_misses_;
    uint32_t window_epoch_ = 0;
    bool prefetch_predict_ = true;
    void pf_window_start();
    void pf_record_miss(int64_t idx);
    void pf_worker();              ///< R5-fetch: one of the four pread threads
    int64_t direct_read_blob(int64_t layer, int64_t expert, uint8_t* dst);  ///< R13: one O_DIRECT blob read
    bool direct_read_span(uint64_t off, uint64_t len, uint8_t* dst);        ///< R13: one O_DIRECT byte span
};

/// Plan v0.3 P6: checks native_experts.txt's GGUF spans against the files, before anything is read: each layer's
/// gate/up/down at its recorded (file, offset) must be that tensor (`blk.L.ffn_<role>_exps.weight`), of the
/// layout's type and dimensions, and inside the file.  `native` is the --native shard (see set_gguf).
bool check_experts_gguf(const std::string& native, const strata::kernels::cpu::ExpertLayout& lay, std::string& err);
/// Fills `dst` (lay.total bytes, the experts.bin layout) from the GGUF files, one role at a time.
/// `unbuffered`: each chunk read past the file cache (Windows).
LoadStats load_experts_gguf(const std::string& native, uint8_t* dst, const strata::kernels::cpu::ExpertLayout& lay,
                            int threads, bool unbuffered = false);

}  // namespace strata::core
