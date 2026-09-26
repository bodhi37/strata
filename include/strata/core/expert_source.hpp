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
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::core {

/// Where one routed expert's bytes come from.
///
/// Phase 2 has NO cache (`phase-2-correct-engine.md`: hit rate `h = 0`), so the only implementation is a
/// file-backed reader.  The interface exists anyway because Phase 3 replaces exactly this object with the VRAM
/// cache, and because a test can supply an in-memory source without a 34 GB artifact.
class ExpertSource {
public:
    virtual ~ExpertSource() = default;

    /// The 1,382,400-byte blob for `(layer, expert)`, or nullptr if it cannot be produced.
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
    int64_t n_expert = strata::kernels::cpu::NE;

    /// Counters, for the driver to report rather than for control flow.
    int64_t layers = 0;
    int64_t experts = 0;
    int64_t missing = 0;

    // ================================ R4: THE VRAM TIER, MEASURED BEFORE IT IS USED ================================
    //
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

    // ================================ R4.2c: THE HITS GO TO THE GPU ================================
    //
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

    // ================================ R4.2d: DID THE GPU ACTUALLY START? ================================
    //
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
    /// Plan v0.3 P6: decayed routing counts per (layer, expert) during decode (sized by the caller; empty = off),
    /// which the driver uses to swap the most-routed missing experts into the VRAM tier between rounds.
    std::vector<float> usage;
    /// R5: cumulative routed counts per (layer, expert), never decayed - what `--dump-profile` ranks into a
    /// STRP profile built from THIS model's actual routing on THIS machine.
    std::vector<uint64_t> usage_total;
    int64_t multi_misses = 0;      ///< distinct (layer, expert) pairs the CPU computed in verify windows
    int64_t multi_entries = 0;     ///< routed (token, expert) entries the CPU served in verify windows
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

    /// Maps `<pack_dir>/experts.bin` and checks its size against `n_layers * n_expert * BLOB`.
    ///
    /// The size check is not a formality: a short file would fault at the END of a long sequence, and an
    /// over-long one means the pack is not the one the geometry came from.  Refuses with the two numbers.
    bool open(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, std::string& err);
    void close();

    bool mapped() const { return base_ != nullptr; }
    int64_t blobs() const { return blobs_; }

    const uint8_t* blob(int64_t layer, int64_t expert) override;

    /// Blobs touched, for the driver to report.  With `h = 0` this is `48 * k` per token and the number is only
    /// interesting once Phase 3 makes it not so.
    int64_t reads() const override { return reads_; }

private:
    const uint8_t* base_ = nullptr;
    int64_t blobs_ = 0;
    int64_t n_expert_ = 0;
    int64_t reads_ = 0;
#if defined(_WIN32)
    void* file_ = nullptr;
    void* mapping_ = nullptr;
#else
    int fd_ = -1;
#endif
};

// ================================ THE RESIDENT ARENA (R2.1) ================================
//
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
    bool open(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, int threads, std::string& err);
    /// Plan v0.3 P6: a native pack without experts.bin takes its experts from the model's shard 1.
    void set_gguf(const std::string& shard1) { gguf_ = shard1; }
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
    void close();

    bool mapped() const { return base_ != nullptr; }
    int64_t blobs() const { return blobs_; }
    const uint8_t* blob(int64_t layer, int64_t expert) override;
    int64_t reads() const { return reads_; }
    bool pinned(int64_t layer, int64_t expert) const override;
    const uint8_t* device_alias(int64_t layer, int64_t expert) const override;
    bool pcie_ready() const override { return file_map_ != nullptr ? hot_dev_ok_ : !dev_slice_.empty(); }

    /// What backing was obtained and why, for the startup print.  "The engine adapts to the machine it is on" is
    /// only true if the engine says what it got.
    const std::string& note() const { return note_; }
    double load_gib_per_second() const { return gib_per_s_; }

private:
    void* arena_ = nullptr;          ///< the PinnedArena, owned
    std::vector<const uint8_t*> dev_slice_;   ///< device alias of each registered slice (or of the whole range)
    uint64_t slice_bytes_ = 0;
    const uint8_t* base_ = nullptr;
    int64_t blobs_ = 0;
    int64_t n_expert_ = 0;
    int64_t reads_ = 0;
    std::string note_;
    double gib_per_s_ = 0.0;
    uint64_t pinned_bytes_ = 0;
    std::string gguf_;
    bool file_backing_ = false;
    void* file_map_ = nullptr;       ///< the mmap of experts-native.bin (file_backing_ path)
    uint64_t file_map_bytes_ = 0;
    int file_fd_ = -1;
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
    /// R8b: guards the LRU structures when prefill's READER threads touch them (decode's mutators are all
    /// host-thread and never overlap prefill, but prefill touches run concurrently on the reader pool).
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
    static constexpr int kSplit = 1;
    static constexpr uint64_t kChunk = 512 * 1024;
    static constexpr int kMaxChunks = 16;                  // (max_blob + kChunk - 1) / kChunk, bounded
    struct PfJob { uint8_t* dst; uint64_t off, len; };     ///< one sub-read of one blob
    std::vector<std::vector<uint8_t>> ring_;   ///< per slot: one whole blob (largest layer's bytes)
    std::vector<int64_t> ring_of_;             ///< blob index -> ring slot for the CURRENT layer, else -1
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
};

}  // namespace strata::core
