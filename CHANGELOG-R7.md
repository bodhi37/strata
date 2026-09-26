# R7 changelog — 2026-09-26 takeover

Everything below is on branch `r7-takeover`.  Each entry says what was measured, because the point of
this file is that the next agent should not have to re-derive any of it.

Baseline at takeover: **0.4-2.0 tok/s decode**, 1-2 tok/s prefill, and (on the pre-R5 engine) pure
`!!!!!!` output.  End of R7: **~10-11 tok/s decode** on IQ3_XXS with coherent output, ~5-8x prefill.

---

## Root cause (see REPORT.md §0.1 for the full argument)

1. The expert arena (49.8 / 57.9 / 65.4 GiB) cannot fit in 30 GiB, so every miss reaches the NVMe.
   Upstream was tuned on 64 GB, where it all stays in the page cache.
2. Pre-R5 the reads went through the file *mapping*, faulting 4 KiB at a time.
3. **R5's pread ring had a 6.4x read amplification**: its ring-full fallback re-issued
   `posix_fadvise(WILLNEED)` for hot-tier-resident experts.  96,231 blobs fetched for 15,007 real misses.
4. The hot-tier hit-rate statistic only counted lookups that reached the hot-tier *check*, so a
   disk-bound run printed "100.0% hot tier".  This is what made the earlier configs look fine.
5. The GPU expert hit path is numerically unverified (the engine prints a warning about it); it is the
   source of the `!!!!!!` garbage.

## Fixed

* `src/core/expert_source.cpp` — hot-tier guard in the ring-full fallback.  **2.4 → 6.1 tok/s.**
* `src/core/expert_source.cpp` — `pin_hot` now `pread`s whole blobs into the tier (was a 4 KiB-faulting
  memcpy out of the mapping), calls `posix_fadvise(DONTNEED)` per blob, and `madvise(MADV_HUGEPAGE)`s the
  tier.  A 20 GiB tier no longer leaves 20 GiB of page cache behind.
* `src/core/expert_source.cpp` — **two-pass dispatch**: `begin_layer` submits and returns, `wait_layer`
  blocks, and `expert_pool_dispatch_multi` computes the layer's *resident* experts while its reads are in
  flight.  Hides 42-58 ms/window of CPU work inside the drive latency.
* `src/core/expert_source.cpp` — honest `BlobStats` counters (requests / hot / ring / map / disk /
  disk_bytes / predictor quality), replacing the misleading hot-only pair.
* `src/core/expert_source.cpp` — per-window miss-predictor scoring (`miss_epoch_`).
* `src/core/expert_source.cpp` — the ring is invalidated when `layer == 0`, so a stale ring blob can no
  longer be served to a prefill suffix that happens to land on the last decode layer's index.
* `src/program/generate.cpp` — `--dump-counts P` (raw per-(layer, expert) routing histogram, STRC format),
  checkpointed after every request in `--serve` mode; per-request phase breakdown (verifier / dispatcher /
  pool / MTP) and expert-source report; the profile-fill byte-size bug (`profile[i]` vs `profile[skip+i]`);
  `verify_slot` now checks a pair the VRAM tier actually holds.
* `tools/make_profile_from_counts.py` — histogram → STRP profile, and it prints the **coverage curve**
  (how much traffic the top N experts carry), which is the number every tier size has to be chosen against.
* `run-engine-mlock.sh`, `build-engine.sh` — `cap_ipc_lock,cap_sys_nice` file capabilities so the hot
  tier's `mlock` actually succeeds; the build re-applies them because `cp` clears xattrs.
* `data/profile-r7.bin` — 19,414 ranked pairs derived from this model's routing on this box (was an
  inherited 8,000-pair profile that carries only a rank, not a coverage figure).
* `bench_all.sh` — the deliverable benchmark: coherence gate first, then per-context throughput, per quant.
* `eval_heretic.py` — `--max-tokens` so the coherence gate is affordable at 10 tok/s.

## Tried and rejected (all measured)

| idea | result |
| --- | --- |
| zram / compression of the arena ("2-4x capacity") | **dead.**  IQ payloads are incompressible: lz4 1.000x, zstd -1 1.008x, zstd -3 1.007x, zstd -19 1.011x, xz -6 1.005x |
| page-cache prefetch of the previous window's misses (`WILLNEED` at window start) | **dead.**  The predictor scores 15.3% (3,991 misses, 609 repeats); 85% of the prefetched I/O is wasted |
| chunked reads, `kSplit 4` (512 KiB sub-reads for queue depth) | **worse.**  8.4 vs 10.7 tok/s, ring-wait 167 vs 113 ms.  266 KiB reads at QD8 move 682 MB/s; 2.08 MiB reads at QD8 move 1000-1080 MB/s |
| balanced per-layer hot-set allocation | **slightly worse** than the global frequency ranking at every budget tested (8k/9868/12k/14k/16k blobs) |
| VRAM expert tier | **not viable here.**  3.35 GiB free after the native projections (2,975 MiB) + MTP (949 MiB) + KV; verify graphs then fail `instantiate: out of memory`.  ~1,218 slots = 4.9% of the arena, and the hit path is the one the engine flags as divergent |
| larger speculative windows | `kVerifyMaxT` is already 8.  `--spec 8 --spec-min-p 0.8` = 11.4 vs 10.5 tok/s.  The draft head's ~47% acceptance is the limit, not the window size |
| NVMe power management / link | PCIe 4.0 x4, device active.  The 2.8 ms QD1 read is the drive's own latency |
| hot tier beyond ~26 GiB | starts swapping (zram 2.3 → 4 GB) and does not help: ring-wait is set by the per-layer read latency, not the miss count |

## Why the target is not met (arithmetic, not assertion)

Per decode token: **480 expert blobs = 1.044 GB** (IQ3_XXS).  Drive: 0.74 GB/s at QD1, ~1.0 GB/s
saturated (cold random 2.08 MiB reads, fresh offsets).  Residency: ~26 GiB of the 49.8 GiB arena =
**97.2% of routed traffic**.  So only ~29 MB/token of *bandwidth* is needed from the drive — fine.

The wall is **latency**: the 1-2 misses per layer arrive over **48 strictly serial layers** (layer L's
router needs layer L-1's output, so nothing can be run ahead or prefetched), the drive sits at QD1-2, and
each read pays its full 2.8 ms.  48 x 2.8 ms = 134 ms per window; measured ring-wait 120-170 ms.
Amortized over the MTP window (1.97-2.57 tokens) that is 52-68 ms/token → 15-19 tok/s theoretical.

Escaping it needs either ≥99% coverage (34.8 GiB resident for IQ3_XXS; the box has ~26) or a predictor
for a miss set that is 15.3% correlated.  Both are closed by measurement.

**The one lever that would still move the number: more resident bytes.**  Each extra GiB buys ~+0.8%
coverage and, more usefully, takes whole layers off the read path — at 99.1% coverage only 18% of layers
need a read at all and the ring-wait falls to ~25 ms.  Practical routes: a second NVMe or more RAM, or
shrinking the engine's own host footprint (1.4 GiB dense + 644 MiB embedding) to widen the tier.

## Reproduce

```bash
cd ~/models/strata
# coverage curve for the tier you are about to size
./venv/bin/python tools/make_profile_from_counts.py data/routing-iq3xxs.bin data/profile-r7.bin --blob-bytes 2176000
# one quant, coherence first then throughput
QUANTS="IQ3_XXS" bash bench_all.sh
# the engine's own account of where the expert bytes came from
grep -E "experts:|ring:|prefetch:|phases/win|pool/win" logs/r7-IQ3_XXS.log
```
