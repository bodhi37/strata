# Strata + Qwen3.8-Flash-Next (Heretic-2, abliterated) — build report

Status: **quant bake-off in progress (2026-09-25 evening).** The original IQ3_XXS build is downloading/validating;
an expanded same-day research pass found two strictly smarter runnable candidates (IQ3_M and IQ4_XS of the
same abliterated line) after an engine patch, and all three are being benchmarked back-to-back tonight
(`orchestrator.sh`). Measurement tables below are filled from `bench/results/*.json` as each pass completes.

## TL;DR

* **Strata resolved.** The given URL `github.com/Nikko1221/Strata` 404s because the account is
  `Niko1221` (one "k"). `https://github.com/Niko1221/Strata` is the real, working project.
* **Chosen build:** `spiritfather/Qwen3.8-Flash-Next-heretic-2-i1-GGUF`, quant **IQ3_XXS**
  (Heretic abliteration, 0/100 refusals, KL 0.0818) — the smartest variant that Strata's CPU
  expert kernels can actually execute.
* **One real deviation from a stock install:** the community GGUF pins the 28.8 GB n-gram (PLE)
  table to Q8_0 (54.4 GB); Strata's ngram kernel accepts only IQ4_NL, so it is requantized
  (F16→Q8_0→IQ4_NL) to 28.8 GB. A second, smaller deviation is a one-tensor `output.weight`
  sidecar, because this split keeps the architecture keys and the head in different shards.
* Everything lives under `~/models/strata/`; endpoint is OpenAI + Anthropic compatible.

---

## 0. R7 TAKEOVER (2026-09-26) — root cause, fixes, and honest numbers

An agent spent 2026-09-25/26 building the three-quant bake-off above and left the engine at
**0.4-2.0 tok/s decode**.  This section is the takeover's findings.  Read it before §9-§11: several of
the earlier conclusions below are corrected here.

### 0.1 Root cause of the ~100x gap (measured, not inferred)

1. **The expert arena is demand-paged from NVMe and this box cannot cache it.**  Upstream Strata was
   tuned on a 64 GB host where the whole canonical 34 GB expert set lives in the OS page cache, so
   `blob()`'s mmap faults resolve from DRAM at ~40 GB/s.  These packs are **49.8 / 57.9 / 65.4 GiB**
   (IQ3_XXS / IQ3_M / IQ4_XS) against **30 GiB RAM**, so every miss reaches the drive.
2. **Pre-R5, the read path faulted 4 KiB at a time.**  ~480 blobs per decode token x ~650 pages per
   blob is ~310,000 page faults per token against a DRAM-less QLC drive.
3. **R5's pread ring fixed the granularity but introduced a 6.4x read amplification.**  Its "ring full"
   fallback loop re-issued `posix_fadvise(WILLNEED)` for every deduped id *including hot-tier-resident
   ones*.  Measured: 2,880 `begin_layer` calls, 67,748 ids seen, 52,741 hot-skipped, **96,231 blobs
   fetched — against the 15,007 that were actually missed.**  176.8 GB of pointless readahead per
   150-token generation, and it was the whole of the decode time.
4. **The engine's "100% hot tier" statistic was misleading.**  It counted only the lookups that reached
   the hot-tier *check*, so blobs answered from the pread ring or straight out of the mapping were
   invisible.  A run that was disk-bound printed "100.0%".
5. **The GPU (VRAM) expert tier is numerically unverified** — the engine itself refuses to vouch for it —
   so the only fast path was unusable, and the one fast path that did run produced the `!!!!!!` garbage
   (token id 0, i.e. degenerate logits).
6. **The inherited profile left 24-48% of expert requests on the SSD** (`--hot-ram-gib 10-12` of a
   49.8 GiB arena).

### 0.2 What R7 changed, in order of measured impact

| change | effect |
| --- | --- |
| **Fixed the 6.4x read amplification** (hot-tier guard in the ring-full fallback, `expert_source.cpp`) | 2.4 → **6.1 tok/s** on IQ3_XXS; disk bytes 209 → 37.7 GB per 150-token generation |
| **Trace-derived routing profile** — new `--dump-counts`, `tools/make_profile_from_counts.py`, `data/profile-r7.bin` (19,414 ranked pairs from this model's real routing on this box) | +coverage at equal tier size; the old `data/expert-profile.bin` carries only a *rank* and cannot say what the top N cover |
| **Honest, inclusive expert telemetry** — `ArenaExpertSource::BlobStats` (requests / hot / ring / map / disk / disk_bytes / predictor quality) | makes "where did the bytes come from" answerable; the misleading counter is gone |
| **`pin_hot` reworked**: whole-blob `pread` straight into the tier (was a 4 KiB-faulting memcpy out of the mapping), `posix_fadvise(DONTNEED)` after each blob, `MADV_HUGEPAGE` on the tier | a 20 GiB tier no longer leaves 20 GiB of page cache behind, and the tier gets 2 MiB pages |
| **Two-pass expert dispatch** — `begin_layer` submits and returns; `wait_layer` blocks; `expert_pool_dispatch_multi` computes the *resident* experts in pass 0 while the layer's reads are in flight | hides the CPU drain (42-58 ms/window) inside the drive's latency |
| **`cap_ipc_lock` + `cap_sys_nice` file capabilities** on `engine/strata` (build script re-applies them, since `cp` clears xattrs) | `mlock` of the hot tier actually succeeds; it was silently "reclaimable" |
| **Fixed two real bugs found on the way**: the VRAM profile-fill copied `profile[i]`'s blob *size* for `profile[skip+i]`; `verify_slot` checked `profile[0]`, which the hot tier now owns | the VRAM tier can at least be started and verified |

New knobs: `--dump-counts P`, `--expert-profile` now accepts a 19,414-pair profile, `STRATA_NO_PREDICT=1`
(A/B arm for the prefetch predictor), `kSplit` (chunked reads, default 1 = whole blob).

### 0.3 Measured progression (IQ3_XXS pack, one 150-token generation, coherent output throughout)

| config | decode tok/s | disk bytes / run |
| --- | ---: | ---: |
| start of R7 (`--expert-cache -2`, no hot tier, pre-R5 read path) | ~1.5-2.0 | ~1.04 GB/token |
| R5 ring + 12 GiB hot tier, inherited profile | 2.4 | **209.4 GB** (6.4x amplified) |
| + amplification fix | 6.1 | 37.7 GB |
| + trace profile, 12 GiB | 6.7 | 209 → (see §0.2) |
| + 20 GiB hot tier + two-pass dispatch | 8.8 | 13.8 GB |
| + `--spec-min-p 0.8` (best) | **10.5-11.4** | 12-14 GB |

`bench/results/r7-<quant>.{json,md}` hold the per-context tables.

### 0.4 Tried and rejected (measured, so nobody burns a day on these)

* **zram / compression of the expert arena.**  The mission's "effectively 2-4x expert capacity" does not
  hold: the IQ payloads are already high-entropy.  Measured on 64 MB sampled from the arena:
  **lz4 1.000x, zstd -1 1.008x, zstd -3 1.007x, zstd -19 1.011x, xz -6 1.005x.**
* **Page-cache prefetch of the previous verify window's misses** (`POSIX_FADV_WILLNEED` at window start).
  The predictor scores **15.3%** — 3,991 misses, 609 repeats — so 85% of the prefetched I/O is wasted.
* **Chunked reads** (`kSplit 4`, 512 KiB sub-reads to raise queue depth): **8.4 tok/s against 10.7**
  whole-blob, ring-wait 167 ms against 113.  Consistent with the micro-benchmark: 266 KiB reads at QD8
  move 682 MB/s while 2.08 MiB reads at QD8 move 1000-1080 MB/s.  This drive's per-request overhead, not
  its queue depth, dominates at small sizes.
* **Balanced per-layer hot-set allocation** (same budget, equal per-layer rank cutoff): slightly *worse*
  than the global frequency ranking at every budget tested (8k/9868/12k/14k/16k blobs).
* **The VRAM expert tier on this box.**  Only **3.35 GiB is free** after the native projections
  (2,975 MiB), the MTP draft head (949 MiB) and the KV cache; `--expert-cache auto` → 1,218 slots, and the
  verify graphs then fail with `instantiate: out of memory`.  Even if it fit, ~1,200 blobs is 4.9% of the
  arena and the hit path is the one the engine itself flags as divergent.
* **Larger speculative windows.**  `kVerifyMaxT` is already **8** (the configs only ever asked for 4).
  `--spec 8 --spec-min-p 0.8` gives 11.4 tok/s against 10.5 for `--spec 4`: the MTP draft head's ~47%
  acceptance, not the window size, is the limit.
* **NVMe power management / link.**  PCIe 4.0 x4 (16 GT/s) and the device is active; the 2.8 ms QD1 read
  is the drive's own per-request latency, not a wake-up penalty.

### 0.5 Why 20-30 tok/s is not reached — the arithmetic

* Per decode token the engine needs **480 expert blobs = 1.044 GB** (IQ3_XXS; 1.28 GB for IQ4_XS).
* This drive delivers **0.74 GB/s at QD1** and **~1.0 GB/s saturated** (measured, cold random 2.08 MiB
  reads, fresh offsets each run).  Raw device sequential is 2.0 GB/s; the arena file is fragmented
  (81,505 extents for IQ4_XS) and delivers ~0.9 GB/s sequential.
* The engine can hold **~26 GiB of the 49.8 GiB arena** before the box starts swapping (30 GiB total, of
  which the engine's own dense weights, token embedding and workspaces take ~4 GiB).  From the measured
  coverage curve that is **97.2% of routed traffic**.
* So ~2.8% of requests — **~29 MB/token** — must come from the drive.  At 1.0 GB/s that is 29 ms/token of
  *bandwidth*, which is fine for 20-30 tok/s.
* **The binding constraint is latency, not bandwidth.**  The misses arrive 1-2 per layer, spread over
  **48 strictly serial layers** (layer L's router depends on layer L-1's output, so nothing can be
  prefetched or run ahead).  The drive therefore sits at QD1-2 and every read pays its full **2.8 ms**.
  48 x 2.8 ms = **134 ms per window**, against a measured ring-wait of 120-170 ms.
* Amortized over the MTP window (1.97-2.57 tokens produced) that is 52-68 ms/token → **15-19 tok/s
  theoretical**, ~10-11 tok/s as implemented.

To beat this you must either (a) hold ≥99% of routed traffic — **34.8 GiB resident for IQ3_XXS**, which
the box cannot fit — or (b) make the per-layer read disappear, which needs a predictor for a miss set
that is **15.3%** correlated window-to-window.  Both are closed by measurement, not by assertion.

The one lever that would move the number materially is **more resident bytes**.  Every extra GiB of hot
tier buys ~+0.8% coverage and, more importantly, removes whole layers from the read path: at 99.1%
coverage only 18% of layers need a read at all, and the ring-wait would fall from ~120 ms to ~25 ms.

---

## 0.7 R14 SESSION (2026-09-27/28) — upstream v0.1.8 merged, the wall measured from the inside, long context proven on q4_0 KV

### 0.7.1 Upstream sync (the big merge)

Merged `origin/main` (v0.1.8) into `r9-hillclimb` on branch **`r14-upstream`** (commit `44c8724`).
Conflict policy: **keep our R7-R13 engine work** (host hot tier, O_DIRECT ring, prefill staging readers,
two-pass dispatch, exact-commit verify windows, pinned-slot refill, R9/R10 pool park, R10 tile prefill,
R11 OOB fix) and **port upstream's features into it**:

| upstream feature | what we did |
| --- | --- |
| **GPU hit-path warning removed** (issue #23: teacher-forced parity 95-98% same top-1, PPL equal) | warning gone from startup; VRAM tier officially usable |
| **Q4_0 KV + FWHT-256** (`--kv q4_0`, PR #21) | merged; **ported into our tile-based prefill QSA path** (fwht256 on K/V/Q/attn, `kv_append_q4` with host+staging pointers) |
| **KV streaming** (`--kv-resident`, 0.1.5/0.1.6) | merged; staged-pool plumbing (`take_stage`, `pools_of`, `kv_stage_from_host`) wired into the tile prefill |
| **Conversation cache** (`--prompt-cache N`, checkpoints) | **replaces our sess_record/GEN KEEP client** (engine-side beats server-side); `RESUME <n>` / `REUSED <n>` protocol |
| **Prompt-lookup suffix drafter** (`--suffix-draft N` + DraftPolicy) | merged into the serve loop alongside our exact-commit logic |
| **Per-request sampling** (temperature/top_p/top_k/min_p/penalties) | merged (req_sp → ver.set_sampling) |
| **server.py rewrite** (web app, telemetry, disconnect-safe) | taken wholesale (same `--config/--port` CLI; srv.sh/guard-launch.sh unchanged) |

Kept our exact-commit window commit (`nk = e+2 or a+1`) over upstream's blanket `commit(a+1)`: it commits
exactly the emitted stream, and `consumed`/`live` bookkeeping is aligned to it; the end-of-request
pending-token fix-up commits the final proposal so the cache covers the whole emitted stream.

**Parity suite post-merge (all PASS):** `sampler_parity --selftest`, `kv_q4_parity --selftest`
(Q4_0 blocks bitwise equal to the host quantizer), `kv_stream_parity --selftest` (streamed vs resident
KV bitwise, fp16 AND q4_0, 89.7-89.9% block hits, 0 failures), `native_expert_parity` on IQ3_XXS/IQ4_NL
(cpu-gpu rel 1.44e-2 = activation-rounding noise).
**Open item:** IQ3_M layers 0-5 (Q5_0 down): layer 0 ok (1.84e-2), **layer 5 gpu rel 3.20e-2 FAIL** —
pre-existing (the Q5_0 kernel predates the merge; the test had never been run on those shards), CPU path
correct; matters only for VRAM-cached Q5_0 experts in the IQ3_M bake-off.

### 0.7.2 OOM incident and the guard

The 23:27 OOM killed the engine + my shell tree: **two concurrent engines** (an orphaned startup attempt
plus a new launch) — the exact failure mode §1 warns about. Second OOM at 01:10 was MARGINAL: the engine
at its normal 25.7 GiB anon footprint was already swapping 1.2 GiB when an 84.9k-token request landed on
top of a heavier desktop; `--prompt-cache 6` checkpoints (~118 MB each, host RAM) contributed.
**Fixes:** `guard-launch.sh` (singleton engine check + ≥24 GiB RAM check + persistence mode + 10 s settle
before every launch; `stop` subcommand), and long-context configs use `--hot-ram-gib 24.0` +
`--prompt-cache 2`. The stale-binary trap also bit once (`build/strata` linked but `engine/strata` not
copied — build-engine.sh does the copy+setcap; direct `cmake --build` does not).

### 0.7.3 Decode: every remaining knob measured (all neutral or floor-only)

Single-shot 6-topic 200-tok sweep (warm), one engine, one client:

| config | min | median | max | verdict |
| --- | ---: | ---: | ---: | --- |
| r13-base (800 slots, spec 4) | 4.32 | 7.72 | 8.63 | baseline post-merge, coherent |
| +1200 VRAM slots (2.43 GiB) | 4.48 | 8.10 | 8.95 | **neutral** (+400 blobs ≈ +0.7% coverage, inside noise; 1300 slots leaves 87 MiB VRAM = stall risk) |
| +dynamic LRU tier +prefill admit | — | — | — | neutral (multi-turn t2+ avg 12.4 vs static 13.1) |
| +spec 6 +suffix-draft 3 | **6.99** | 7.92 | 9.01 | **floor +2.6 tok/s** (worst-topic cold starts); median flat — MTP acceptance 42% is the ceiling, suffix drafter fired on 4/110 windows |
| +pcie-frac 1.0 | 7.03 | 7.61 | 9.78 | neutral (PCIe offload can't bypass the serial drive latency, confirming R7) |
| +fresh profile (profile-r14.bin from fresh counts) | 4.66 | 7.34 | 8.33 | neutral within run noise |

Multi-turn (3-turn conversations, full history resent — `scratch/r14_multiturn_bench.py`):
turn-2/3 decode climbs to **9.3-17.1 tok/s** as the working set warms (turn 1 is prefill-dominated).

### 0.7.4 The wall, measured from the inside (replaces §3's arithmetic)

From the per-request telemetry (R7 lines) on a 200-token request:

* 280.9 expert requests/token; hot tier serves **93.6%**; **21.9 disk misses/token** at ~3.5 ms QD1 ≈
  **77 ms/token of ringwait = 85% of decode time**. Compute+attention+draft ≈ 18 ms/token.
* Fresh-profile coverage curve (measured from 737k routed entries): **12k blobs (24.3 GiB) → 93.3%**,
  16k (32.4 GiB) → 98.2%, 18k (36.5 GiB) → 99.3%.
* VRAM budget (measured): native projections 2975 MiB + MTP draft 950 + head 675 + dense pool ~1416 +
  KV/graphs/context ~1900 ≈ **7.9 GiB non-expert** → only **~2.4-2.6 GiB (1100-1200 slots)** for the
  expert tier on a 12 GiB card.
* **Resident ceiling = 24.5 (RAM) + 2.6 (VRAM) ≈ 27.1 GiB → ~95% coverage → ~10-11 tok/s single-shot
  ceiling for IQ3_XXS on this box.** The ≥99.1% coverage gate (§3) needs 34.8 GiB resident —
  **physically unreachable at 30 GiB RAM + 12 GiB VRAM with 7.9 GiB of non-expert VRAM users.**
  §3's "3,500-4,096 VRAM slots" assumed ~4.3 GiB of VRAM users; the real number is 7.9.
* Remaining measured levers and their honest ROI: Q8-requantize the dense projections + dense pool
  (−2.2 GiB VRAM → +1000 slots → ~+3-4 tok/s; requires shard surgery); more host RAM (+8 GiB DIMMs →
  tier 24.5→32 GiB → 98.6% → ~25-30 tok/s); IQ2_XXS pack (arena 36 GiB → 81% resident → ~98.5% →
  ~30 tok/s, at a large quality cost against the mission's intelligence-first ladder).

### 0.7.5 Long context: the q4_0 KV payoff (`strata-r14-lc24-q4.json`)

* **84,631-token needle test** (three needles planted at 10%/50%/90% depth + a cross-constraint CHECK):
  **all three recalled + correct reasoning**, on `--kv q4_0`, decode coherent.
* **102k cross-turn resume (the session's biggest win):** re-sending the same 101,950-token conversation,
  the engine's conversation cache resumed **101,945 of 101,950 tokens in 479 ms** (`RESUME 101945`) —
  a follow-up agentic turn on a 102k conversation pays 0.5 s of prompt processing instead of 9.4 minutes.
  Decode at 102k context: **5.3-5.8 tok/s**, draft acceptance **76%** (329 of 432) on the repeated content.
  A 17.3k-token suffix read after a 84.6k resume ran at **~1,440 tok/s** (warm tier + windows).
* q4_0 KV freed **~750 MiB of VRAM** (1062 MiB free at startup vs 318 with int8 + dynamic tier).
  With int8 KV the same request **OOM'd the verify graph at decode position 84,922** — q4_0 is what
  makes 84k+ decode possible on this card.
* Prefill cold at 84.6k-102k: **155-180 tok/s** (TTFT ~8-9.5 min). Drive-bound, and the telemetry shows
  why: 809k streamed blobs across the long-context session (~2.8 blob-reads per prompt token) — the
  non-tier ~12.5k unique blobs re-stream across all 11 chunks. The dynamic LRU capture helps only
  marginally while the profile fill occupies the tier: for prefill-heavy long-context work, a trimmed
  profile (~6k blobs) + wider LRU share is the untested config that could ~2x prefill.
* Workload split, measured: admission ON helps the long-conversation pattern (warm resumes, suffix at
  1.4k tok/s) and hurts the mixed short-topic pattern (LRU holds the long conversation's blobs; the
  next topic switch pays: medicine 2.98 tok/s right after the 102k runs). Static profile tier is the
  right choice for mixed short topics; admission for sustained long-context conversations.

### 0.7.6 Next session starts here

1. 80k/127k prefill with admission engaged at tier 24.0 (`strata-r14-lc24-q4.json` is the config) —
   read `streamed`/`lru:` telemetry; if the capture works, prefill ~2x.
2. Q8-requant of dense projections + dense pool (tools/make_*.py path) — the last real decode lever
   (+1000 VRAM slots).
3. IQ3_M bake-off caveat: fix/characterize the Q5_0 layer-5 gpu rel 3.2e-2 first.
4. Agentic stress: reasoning_effort=xhigh, giant budgets (untested this session).

---

## 0.8 R16-R19 SESSION (2026-09-28) — the prefill fix (4-8x), the env bug that hid every A/B, and the real decode budget

Session configs: `strata-r16-v1800-q4.json`, `strata-r17-tiers.json` (best), `strata-r18b-nodirect.json`,
`strata-r19-best.json` = r17 with `env` re-derived after the server fix. All measured with one client,
port 8123, `scratch/r15_probe.py`, `scratch/r11_decode_bench.py`, `scratch/prefillbench.py`, `/tmp/pf_disk2.py`
(diskstats-wrapped prefill).

### 0.8.1 THE BIG WIN: prefill was 4-5x under-performing because the chunk is auto-halved

`generate.cpp:2114` halves `--prefill` until the prefill buffers fit in the **expert cache's lendable
slots** (`k + 128 <= xcache.slots()`). With the 800-slot cache that shipped, **`--prefill 16384` was
silently served as 2048-token chunks** — and every chunk re-reads the ~12.5k-blob non-resident set
(the arena sweep), so a 6.4k prompt paid **4 chunks x 27 GB** instead of one.

| prompt | chunk served | prefill tok/s | read bytes / prompt-token | config |
| ---: | ---: | ---: | ---: | --- |
| 6,420 | 2048 | **57** | 21.4 MB (9.8 blobs) | r13-base (chunk 8192 offered) |
| 7,567 | 8192 | **255** | 5.8 MB (2.7 blobs) | r16 (cache 1200) |
| 15,029 | 16384 | **450** | 1.53 MB (0.70 blobs) | r17 (cache 1500) |
| 29,953 | 16384 | **419** | ~1.5 MB | r17 |
| 77,513 | 16384 | **336-379** | 0.35 blobs → 27.4k blobs total | r17 |

**Fix: the VRAM expert cache must have enough slots to lend the prompt path its buffers.** 1500 slots
(3.04 GiB) + `--kv q4_0` (-750 MB) makes `chunk 16384` real. `--prefill 16384` + `--expert-cache 1500` is
the new floor; a 2000-slot cache would let 32768 fit. **This is the single highest-value knob found in
this session and it was invisible: the only sign is one stderr line ("prompt chunk 16384 -> 8192 tokens
so its buffers fit in the expert cache").**

End-to-end effect, same `scratch/r11_coherence.py` suite, same box, coherent output throughout:

| case | r13-base (chunk 2048) | r19 (chunk 16384) | speedup |
| --- | ---: | ---: | ---: |
| LONG-PREFILL (6,425-token prompt) | 123.4 s | **56.9 s** | 2.2x |
| LONG-TASK (6,419-token prompt + generation) | 188.2 s | **69.7 s** | 2.7x |
| SHORT (64-token prompt) | 19.8 s | 43.3 s | (cold-start variance; the first request pays the tier's cold sweep) |

### 0.8.2 THE ENV BUG — every `STRATA_*` knob in every config was silently dropped

`serve/server.py:child_env()` built the engine environment from `os.environ` only; the config's `"env"`
block was never merged. So `STRATA_STATIC_TIER`, `STRATA_PREFILL_ADMIT`, `STRATA_NO_ODIRECT`,
`STRATA_PREFILL_TILE`, `STRATA_RSPLIT`, `STRATA_PARK_SPIN_US` … never reached the engine — r13-base read
"static" only because the tmux server happened to carry it. **Fixed** (the merge is now in `child_env`).
Two consequences: (a) every env-based A/B in §0.6/§0.7 is suspect, (b) `STRATA_STATIC_TIER=0` means
STATIC (the code tests presence, not value) — to get the dynamic LRU tier, **omit** the variable.

### 0.8.3 The drive ground truth (measured with `scratch/qdbench` on the real arena)

| read | QD1 | QD2 | QD8 | QD16 | QD32 | seq |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.18 MB O_DIRECT, iconq3xxs/experts-native.bin | 816 | 969 | 969 | 968 | 963 | 1289 (4 threads, 8 MB) |
| raw block device /dev/nvme0n1 | 886 | — | 722-1063 | — | 935 | — |
| 128 KB | — | — | 586 (QD8) | — | 356 (QD32) | — |
| 4 KB | — | — | — | — | **74 (18.5k IOPS)** | — |

**The drive is a flat ~1.0-1.3 GB/s device at every queue depth** (Kingston SNV2S1000G, DRAM-less
4-channel TC2200). PCIe link is Gen4 x4 (16 GT/s) — verified, ASPM off. `max_sectors_kb` 128→2048: +10%.
Scheduler kyber vs none: no difference (422 vs 420 MB/s on a prefill). dmesg clean, 36.9 °C. The mission's
"4+ GB/s" premise does not hold on this SSD; **~1.0 GB/s is the ceiling for the 2.18 MB random reads the
engine issues**, and it is reached (prefill rd-wait runs at 1.12-1.35 GB/s effective).

### 0.8.4 The decode budget, measured per window (`r15_probe` + `phases/win`)

r17, 200-token agentic generation, 120.2 ms/window, **1.52-1.57 tokens/window** (spec 4):

```
ver[wait 15.5  pool 91.3  host 9.8  commit 0.7]  dispatch[plan 2.3 actq 0.2 run 13.5 ringwait ~74]  mtp 1.2
pool/win: drain 19.2 (gu 11.7 + down 7.4)   <- the CPU pool's real kernel time, NOT the bottleneck
```

| per token | ms | what it is |
| --- | ---: | --- |
| ringwait | **47** | the missed-blob reads: ~20 blobs x 2.18 MB at ~0.93 GB/s |
| ver wait | 9.9 | GPU (attention/dense graphs) |
| ver host | 6.2 | per-layer host staging in the verify loop |
| pool pass-0 | 8.6 | resident experts on the CPU, overlapped with the reads |
| other | 2.8 | mtp, commit, plan, actq |
| **total** | **74.5** | = 13.4 tok/s (measured 12.0-13.4) |

So: **decode is miss-BYTE-bound, not latency-bound and not compute-bound.** `ringwait = miss_bytes /
~0.93 GB/s`, and the drive is already at its ceiling while it waits. The pool kernels are 19 ms/window of
a 120 ms window; the GPU is idle most of the window.

**The ceiling when the working set IS resident: 28.36 tok/s** (cooking topic, run immediately after a probe
on the same topic — 200 tokens in 7.1 s). That is the pipeline's non-drive floor: ~35 ms/token.

### 0.8.5 The coverage curve, re-measured, and what a bigger tier would buy

From the full 6.03M-entry routing dump (and confirmed on the R7-era dump): **12k blobs (24.3 GiB) →
95.56%, 14k (28.4 GiB) → 97.81%, 16k (32.4 GiB) → 99.12%, 18k (36.5 GiB) → 99.75%.**

* Per-window misses at 95.3% RAM coverage: ~20 blobs/token = 43 MB/token. At 98%: ~8 = 17 MB. At 99.1%:
  ~4 = 8.7 MB. **Every +2,000 resident blobs ≈ halves the miss bytes.**
* **Side-buffer "miss cache" (top-K non-resident experts per layer, prefetched): only 6-27% of misses are
  captured by 0.4-1.6 GiB** — strictly worse than spending the same bytes on the main tier. Rejected.
* The resident ceiling on this box is **~13,000 blobs** = 11,826 host (24 GiB) + 1,500 VRAM. Reachable
  growth found: +0 (see below).

### 0.8.6 Why the tier cannot grow (measured, not assumed)

* **Host RAM is at the wall.** Engine anon = 25.17 GB (24.00 GiB tier, mlocked) + zram 1.98 GB physical
  (6.8 GB of process pages, 3.4x compressed) + ~1 GB apps + 1.4 GB cache = 30.9 of 31.1 GB. `used 30,
  available 0.28`. A 26 GiB tier would OOM (the previous session's incident). The mission's "reclaim the
  dense pool (1416 MiB) and the token embedding (644 MiB)" does not apply to this build: both are
  **cudaMalloc'd VRAM** (the weight arena + the native dense set), and the token embedding failed to pin
  and lives in a file mapping, not anonymous RAM.
* **VRAM is at the wall.** 12,282 MiB total: native projections 2975 + weight arena 1416 + MTP draft 891 +
  draft head 105 + native head 675 + q4_0 KV @131k 906 + verify 67 + expert cache 1500 slots 3040 +
  graphs/workspaces ≈ 11.5 GiB used, **413 MiB free**. `--expert-cache 1800` fails (the head upload OOMs);
  1500 is the largest that boots with q4_0 KV at 131k context.
* **Smaller blobs are the only remaining byte lever, and they cost model quality** (see 0.8.7).

### 0.8.7 Tried and rejected in this session (each with its measurement)

| lever | result |
| --- | --- |
| `--kv q4_0` + `--expert-cache 1200 -> 1500` | **kept**: frees 750 MB and +300 blobs (~+1% coverage) |
| `--spec 8` (vs 4) | 1.85 vs 1.52 tokens/window but a worse floor (min 3.97 vs 6.03); **spec 4 kept** |
| `--prefill 16384` + cache 1500 | **kept** (the 0.8.1 fix) |
| buffered reads (`STRATA_NO_ODIRECT=1`) | prefill 232-385 tok/s vs 277-284 O_DIRECT — **no win**; r19 keeps O_DIRECT |
| `STRATA_PREFILL_ADMIT=1` | **hurts**: 1.82-2.09 MB/prompt-token vs 1.53 without (the admits evict each other, R10's finding stands) |
| I/O scheduler kyber vs none | 422 vs 420 MB/s — no difference |
| `max_sectors_kb` 128 -> 2048 | +10% on the drive micro-bench (816 -> 886 MB/s QD1); kept |
| bigger prefill chunk without lending (2000+ slots) | needs >1800 slots, which OOMs at 131k context |
| VRAM `--expert-cache-per-layer` | R7 measured per-layer allocation *worse* than the global ranking at every budget (8k-16k) |
| requant to shrink the arena | **blocked**: the down projection's row width is 640, so only 32/64-block types fit (Q4_0/IQ4_NL/Q2_0) — Q2_0 is a 2-bit down (large quality cost, against the mission's ladder), and btrfs CoW needs ~46 GB of free space for in-place compaction, of which there is 31 GB |

### 0.8.8 Where the remaining 2x is

1. **The prefill staging pipeline reads at ~350-680 MB/s while the drive does 969.** `rd-wait` dominates
   (`moe 174 s: rd-wait 131.6 s` on the 77.5k prompt). The reader pool is 16 threads with a 48-slot
   staging ring and `rd_q_.pop_back()` (LIFO — the consumer's next blob is served last); the dequant ring
   is DQ=2. Candidates: FIFO the reader queue, raise DQ/readers, or `io_uring`. **A 2x prefill (419 ->
   850) is sitting here**, and it is the mission's weakest gate.
2. **Decode's non-drive 28 ms/token** (ver wait 9.9 + ver host 6.2 + pool 8.6 + 2.8): overlap the verify
   staging and/or move resident-expert compute to the idle GPU (`--pcie-frac` applies the staging path to
   misses today, not to residents). At 28.36 tok/s the pipeline is *proven* to run at 35 ms/token.
3. Anything that adds resident blobs.

**Correctness gate:** every config above ran `r11_decode_bench.py` (6 topics) with all outputs OK/coherent;
no truncation or garbage was observed in any run.

---

### 0.6.1 Environment bugs found and fixed (each one blocked real work)

| bug | symptom | fix |
| --- | --- | --- |
| **GPU persistence mode off** | every engine start dies with `native embedding: cannot pin 644 MiB` — actually `cudaErrorDevicesUnavailable`; the ~1 s of CPU init races the driver tearing GPU state down | `sudo nvidia-smi -pm 1` (reboot-volatile; re-apply after reboot) |
| **Concurrent clients on one session** | outputs mixing topics across requests, 0-token responses, HTTP 400s — the kv-cache agent's harness and the bench shared one engine session; interleaved renders contaminated each other | never benchmark a shared server; one client at a time, private port |
| **`pkill -f` self-match** | launcher shells SIGTERM'd themselves mid-restart | srv.sh's `[e]` bracket pattern, or kill by PID |
| **Port 8111 contention** | engine killed silently mid-bench (harness restarts "its" port) | bench on 8123+ |

### 0.6.2 Engine changes this session (committed on `r9-hillclimb`)

* **R13a: O_DIRECT expert reads** (`expert_source.cpp/.hpp`, commit `4eca284`). Ring / prefill-staging /
  tier-fill reads go through an `O_RDONLY|O_DIRECT` twin fd with a thread-local 4 KiB-aligned bounce buffer
  (blob offsets are 256 B-aligned, ring dsts are heap vectors). Kills the per-read page-cache
  alloc/copy/`DONTNEED` cycle. Buffered pread+DONTNEED remains as automatic fallback; `STRATA_NO_ODIRECT=1`
  is the A/B arm. **Measured: NEUTRAL on decode** (30.4 s vs 31.1 s for the same 150-token request; identical
  output text at temperature 0 — a free correctness check of the direct path).
* `NativeEmbed::load` now reports the cudaErrorString (this is what exposed the persistence-mode bug).

### 0.6.3 Honest baseline (IQ3_XXS, r13-base = r12-static config + O_DIRECT, single client, 200-token gens)

`scratch/r11_decode_bench.py`, six topics, temperature 0, all **OK** (coherent; single client):

| run | min | median | max |
| --- | ---: | ---: | ---: |
| run1 (cold-ish) | 6.31 | 8.00 | 9.18 |
| run2 (warm) | 4.79 | 7.25 | 9.88 |

Warm is NOT faster: the tier is static and every topic switch is a full cache MISS that re-reads that
domain's tail from the SSD. 300-token gens reach ~14.1 tok/s (the working set warms within a request).

### 0.6.4 Where the wall actually is (measured this session, replaces guesswork)

* Per ~200-token request the engine pulls **5.3-5.4k blobs / 11.7 GB** from the arena (8.3% of routed ids
  miss the 24.5 GiB tier). Prefill **saturates the drive** (peak 845 MB/s, 16 parallel staging readers).
* **Decode cannot** — the misses arrive spread across the 48 strictly serial layers, so the drive sits at
  QD1-2: 35-190 MB/s during decode, ringwait ~230 ms per verify window of ~2.2 tokens. 2.18 MB / ~3.5 ms =
  0.63 GB/s ≈ the observed 0.54-0.56 GB/s effective rate. **This is a latency wall, not a bandwidth wall:
  parallel readers, O_DIRECT, and kSplit cannot touch it.** (O_DIRECT measured neutral, confirming.)
* Consequences: decode improves only via (a) fewer misses (more resident GiB — zram is DEAD for experts,
  the payloads measured 1.008x incompressible in R7; host footprint reclaim is the remaining RAM source),
  (b) more tokens per verify window (MTP spec tuning — but larger windows fetch MORE distinct experts per
  window, so the amortization is not free), (c) offloading misses to the GPU PCIe path (`--pcie-frac`,
  default 0.55, untested higher), or (d) merging neighboring blob reads per layer.
* Prefill's ceiling on big prompts is the ~53% of the arena a chunk sweeps; coverage ~97% would make
  prefill compute-bound (R10.5 measured 694 tok/s warm).

### 0.6.5 Staged and NOT yet measured (next session starts here)

1. `strata-r13-admit.json` — dynamic LRU tier + `STRATA_PREFILL_ADMIT=1`: prefill's sweep pays for its
   reads anyway, so admitting them lets the decode after a prefill run warm (multi-turn/agentic reuse).
   Count-0 admits + frequency-aware eviction protect the profile blobs. **Bench was interrupted — run it.**
2. `strata-r13-spec6.json` (admit + `--spec 6`) and `strata-r13-spin.json` (+ `STRATA_PARK_SPIN_US=250`).
3. `--pcie-frac` sweep (0.8/1.0): GPU computes more of the misses beside the CPU pool.
4. Host RAM reclaim (dense pool 1416 MiB, embedding 644 MiB — VRAM already holds both) → bigger tier.
5. Rebuild `data/profile-r10.bin` from the freshest counts (24576 sampled now).
6. IQ3_M / IQ4_XS bake-off at the 20-30 gate (configs exist from R7; re-derive tier for the bigger arenas).
7. The GPU expert hit path still carries the engine's "NOT CORRECT" warning (divergence vs cache-off at
   2.97% hits: first diff token 40). Coherence gates pass with it on, but a KL eval cache-on/off would
   settle whether the VRAM tier can grow.



The task's reference URL returned 404 on 2026-09-25 and the `nikko1221` account had 0 public repos.
Case variants were tested; **`Niko1221`** (capital N, single k) owns the repo and it clones and
builds:

```
git clone https://github.com/Niko1221/Strata.git      # commit 1ee8b66, branch main
```

This is a **different project** from the published paper *"Strata: Hierarchical Context Caching for
Long Context Language Model Serving"* (arXiv 2508.18572, OSDI '26) whose implementation lives in
SGLang as `--enable-hierarchical-cache` / HiCache. This repo is a self-contained CUDA/C++ engine
that runs Qwen3.8-Flash-Next across **GPU VRAM + system RAM + SSD** tiers with its own MTP
speculative decoding and an OpenAI/Anthropic server (`serve/server.py`).

`Niko1221/Strata` is the one to use: it is purpose-built for exactly this model on exactly this
class of hardware (12–24 GB NVIDIA + 64 GB RAM), and its README reports 45–95 tok/s and 262K
context on a 12 GB card. The SGLang HiCache "Strata" is a related-but-separate KV-cache tiering
idea; it is not a drop-in for a `qwen4exp` model on 12 GB VRAM (SGLang would need the weights in
HBM/RAM at a precision this host cannot hold) and is recorded here as the stated fallback only.

## 2. Hardware (re-verified 2026-09-25)

| | |
| --- | --- |
| GPU | RTX 4070 SUPER, 12 282 MiB VRAM, driver 615.71.09, CUDA UMD 13.4 |
| CPU | Ryzen 9 9900X, 12C/24T, AVX-512 + AVX512-VNNI + AVX512-BF16 |
| RAM | 30 GiB (≈25 GiB available) + 31 GiB swap |
| Disk | 745 GB NVMe, ≈302 GB free |
| Host name / OS | `one`, CachyOS, kernel 7.2.6 |
| LAN | `wlan0` on 5 GHz ch40 (signal −68 dBm, single stream) to `x64` |

## 3. Model facts and the frontier target

Official model card (`Qwen/Qwen3.8-Flash-Next`, released 2026-08-26):

* 125 B parameters with **6 B active**, plus a **51 B n-gram embedding** table and a **4 B MTP** head.
* 48 layers: `12 × (3 × (Gated DeltaNet → MoE) → 1 × (Qwen Sparse Attention → MoE))`.
  QSA: 24 Q / 2 KV heads, head dim 256, RoPE dim 64. MoE: 512 experts, 10 routed + 1 shared, inter dim 640.
* Context 262 144 native, YaRN-extensible to 1 000 000.

Reported head-to-head versus **Claude Opus 4.6 (Max)** (Qwen's own table, 256K context):

| Benchmark | Qwen3.8-Flash-Next | Claude Opus 4.6 (Max) |
| --- | ---: | ---: |
| SWE-bench Pro | **62.5** | 53.4 |
| SWE-bench Multilingual | **81.0** | 77.5 |
| LiveCodeBench v6 | **91.9** | 88.8 |
| GPQA Diamond | **91.7** | 91.3 |
| CoWorkBench | **73.9** | 68.2 |
| JobBench | **55.7** | 36.6 |
| HLE | 35.9 | **40.0** |

So the *model itself* is already at or above Opus-4.6 level on coding/agentic work and slightly
below it on HLE-style deep knowledge. The goal of this build is therefore to preserve as much of
that as the hardware allows, not to reach it from a weaker base.

## 4. Quant landscape — chosen with evidence

Two independent constraints decide the field:

1. **Engine compatibility.** Strata's CPU expert pool accepts gate/up ∈
   {IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S} and down ∈ {Q2_0, IQ4_NL}; dense mmvq accepts
   Q2_0/Q4_0/Q5_0/Q8_0/Q3_K/Q4_K/Q5_K/Q6_K/IQ4_NL/IQ4_XS/BF16/F16/F32. Any build whose routed
   experts use IQ4_XS/Q4_K/Q5_*/Q8_0 experts cannot use the CPU pool.
2. **Measured reconstruction error.** The `spiritfather` repo publishes PPL and KLD against its own
   Q8_0 reference for every quant (wikitext-2, ctx 512), which is the most direct "intelligence
   retained" evidence available for an abliterated build of this exact model.

| Candidate build | Experts (gu/down) | Strata CPU pool | KLD vs own Q8_0 | Verdict |
| --- | --- | --- | ---: | --- |
| spiritfather heretic-2 **IQ3_XXS** | IQ3_XXS / IQ4_NL | ✅ | **0.1096** | **chosen** |
| spiritfather heretic-2 IQ3_M | IQ3_S / IQ4_NL+Q5_0(6 layers) | ✅ after §4b patch | 0.0857 | **bake-off #2** |
| spiritfather heretic-2 IQ2_M | IQ2_S / IQ4_NL | ✅ | 0.1787 | lower quality |
| spiritfather heretic-2 IQ4_XS | IQ4_XS / IQ4_NL | ✅ after §4b patch | 0.0472 | **bake-off #3, smartest** |
| cygnal Uncensored IQ4XS-NGQ4 | IQ4_XS / Q5_1 | ❌ (Q5_1 down not in {20,42,6}) | n/a | unrunnable without a further kernel |
| huihui UD-Q4_K_XL | Q4_K / Q5_1+Q8_0 | ❌ | n/a | unrunnable |
| 0bserverx RVN IQ3_S | IQ3_S / IQ4_NL | ✅ (compact PLE) | n/a | viable, but: 2/100 strict refusals, no total-KL number, text-only (no MTP head, no vision) — see below |
| ISTA-DASLab GSQ-RCO IQ3_XXS | IQ3_XXS / IQ4_NL | ✅ | 99.4 % task avg of BF16 | best *stock* quant, but **not** abliterated |

Other abliterated builds examined and set aside (2026-09-25):

* **orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF** — ships IQ3_M and IQ4_XS too (2–3-file splits, imatrix),
  and a separate MTP-draft.gguf, but: the repo is gated (auth + terms), the card's numbers are self-reported
  (refusals "≈0–3.3 %", capability "±2 points" on MMLU-Pro/GSM8K-class checks) with **no KLD and no
  per-quant quality table**, and the GGUF line drops the MTP head entirely (llama.cpp qwen4exp had no MTP
  at their build). Heretic-2 dominates it on evidence: 0/100 refusals, total KL 0.0818, per-quant KLD table,
  MTP head shipped per-quant, vision included.
* **0bserverx/RVN (V6)** — the most scientifically documented abliteration of this model (embedding-preserving
  R2 projection on 145/146 writers + a rank-16 repair LoRA, matched Q6_K downstream evals: IFEval +0.92 pp,
  GSM8K −0.83 pp, BBH-5obj +4.0 pp), but its release is **text/main-only** (no vision, no NextN/MTP), its
  strict-refusal count is 2/100 (worse than heretic-2's 0/100), and its published KL (0.073) is *incremental
  LoRA-vs-projected*, not total-vs-base, so it is not comparable to heretic-2's 0.0818 total. Without MTP it
  also gives up the biggest single throughput lever on this hardware. Kept as the documented alternative.

The official GSQ-RCO table is the strongest quality evidence for this family: IQ3_XXS (3.00 bpw)
reaches a **92.57/93.12 task average (99.4 %)** and matches BF16 exactly on AIME25, 0.51 behind on
GPQA-Diamond and 1.14 on LiveCodeBench v6. The author-recommended reasoning mode is **xhigh**.

Decision (updated 2026-09-25): run a **three-way bake-off** of the same abliterated line — IQ3_XXS, IQ3_M,
IQ4_XS — because the engine constraint that eliminated the two smarter quants turned out to be **patchable,
not physical** (see §4b). IQ4_XS (KLD 0.0472) is the leading candidate for the final endpoint on the priority
ladder (intelligence > throughput); IQ3_XXS remains the fallback if IQ4_XS's larger expert set pages too hard
on 30 GB RAM. All three share shards 2+3 byte-identically (verified via HF LFS oids), so the switch costs only
shards 1/4/5 (~63–70 GB per quant).

### 4b. Engine patch: Q5_0 down + IQ4_XS gate/up experts (2026-09-25)

Direct GGUF-header probes (ranged HTTP reads, no full download) of `spiritfather/...-IQ3_M-00004/5` show:

* `ffn_gate_exps`/`ffn_up_exps` = **IQ3_S** (type 21) — already in Strata's CUDA/CPU expert kernels.
* `ffn_down_exps` = **IQ4_NL** for 42 layers but **Q5_0 (type 6) for layers 0–5** — the one gap.
* `IQ4_XS` gate/up = type 23 — in `iq_row_bytes`/dequant but missing from the grouped expert switch.

`src/kernels/cuda/iq_kernels.cu` gained two vec_dot kernels transcribed from the pinned llama.cpp
`vecdotq.cuh` (`vec_dot_q5_0_q8_1`, `vec_dot_iq4_xs_q8_1`), `Fmt<6>`/`Fmt<23>` entries, `iq_row_bytes` case 6,
and the two `native_expert_grouped` switch cases. Rebuilt engine passes the repo's own three-way parity
(`build/native_expert_parity`) on the existing IQ3_XXS/IQ4_NL layers (cpu/gpu rel ~1.2–1.6e-2 vs float =
activation-rounding noise, same as before the patch); the new formats get the same parity check the moment
their shards land. The 6 Q5_0 layers were **not** requantized — the kernel path keeps the published
KLD 0.0857 exactly.

## 5. Abliteration

Pre-made, no self-abliteration needed (OBLITERATUS would require an FLA/causal-conv1d correct
forward pass and ~15 % device headroom; unnecessary here):

* Base model of this build: `trohrbaugh/Qwen3.8-Flash-Next-heretic-2`, produced with a custom fork
  of **Heretic v1.3.0+custom (per-layer direction)**.
* Author-reported: **0/100 refusals (base 99/100), KL divergence 0.0818** — the lowest-KL Heretic of
  Qwen3.8-Flash-Next at time of writing. Heretic co-minimizes refusals and KL by construction; its
  reference table shows the same pattern (e.g. 3/100 refusals at KL 0.16 vs 0.45–1.04 for
  alternatives).
* Abliteration for this architecture edits residual-writer tensors and does **not** touch the MoE
  router, the 51 B n-gram table, or the vision tower (confirmed in the OrcaRouter notes for the
  sibling Uncensored build: 149 residual writers, router/PLE/vision untouched).

Verification performed here: the auto-run spot-check (`eval_heretic.py`) reports capability on
arithmetic/logic/coding prompts and a refusal-rate probe; results in §8.

## 6. Stack decision

llama.cpp (SM80-89) — `qwen4exp` landed in PR #27742, MTP in #27836. It is a fine reference and is
what the quant community targets, but on 12 GB + 30 GB RAM it cannot hold the resident expert set
and re-reads the whole conversation every turn. vLLM/SGLang need the weights resident in
HBM/RAM at a precision this host cannot hold.

**Strata** is chosen because it is the only stack that treats the three tiers as first-class:
GPU runs the always-used weights and an adaptive expert cache; RAM holds the experts and the CPU
computes the ones the GPU misses *concurrently*; the 28.8 GB n-gram table is read sparsely from
SSD; MTP drafts 3 tokens/pass. It also exposes `/v1/chat/completions` and `/v1/messages`.

Build: `build-engine.sh` — CMake/Ninja, `-DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
-DSTRATA_NATIVE_EXPERTS=ON`, CUDA 13.3, resulting in `engine/strata`.

## 7. Pipeline deviations for this build

Stock `setup.py` assumes the official two-shard layout (shard 1 = trunk + head, shard 2 = PLE at
IQ4_NL). The community heretic split differs, so three adjustments are needed:

1. **PLE requantization.** `tools/make_ple_iq4nl.py` reads `per_layer_token_embd.weight`
   [160 × 320 001 536] from shard 3 (Q8_0) and rewrites it as a one-tensor IQ4_NL GGUF
   (90 B/row → 28.8 GB). `tools/ple_requant.c` cross-checks the quantizer against ggml's own
   `quantize_iq4_nl`.
2. **Native head sidecar.** `NativeHead::load` opens a single file and requires both the
   `qwen4exp` architecture guard and `output.weight`. Here the guard lives in shard 1 (0 tensors)
   and the head in shard 2 — `tools/make_native_head.py` writes `ple/head-native.gguf`
   (arch KV + the 675 MB Q8_0 head). `--native` points at shard 1 so the engine's `model_shards()`
   expands to all five, and `--native-head-gguf` points at the sidecar.
3. **Multi-shard discovery.** `tools/iq_pack.py` and the engine both discover
   `-0000N-of-0000M` siblings; the pack is built with `--gguf <shard 1>` and the engine launched
   with `--native <shard 1>`.

Runtime arguments (mirroring `setup.py`):

```
engine/strata --serve \
  --pack packs/heretic-iq3xxs --native model/…-00001-of-00005.gguf \
  --native-head-gguf ple/head-native.gguf --ple-gguf ple/ple-iq4nl.gguf \
  --expert-profile data/expert-profile.bin --expert-cache auto \
  --prefill 2048 --spec 4 --spec-min-p 0.5 --mtp mtp/rt \
  --max-context 131072 --kv int8
```

## 8. Pi harness

Added (not replacing anything) to `~/.pi/agent/models.json`:

```
strata-flash-next-heretic-local  →  http://127.0.0.1:8101/v1  (openai-completions)
  model: qwen3.8-flash-next-heretic-2-iq3xxs-strata
  reasoning: true, contextWindow: 131072, maxTokens: 131072
  compat: supportsDeveloperRole=false, supportsReasoningEffort=true
  thinkingLevelMap: xhigh/max → high
```

`~/.pi/agent/settings.json` gets giant budgets: `thinkingBudgets` =
`{minimal:2048, low:16384, medium:65536, high:262144}`. Strata's OpenAI surface takes a
`reasoning_effort` level as an instruction (the model has no hard thinking cap); the model card's
own recommendation is 262 144 reasoning tokens / 131 072 answer tokens, so `maxTokens` is set to the
full 131 072 and no artificial cap is imposed. Port 8101 is free and distinct from every existing
provider (8080, 8091–8103, 1919, 1933/1934).

## 9. Measured results

**R7 (2026-09-26) — see §0 for the config and the root cause.**  Generated by `bench_all.sh` into
`bench/results/r7-<quant>.{json,md}`; every number is behind the coherence gate in the same file.

| Context | prompt tokens | TTFT (s) | prefill tok/s | decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| ~1K | | | | |
| ~4K | | | | |
| ~32K | | | | |
| ~128K | | | | |

**Decode rate is a function of generation length, and this must be read before quoting a number.**
The expert working set grows as the generation diverges, so the hot tier's coverage falls:

| generation length (IQ3_XXS, 24 GiB hot tier) | decode tok/s |
| ---: | ---: |
| 150 tokens (sweep benchmark) | 10.5-11.4 |
| 320 tokens (coherence eval, diverse questions) | 3.7 |

The 10-11 tok/s figure is the honest steady-state number for short-to-medium turns; a long, wide-ranging
generation walks more of the 49.8 GiB tail and pays for it.  Any benchmark that quotes a single decode
number without its completion length is worthless.

Also: device load time, peak RAM/VRAM/swap, engine log (`strata-heretic.log`), capability and
refusal spot-check, and the thinking-mode 4K measurement.

## 10. Gap analysis (final numbers to be inserted)

**Corrected by R7 — see §0.5.**  The binding constraint on this box is not bandwidth but the NVMe's
per-request latency: 1-2 missed experts per layer, over 48 strictly serial layers, at 2.8 ms a read.
That is 134 ms per verify window, which is 52-68 ms/token after MTP amortisation → 15-19 tok/s
theoretical, ~10-11 as implemented.  Escaping it needs ≥99% routing coverage (34.8 GiB resident for
IQ3_XXS; the box has ~26 GiB after the engine's own weights) or a predictor for a miss set that is
15.3% correlated window-to-window.  Both are closed by measurement, not assertion.

Intelligence is bounded by IQ3_XXS (≈99.4 % of BF16 on task average) × abliteration (KL 0.0818), i.e.
very close to the model's own Opus-4.6-class standing.

## 11. Highest-value next optimizations

R7 ordering, by measured leverage:

* **More resident expert bytes** — the only lever that moves decode materially, because it takes whole
  layers off the read path.  At 99.1% coverage only 18% of layers need a read at all and the ring-wait
  falls from ~120 ms to ~25 ms.  Routes: more RAM, a second NVMe, or shrinking the engine's own host
  footprint (1.4 GiB dense + 644 MiB token embedding).
* **Parallelise the prefill staging reads** — prefill stages one blob at a time on the host thread (QD1),
  the same latency trap decode was in.  A 4K prompt measured 40 tok/s with 7.7 s of CPU expert drain and
  the rest waiting on the drive.
* MTP acceptance (`--spec-min-p` 0.8 was the best of {0, 0.2, 0.5, 0.8}); the draft head's ~47% acceptance
  is the ceiling, not the window size.
* A KVarN-style or int8 KV on the 12 full-attention layers to push context toward 512K.
* ~~`--expert-profile` re-derived from this model's routing~~ — **done in R7** (`data/profile-r7.bin`).

## 12. Reproduce

```
# one-time shared artifacts (PLE requant, head sidecar, pack, MTP) + IQ3_XXS endpoint + bench
bash ~/models/strata/heretic_finish.sh
# per-quant passes for the two smarter quants (stop the previous server, launch, bench, eval)
bash ~/models/strata/finish_quant.sh IQ3_M  8102
bash ~/models/strata/finish_quant.sh IQ4_XS 8103
# or all of it, in order, hands-off
bash ~/models/strata/orchestrator.sh
curl http://127.0.0.1:8101/v1/models
python3 ~/models/strata/bench_endpoint.py --port 8101 --contexts 1024,4096,32768,131072
# expert-format parity for the patched kernels (Q5_0 down, IQ4_XS gate/up)
LD_LIBRARY_PATH=$HOME/deps/cuda-13.3/opt/cuda/lib64 ~/models/strata/build/native_expert_parity <shard4.gguf> 0 5
```

---

## 0.9 R22 SESSION (2026-09-29) — prefill is stall-bound; the tier mutex is NOT the lever; and the measurement method was the real bug

Takeover state: branch `r14-upstream` with the R20/R21 work **uncommitted in the working tree** (FIFO
prefill reader queue, O_DIRECT 512-byte alignment probe + no-copy read, tunable
`STRATA_PREFILL_READERS`, `--gpu-share`, cublas/embed retry loops), and a live `strata-r21b.json` run.
Everything below is on that tree; the code is left in the committed-behaviour state (see 0.9.3).

### 0.9.1 The live run was a regression — `--gpu-share` moves RESIDENT experts over PCIe

`strata-r21b.json` = r19 + `--gpu-share 0.5` + `STRATA_RSPLIT=2` + `--pool-workers 12`. Its decode was
**5.3-9.4 tok/s** against r19's measured median 7.75, with `ringwait` exploding to **1074-8001 ms/window**
(vs ~74 ms in r17's breakdown). The mechanism: `--gpu-share` routes a fraction of *all* distinct experts -
RAM-tier hits included, not just VRAM-cache misses - to the GPU over PCIe. A resident expert costs nothing
to read from DRAM, but sending it to the GPU means a 2.18 MB DMA per blob; at ~20 experts/layer x 48 layers
that is ~2 GB/token across PCIe, which cannot beat the CPU pool reading the same bytes from RAM. **Moving
resident-expert compute to the idle GPU is the wrong trade on this box: the data movement dominates.**
(The separate `--pcie-frac` path - misses only - is unaffected and was already measured neutral in R7.)

### 0.9.2 Single-sample A/B is invalid on this box — the run-to-run variance is 3x

Every prefill number in this project (including R7's and R19's tables) was a single sample. Measured with
`scratch/pf_multi.py` (new: K UNIQUE prompts, since the conversation cache makes repeats meaningless), same
binary, same config, one engine, n=6 on a ~10.8k-token prompt:

| run | min | median | max |
| --- | ---: | ---: | ---: |
| baseline (r19-best), n=6 | 104 | **176** | 264 |
| `STRATA_PREFILL_READERS=48`, n=6 | 87 | **181** | 264 |

The spread within ONE config is larger than any difference between configs. **Any prefill A/B in this repo
has to be n>=5 samples; a single 385 tok/s sample and a single 173 tok/s sample mean nothing.** (For
calibration, the same 13,758-token prompt measured 385, 312 and 173 tok/s across three launches of
identical-or-near-identical configs.) The variance tracks `kswapd0` reclaim and co-tenant disk I/O
(`slskd`, the desktop) rather than anything the engine does.

### 0.9.3 The tier mutex is NOT the prefill bottleneck (two attempts, both rejected)

The engine's own telemetry says prefill is read-bound: on a 15.7k-token prefill `rd-wait` was **90.2 s of
109.2 s** of MoE time, and the request pulled **18.52 GB at 0.52 GB/s**. The obvious suspect was
`ArenaExpertSource::dc_mu_`: *every* staging read holds it across a whole-blob (2.18 MB) `memcpy` - out of
the tier on a hit, into it on an admission - so 16 reader threads queue on one lock and the disk reads
behind them are not issued. Two shapes were built and measured:

* **`shared_mutex`** (`read_blob_raw` shared, `dc_stage_admit` unique): **2.2x WORSE**, 173 vs 385 tok/s on
  the same prompt. The reader publishes its slot's `done` flag only *after* `stage_admit`, which then needs
  the exclusive side while 16 readers hold shared across their copies - writer starvation on the exact flag
  the consumer waits on. Rejected.
* **Plain mutex, memcpy moved outside the critical section** (touch kept under the lock, because the
  `window_epoch_` stamp it sets is what stops `dc_alloc_locked` evicting the slot mid-copy): n=6 median
  **192 vs 176** baseline - indistinguishable. Rejected.

Conclusion: the lock is real but not load-bearing here. The code is left as the R20/R21 tree (memcpy under
the lock), with the measurements recorded on `dc_mu_` so the next agent does not re-derive this.

### 0.9.4 What prefill actually is: stall/depth-bound, and not by reader count either

`scratch/r16_bottleneck.py` during a prefill:

```
NVMe read rate during run : 376 MB/s   (525 MB/s with 48 readers)
GPU util  min/med/p90/max : 0 / 8 / 79 / 100 %
VERDICT: GPU idle AND drive idle -> prefill is STALL/DEPTH-bound
```

The drive is NOT the wall: `dd iflag=direct` moves **985 MB/s sustained over 20 GB** (twice, stable), and
4 KiB random does 272 MB/s. So the engine leaves ~2x of drive rate unused during prefill, yet raising the
reader count 3x changed nothing - the stall is upstream of the readers. Prefill's consume loop is
single-threaded and interleaves, per expert, `cudaStreamWaitEvent` -> H2D -> `cudaEventRecord` -> dequant ->
2 cuBLAS GEMMs, with a full `cudaStreamSynchronize` per tile in the router pass; with only `STAGE=48`
staging slots the read queue drains whenever that thread stalls on a slot (`used-wait` 8.5 s) or on compute.
Raising `STAGE` is the untested lever but it is coupled to the prefill borrow region (a bigger buffer set
makes the driver halve `--prefill`), so it was not attempted blind.

### 0.9.5 Where that leaves the levers (honest)

* **Decode is at its structural wall.** ~0.4 misses/layer over 48 strictly serial layers = QD1 per layer at
  ~2.8 ms, ~20 blobs/token = 43 MB/token; the drive delivers ~0.93 GB/s for that shape, so ~46 ms/token is
  the floor (~21 tok/s) and r19 measures 7.75 median / 9.62 max. It improves only with more resident bytes.
* **Prefill's 2x is in the host consume loop / pipeline depth**, not the drive, not the lock, not the
  reader count. The next concrete experiment is a deeper staging ring decoupled from the borrow region (a
  separate read-buffer pool), or removing the per-tile `cudaStreamSynchronize` in the router pass.
* Do not trust a single prefill number from any earlier section: use `scratch/pf_multi.py`, n>=5.

## 0.10 R23 WRAP-UP (2026-09-29) — the GSQ-RCO Coder was evaluated and declined; IQ3_XXS stands as the endpoint

The Coder quant (`Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF`, released as "IQ1_M") was downloaded, inspected at the
byte level, and its upstream support was ported into this fork. It was then **declined**: it does not buy the speed
it was bought for, and the user is not using it for its one strength. Everything below is measured off the file on
disk, not off the model card. **The live endpoint is unchanged IQ3_XXS** (branch `r14-upstream` @ `dd609f9`,
`engine/strata`, config `strata-r23-wrapup.json` = `strata-r19-best.json` with only the log/counts paths renamed).

### 0.10.1 "IQ1_M" is a misnomer, and that is the whole story

The file contains **no IQ1_M tensor at all**. Shard 1's type histogram is
`Q6_K x129, BF16 x484, F32 x292, IQ4_XS x45, Q4_K x47, IQ4_NL x86, IQ3_XXS x34, Q5_K x35, Q2_0 x9, IQ2_S x40, IQ3_S x20, F16, Q8_0`
(`tools/probe_gguf.py`, 1223 tensors). The name is the release's label for **1.89 bits per _original_ parameter** -
i.e. the pruning (256 of 512 experts per layer) is folded into the bitrate. The kept experts are stored with ordinary
encodings: gate/up in IQ3_XXS/IQ2_S/IQ3_S/IQ4_XS, down in IQ4_NL/Q2_0.

Consequence, and this is the number that kills the premise: an expert blob's size is set by the expert's **shape**
(`ffn_*_exps` is 2560x640 per expert, 144 such tensors), not by the headline quant.

| | expert arena | bytes/expert blob | MiB per token (480 blobs) |
| --- | ---: | ---: | ---: |
| heretic IQ3_XXS (live) | 49.81 GiB | 2.075 MiB | **996** |
| Coder "IQ1_M" shard 1 | 23.42 GiB | 1.952 MiB | **937** |

A token walks `top_k 10 x 48 layers = 480` blobs in **both** models. The Coder reads **6% fewer bytes per token**.
The halving is in the number of *experts*, which a token never touches - it only shrinks the arena that has to fit
somewhere. Measured DRAM read bandwidth on this box (`build/bwbench`, 12 threads over scattered 2 MiB pages) is
**51.8 GB/s**, so the fully-RAM-resident decode ceiling is 51.8/(996/1024) = **53 tok/s** for IQ3_XXS and
**56.6 tok/s** for the Coder. "300-700 tok/s decode, fully resident" was never reachable on this architecture at
this top-k: the bytes per token are the wall, and they barely moved.

Upstream's own numbers agree, and they are the honest verdict (RTX 5070 12 GB, Ryzen 5 7600, 64 GB -
`bench/results/2026-09-28-coder/` vs `2026-09-28-speed-0114/`):

| 4K prompt | Coder | IQ3_XXS | | 32K | Coder | IQ3_XXS |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| prefill tok/s | **1,152** | 770 | | prefill tok/s | **1,298** | 1,108 |
| decode tok/s | 50.6 | **62.1** | | decode tok/s | **53.3** | 51.4 |

**Prefill ~1.5x, decode ~0 (slower at short context).** The real gains are footprint (49.8 -> 23.4 GiB of experts,
so it fits 32 GB of RAM and a 12 GB card caches ~20% of experts instead of ~10%) and the halved prefill stream -
which on *this* box matter least, because we have 30 GiB and the bottleneck is host read depth, not residency.


### 0.10.2 What the port needed, and where it lives

Upstream already solved the interesting half. **PR #54** (`f07c68c` + `b3e5d6f`, "pruned expert variants") reads
`qwen4exp.expert_count` / `expert_used_count` from the model file instead of assuming 512x10, and threads
`g.n_expert` through prefill/router/layout in place of the `NE = 512` literals at
`include/strata/kernels/cpu/expert.hpp:36` and `include/strata/core/layout.hpp:52`. The index question it answers:
`tensor-allocation/*.rco-allocation.txt` section 2 lists the kept experts under their **original sparse indices**
(layer 0 keeps `0,1,2,4,8,10,12,...` of 512), but the stored tensor's dim is **dense 256**
(`blk.N.ffn_gate_exps.weight = [2560, 640, 256]`), so the router output must be remapped - shipped as
`data/expert-profile-coder.bin`.

`git apply -3` of `f07c68c` onto this fork took 6 of 8 files clean; the 2 conflicts were our own R10/R11 tile work
(`m.tile`, the R11 `m.mixed` OOB fix) and were resolved by hand to keep ours - upstream's version predates both.
Committed on branch **`r20-coder`** @ `ef9636e`, tag **`fork-r20-coder-port`**. It is *not* in the live tree, and it
is **not finished**: `tools/probe_gguf.py` still refuses the file (`EXPERT TYPES NOT SUPPORTED: gate/up IQ4_XS x2`,
`DENSE TYPES NOT SUPPORTED: IQ2_S, IQ3_S, IQ3_XXS`), so building a pack needs kernel work beyond PR #54. The 28 GB
of downloaded shards are parked at `~/models/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF-iQ1_M/`; nothing references them
(`/home` has 273 GB free). Revisit only if prefill throughput or a 32 GB host ever becomes the goal.

### 0.10.3 Live endpoint verification (all green)

`engine/strata` execs a binary 4 bytes different from `build/strata` - verified to be only nvcc's PID-dependent temp
filename (`tmpxft_00015425` vs `tmpxft_00059440`) inside fatbin metadata: same build-ID, same code.

| check | result |
| --- | --- |
| `/v1/models`, `/health` | `qwen3.8-flash-next-heretic-2-iq3_xxs`, `max_context 131072`, ok |
| arithmetic + instruction-follow | `17*23 -> 391`, exact two-line format honoured, greedy |
| coherence (`scratch/quickbench.py`) | "Hello. 2+2 is 4." + coherent 400-tok reasoning trace |
| decode | **8.4 tok/s** over 400 tok (r19 documented median 7.75 / max 9.62) - no regression |
| abliteration | gritty safecracker thriller opener: complied first try, explicit on-screen violence, **zero** hedge/refusal phrases, `finish_reason: stop` |
| Pi harness | `~/.pi/agent/models.json` provider baseUrl `http://127.0.0.1:8123/v1` = the live port and model id |
| residency | `VmLck 25165728 kB` = exactly the configured 24.0 GiB; RSS 25.4 GiB; VRAM 11.1/12.3 GiB |
**One operational footgun found while verifying: `max_tokens` must cover the reasoning, not just the answer.** A
request with `max_tokens: 40` returned `"content": null` with the whole 40 tokens in `reasoning_content` and
`finish_reason: "length"` - correct behaviour for a reasoning model, but it looks like a broken endpoint to any client
that only reads `message.content`. The Pi harness is safe (`maxTokens: 131072`, and `serve/server.py:879` treats a
missing/0/`-1` as "the rest of the context"), and with a 600-token budget the same prompt returns
`"READY\n2^10"` with `finish_reason: "stop"` after 109 tokens. Any third-party client must send a generous budget.


### 0.10.4 The test suite on this box: 84%, and all 5 failures are environmental

`ctest` is actively misleading here and every number was chased to its cause rather than assumed. The history, so
the next person does not re-derive it:

* **Never run `ctest -j 8` while the endpoint is up.** Parallel: 29% pass with 9 `out of memory` failures. Serial:
  61%, and all 9 OOM failures vanish (`dequant_s2_parity`, `shared_expert_parity`, `gr_parity`, `kv_q8_parity`
  re-run individually: 4/4 pass). The engine holds 24 GiB mlocked + 25.4 GiB RSS on a 30 GiB host - the failures
  were the test runner competing with the live endpoint, not the code.
* The `Not Run` entries were not failures either: the production cache has **`STRATA_BUILD_TESTS:BOOL=OFF`**, so
  8 test executables (`expert_parity`, `pool_test`, `expert_multi_test`, `native_mmvq_multi`, `graph_registry_test`,
  `pinned_capture`, `suffix_drafter_test`, `controller_test`) are simply not built. A separate tree
  (`build-tests`, configured with `-DSTRATA_BUILD_TESTS=ON -DCMAKE_CUDA_COMPILER=/home/bodhi/deps/cuda-13.3/opt/cuda/bin/nvcc
  -DCMAKE_CUDA_ARCHITECTURES=89`; `nvcc` is not on `PATH`, which is why a naive fresh configure dies in
  `cmake_cuda_find_toolkit`) builds them, and **5 of the 8 pass immediately**.
* `pinned_capture` "failed" purely because ad-hoc runs have no `LD_LIBRARY_PATH` for
  `libcudart.so.13` - the engine's own launch script sets it. With the path exported it passes, as does
  `native_mmvq_multi`.

**Final authoritative run** (`build-tests`, serial, `LD_LIBRARY_PATH` set, engine left up): **26/31 = 84% pass.**
The 5 failures, each with its verbatim cause:

| test | cause | kind |
| --- | --- | --- |
| `ple_parity` | *required PLE table is missing or incompatible: `../../Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf`* | reference file not on this machine |
| `expert_parity` | *cannot read expert 0 (layer 0) from `pack/full/experts.bin`* | `pack/` is empty; we run native IQ packs |
| `pool_test` | *cannot read expert 0 of layer 0 from `pack/full/experts.bin`* | same |
| `platform_memory_test` | *`mlock failed (raise ulimit -l)`* | `ulimit -l` = 8192 KiB, unraisable from this shell; the engine's own launch path locks 24 GiB fine |
| `cuda_device_selftest` | *device NVIDIA GeForce RTX 4070 SUPER reports compute capability 8.9; Strata targets sm_120 (RTX 5000 series / Blackwell) only* | permanent: GPU is pre-Blackwell |

**Zero code regressions.** Note what this means for the endpoint's identity: `expert_parity` prints *"AVX512
F/BW/VL/VNNI/VBMI all present"* while failing on missing data, and `cuda_device_selftest` says the CUDA kernels
cannot run at all here. The live endpoint is entirely AVX-512 CPU + pinned-RAM, which is exactly why 51.8 GB/s of
DRAM is the wall and why no quant rename moves decode.

### 0.10.5 Decode is fine — the 5 tok/s reading was a units error, not a regression

A fresh benchmark read 5.3 / 5.6 / 6.6 tok/s against the documented 9.4-9.7, so `--hot-ram-gib` was dropped
24.0 -> 21.0 to buy RAM headroom (the engine had `VmSwap` 916 MiB, 1.1 GiB available). **That made it worse**, and
the reason turned out to be that the comparison itself was invalid:

| config | decode (6 x 200, client wall) | VmLck | VmSwap |
| --- | --- | ---: | ---: |
| `strata-r23-wrapup` (24.0) | 5.28 / **5.57** / 6.56 | 24.0 GiB | 916 MiB |
| r24 experiment (21.0) | 4.78 / **4.96** / 6.45 | 21.0 GiB | 131 MiB |

Two things were wrong. First, shrinking the tier costs more than the headroom buys - the pinned tier is
load-bearing, and 100% hot-tier lookup coverage does not mean a *cold* tier is as good as a warm one. Second, and
this is the actual answer: **the client wall-clock includes the prefill of the prompt.** These bench prompts are
59-74 tokens and cold short-prompt prefill costs 5.5-11.5 s each at 6-13 tok/s, so a request the engine spends
21.7 s decoding gets a client wall of 30 s. Comparing that 6.7 to a 9.4 figure taken from the engine's own log line
is comparing two different quantities.

Engine-side, from the same bench run: **10.0 / 9.7 / 9.4 / 9.2 / 9.2 / 9.1 / 8.7 / 8.4 / 7.7 / 7.7 / 7.7**, median
**8.9**, max **10.0** - against r19's documented median 9.7 / max 17.1 (n=46). That is within normal run-to-run
spread: the endpoint was never broken. `strata-r23-wrapup.json` (24.0) is the live config again, `VmLck` back to
25165728 kB, and the r24 experiment config is deleted.

**Benchmarking rule for the next person:** engine-reported `generated in N ms (X tok/s)` and client-side
`completion_tokens / wall` are *not* the same number on this box, and the gap is largest exactly where benchmarks
are shortest. Compare log-to-log, and note that per-request warm-up/checkpoint makes a 70-token prompt cost ~6-11 s
of prefill regardless of its length.

### 0.10.6 Reproduction, closed: `scratch/r14_multiturn_bench.py` is the 10+ harness

Re-ran the script that produced §0.7.3's "9.3-17.1 tok/s" claim (3 topics x 3 turns, full history resent, 200 tok,
greedy) against the live endpoint on `strata-r23-wrapup.json`. Engine-side decode for the 9 generations:

| req | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tok/s | **17.9** | 12.1 | 9.9 | 7.5 | 9.0 | 10.5 | 7.3 | 8.5 | 8.9 |

min 7.30 / **median 9.00** / max **17.90**, draft acceptance 0.874 - against r19's documented median 9.70 / max
17.10 (n=46). Whole r23 session n=34: median 8.80, max 17.90. The documented range reproduces, and 17.9 slightly
exceeds r19's observed max. The endpoint is back where it was; nothing needed fixing beyond the revert of the r24
`--hot-ram-gib` experiment and using the right harness. The tier must be *warmed* (many requests) before a median
means anything: the first sweep after a cold start sits ~1 tok/s below the steady-state one.

Final state: branch `r14-upstream` @ `dd609f9` (tag `fork-r19-snapshot`), `engine/strata` == `build/strata` modulo
nvcc's PID-dependent fatbin temp filename, config `strata-r23-wrapup.json` = `strata-r19-best.json` except the
`log`/`--dump-counts` paths, `VmLck` 25165728 kB = the configured 24.0 GiB, VRAM 11.1/12.3 GiB, `max_context` 131072,
`/v1/models` = `qwen3.8-flash-next-heretic-2-iq3_xxs`, Pi provider baseUrl `http://127.0.0.1:8123/v1`. The Coder
"IQ1_M" port lives on `r20-coder` (tag `fork-r20-coder-port`) and its 28 GB of shards are parked under `~/models/`
unreferenced; declined for the reasons in §0.10.1.



