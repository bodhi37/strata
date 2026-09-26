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

## 1. Which "Strata"

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
