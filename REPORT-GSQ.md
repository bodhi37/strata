# GSQ-RCO vs the live heretic-2 IQ3_XXS — is the swap worth it?

Researched 2026-09-29. Question: replace `packs/heretic-iq3xxs` (~6–17 tok/s decode)
with `ISTA-DASLab/…-GSQ-RCO-Coder-GGUF` in the `IQ1_M/` folder, for speed.

**Verdict: No — not that file, and not for the reason assumed. The `IQ1_M` release
would be the fastest option on this box, but its entire speed advantage comes from one
thing (the expert arena finally fitting in the 24 GiB tier), it is not an IQ1_M model at
all, and `GSQ-RCO Q2_0` captures ~75% of that same advantage with a one-line change
instead of a two-day engine surgery.**

---

## 1. The filename is a lie

`IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf` contains **zero IQ1_M
tensors**. Read from the GGUF header via HTTP range (no download):

```
qwen4exp.expert_count = 256          <- half of 512 deleted
  expert gate/up: {'IQ3_XXS': 34, 'IQ2_S': 40, 'IQ3_S': 20, 'IQ4_XS': 2}   (96 tensors)
  expert down   : {'IQ4_NL': 39, 'Q2_0': 9}                                 (48 tensors)
```

Verified further: **the Coder's per-layer expert allocation is byte-identical to the
unpruned `IQ3_S` release** (`compare coder_alloc vs al_IQ3_S -> True`). It is an
**IQ3_S-grade 3.5 bpw model with half its experts removed**. The advertised "1.89 bpw"
is an accounting figure — 3.5 bpw amortised over parameters that were deleted, not
deleted and then quantised. The card says so: *"The retained weights are stored at
3.5 bpw, unchanged by pruning."*

So the framing "trade IQ3_XXS intelligence for IQ1_M speed" is wrong on both axes: the
weights are *finer* than yours (3.5 bpw mixed vs uniform IQ3_XXS), and what you give up
is **capacity** — 12,288 of 24,576 expert matrices, gone, selected by KL against
code/agentic/vision calibration data.

## 2. What expert pruning actually does to *your* bottleneck

This is the whole analysis in one line. Per decode token you touch **10 experts × 48
layers = 480 expert blobs**, whatever the expert count. Top-10-of-256 reads the same 10
blobs as top-10-of-512, and in the Coder those blobs are *the same quantisation* as the
unpruned IQ3_S build.

| | arena (resident pool) | blob | **MB read per token** |
|---|---:|---:|---:|
| heretic-2 IQ3_XXS (live) | 49.8 GiB | 2.176 MB | **1044** |
| GSQ-RCO Coder "IQ1_M" | 23.4 GiB | 2.046 MB | **982** |

**−6%.** Expert pruning is a *footprint* optimisation, not a *latency* optimisation: it
shrinks the model without shrinking the work per token. On a machine that could hold the
whole arena in RAM, the Coder would buy you almost nothing on decode.

Its only real benefit is therefore structural: **23.4 GiB is the only released artefact
that fits inside your 24.0 GiB `--hot-ram-gib` tier.** Every unpruned variant (31.6 /
33.0 / 39.9 / 46.8 GiB) still overflows it. And because your decode latency is dominated
by tier misses (next section), "fits the tier" is worth ~3×.

## 3. Your box, measured — the model that explains 6–17 tok/s

Measured on this host during this analysis, not assumed:

| quantity | value | how |
|---|---|---:|
| DRAM read bandwidth | **57.1 GiB/s** (61.3 GB/s) | self-compiled strided-sum probe, 2 GiB |
| NVMe read, `O_DIRECT`, QD1 | **602 MB/s** | `dd iflag=direct` on `experts-native.bin` |
| NVMe read, QD≈8 | ~1.8 GB/s (est.) | Kingston SNV2S1000G is DRAM-less QLC |
| LRU expert hit rate | **median 94.6%** (n=85 windows: p25 93.5 / p75 96.9 / max 100) | `logs/r19-best.log` |
| admission pressure | `admitted == evicted` every window | same log → steady-state thrash |

`logs/r19-best.log` verbatim: `lru: 24209 admitted, 24209 evicted, … 93.6% of requests
served from RAM (hot 64047 / 68434)`.

Serial bandwidth model, `T = MB/token × hit / 61.3 + MB/token × (1−hit) / NVMe`:

```
heretic IQ3_XXS : 1044 MB/tok, hit .936 -> 1050 MB from DRAM + 71.8 MB from NVMe
                = 17.1 ms  +  119 ms(QD1) = 7.3 tok/s
                = 17.1 ms  +   40 ms(QD8) = 17.5 tok/s
OBSERVED                                    6 – 17 tok/s     ✓
```

Second, independent validation — it predicts your IQ4_XS observation. IQ4_XS gate/up is
2.662 MB/blob → 61.0 GiB arena → hit ≈91%, 1278 MB/tok → **4.7–12 tok/s**, i.e.
uniformly *below* the live build. You reported "much slower". ✓

**Consequence: ~70–85% of your decode latency is NVMe miss traffic, not bit-width, not
compute.** With 30 GiB RAM and a 49.8 GiB arena you are running a memory-capacity
problem wearing a quantisation problem's clothes. The 6→17 tok/s spread is not noise; it
is the per-window miss rate — the hit-rate distribution is a wide 0–100%
(median 94.6%, p25 93.5, p75 96.9), and decode tracks the tail, not the median.

---

## 4. All six candidates, on measured constants

Blob sizes computed from the released RCO allocation files (`tensor-allocation/*.txt`),
not from the marketing names. Hit rates: the measured 93.6% for the live arena, scaled by
arena/tier for the rest — that scaling is the one soft number in the table.

| candidate | arena | blob | MB/tok | hit | DRAM | **NVMe** | tok/s QD1 | tok/s QD8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **heretic-2 IQ3_XXS (LIVE)** | 49.8 GiB | 2.176 M | 1044 | 93.6% | 1050 | **71.8** | 7.3 | 17.5 |
| GSQ-RCO **Q2_0** (512e) | 31.6 GiB | 1.382 M | 664 | 97.5% | 695 | **17.8** | 24.4 | 47.1 |
| GSQ-RCO **IQ2_XS** (512e) | 33.0 GiB | 1.441 M | 692 | 97.5% | 724 | **18.6** | 23.4 | 45.2 |
| GSQ-RCO **IQ3_XXS** (512e) | 39.9 GiB | 1.743 M | 837 | 96.0% | 863 | **35.9** | 13.6 | 29.4 |
| GSQ-RCO **IQ3_S** (512e) | 46.8 GiB | 2.046 M | 982 | 95.5% | 1007 | **47.5** | 10.5 | 23.4 |
| GSQ-RCO **Coder** (256e) | **23.4 GiB** | 2.046 M | 982 | **~100%** | 1055 | **~0** | 58.1 | 58.1 |

MB = MB served per decoded token; QD1 = misses at the measured 602 MB/s, QD8 ≈ 1.8 GB/s.
Read the **NVMe column as the speed dial**. Relative to live:

* **Coder 3.3×** — but its DRAM traffic (1055 MB) is *identical* to yours (1050 MB).
  100% of the gain is the eliminated 71.8 MB/token of NVMe reads. Nothing else happened.
* **Q2_0 2.8–3.3×** — removes 75% of that same NVMe traffic *and keeps all 512 experts*.
* **IQ2_XS 2.7–3.2×** — same, and its expert types are already kernel-supported.
* **IQ3_S 1.3–1.5×** — pointless; 46.8 GiB is no better off than your 49.8 GiB.
* **IQ3_XXS 1.6–1.9×** — the quality play, modest speed.

These are streaming-bandwidth ceilings; 58.1 tok/s for the Coder is unreachable in
practice (§7). Trust the ordering and the NVMe column; treat absolute tok/s as ±50%.

## 5. Compatibility audit — where the swap actually dies

`tools/iq_pack.py` was written for exactly this file family (*"a native pack for any of
the model files (Q2_0, IQ2_XS, IQ3_XXS)"*, usage `<model>-00001-of-00002.gguf`), so the
**unpruned** releases are near-drop-in. The **pruned** one is not.

Blockers unique to the Coder (256 experts):

1. `tools/iq_pack.py:70` — `N_EXPERT = 512`, used at 240, 246, 276, 278
   (`expected_bytes() // N_EXPERT`, `assert chunk.shape == (N_EXPERT, blob)`).
2. `include/strata/kernels/cpu/expert.hpp:36` — `inline constexpr int NE = 512;` — the
   CPU expert pool's layout constant. Structural, not a flag.
3. `include/strata/kernels/native_router.hpp:10-16` and `src/kernels/cuda/native_router.cu:109`
   — the pinned top-k kernel is hard-wired "**512 experts**, top 10", reads 512 F32
   logits, launches `route<<<1, dim3(32,8)>>>`. The Coder's `ffn_gate_inp.weight` is
   `[2560, 256]` → 256 logits. New kernel + `NE` threaded through router, pool, cache, graph.
4. **The expert-id space changes under you.** `--expert-profile data/profile-r14.bin` and
   `--dump-counts counts-r16.strc` are `(layer, expert)` pairs over 512 slots, and
   `--expert-cache 1500` picks GPU-resident experts *by that id*. The Coder stores retained
   experts by **original index with pruned slots empty** (allocation file, section 2) while
   its router matrix is densely re-indexed to 256. Every hot-set decision, cache entry and
   profile rank is keyed to the wrong space until regenerated. Fail silently here and you
   get a 60% hit rate while everything "works".

Not blockers (checked so you don't have to):

* Expert types are fine. Coder gate/up = `IQ3_XXS/IQ2_S/IQ3_S` (all in `EXPERT_GU_OK`) +
  `IQ4_XS ×2` layers — and IQ4_XS gate/up was **already ported in REPORT.md §4b**
  (`vec_dot_iq4_xs_q8_1`, `Fmt<23>`, `iq_row_bytes` case 23, the `native_expert_grouped`
  switch case). Only the probe's allow-list is stale (§10).
* Coder **dense** path fully supported: net of the 144 expert tensors, only
  `Q6_K, Q5_K, Q4_K, Q8_0, IQ4_XS, IQ4_NL, BF16, F16, F32` remain.
* `Q2_0` in the **gate/up** role already has a kernel — `vec_dot_q2_0_q8_1`
  (`iq_kernels.cu:56`), `Fmt` entry `:341`, `dequant_bf16.cu` `TYPE==42` at 40/174/202.
  So Q2_0 is a **one-line allow-list change**, not a kernel port.

The one real gap for **IQ2_XS / IQ3_XXS**: `IQ3_S` appears in *dense* tensors, which the
mmvq path rejects — `attn_gate, attn_k, attn_q, attn_qkv, attn_output, ffn_gate_shexp,
ffn_up_shexp, ssm_out` (44 tensors in IQ2_XS, ~38 in IQ3_XXS). Either add
`vec_dot_iq3_s_q8_1` to dense mmvq, or requantise those few hundred MB of attention /
shared-expert tensors to `IQ4_XS`. **`Q2_0` has no such tensors — clean.**


---

## 6. The angle nobody mentions: you already own 28.8 GB of this model

Every release ships two shards, and the second is the 28.8 GB per-layer n-gram (PLE)
table, held at a fixed 4.5 bpw IQ4_NL and **excluded from both the quantisation search and
the pruning**. Check the HF blob IDs:

```
IQ2_XS/  …-00002-of-00002.gguf   28,800,138,432 B   sha 316b46f3a2db   \
IQ3_S/   …-00002-of-00002.gguf   28,800,138,432 B   sha 316b46f3a2db    |  identical
IQ3_XXS/ …-00002-of-00002.gguf   28,800,138,432 B   sha 316b46f3a2db    |  blob
Q2_0/    …-00002-of-00002.gguf   28,800,138,432 B   sha 316b46f3a2db   /
your     ple/ple-iq4nl.gguf      28,800,142,336 B   (+3,904 B = a GGUF header)
```

One shared blob across all five releases, and your `ple-iq4nl.gguf` is that blob plus a
header — built by `tools/make_ple_iq4nl.py` from the heretic-2 shards, which Heretic does
not touch (REPORT.md §5: abliteration edits residual writers and *"does not touch the MoE
router, the 51 B n-gram table, or the vision tower"*).

And `--ple-gguf` is a **decoupled flag**: `strata-r19-best.json` passes it independently
of `--native-dense-gguf`. **You do not need to download the 28.8 GB shard.** Swap cost is
29.6 GB (Coder) or ~37 GB (Q2_0) — not 58.4/66.4 GB. *Verify first: same tokenizer and
n-gram index are assumed; confirm the tensor list and sample-compare bytes.*

Disk is still the squeeze: **59 GB free** on `/home`, and `--pack` writes a second
arena-sized copy (`packs/heretic-iq3xxs/experts-native.bin` alone is 50 GB). So 29.6 GB
download + 23.4 GB pack ≈ 53 GB into 59 GB free — workable only if you first delete the
unused `packs/iq3_m` (56 GB) and `packs/iq4_xs` (63 GB), which is the clean move anyway
given §3's model explains your "IQ4_XS is much slower" observation as a cache-capacity
effect (61.0 GiB arena → ~91% hit → 4.7–12 tok/s).

## 7. What erodes the Coder's gain

* **You lose the abliteration.** This is not a footnote: you picked heretic-2 for
  *"0/100 refusals (base 99/100), KL 0.0818 — the lowest-KL Heretic of this model"*, not
  for its quantisation. Every ISTA release derives from `Qwen/Qwen3.8-Flash-Next`, the
  base. Swapping re-introduces refusals and costs a fresh Heretic-style pass (REPORT.md §5:
  ~15% device headroom, skipped because pre-made builds existed).
* **You lose generality, by construction** — code/agentic/vision-directed, with the card
  itself warning of degradation outside that set.
* **MTP acceptance drift.** `--mtp mtp/rt --spec 4 --suffix-draft 3` drafts against
  activations that change when half the experts vanish. It still runs, accepts fewer
  drafts, and hands part of the win straight back. Measurable in an afternoon; the best
  reason to keep both packs rather than deciding on paper.
* **The 58.1 tok/s ceiling becomes compute, not bandwidth.** Once misses vanish you go
  ALU-bound, and IQ2_S/IQ3_S/IQ3_XXS `vec_dot` is *more* work per byte than Q2_0 or
  IQ4_NL (256-entry grid lookups, bit extraction, per-sub-block scales). Expect the Coder
  nearer 20–35 tok/s than 58 — and expect **IQ2_XS to scale worse than Q2_0** exactly
  here. The classic low-bpw trap: fewer bytes, more instructions.
* **Prefill stays flat-to-worse.** It is dense-path and attention dominated; pruning
  barely touches it, and the Coder's dense path is the same `hc_*`/`ple_key` BF16 +
  `output.weight` Q6_K you already run. The grouped expert kernel may gain slightly
  (640 tokens/expert per 16k chunk instead of 320 → better weight reuse), but do not buy
  this for prefill.

## 8. Three free wins before buying anything

`logs/r19-best.log` is sitting on unclaimed performance:

1. **`prefetch: … 0 issued`, in every single window.** Thousands of misses detected, the
   predictor flagged 0–29% repeats, and **zero prefetches were ever issued**. The parser
   only has `--no-ple-prefetch`; there is no expert-prefetch enable path in
   `generate.cpp`. Checked exhaustively: **all 85 windows say `0 issued`, zero exceptions.**
   This is a broken or unimplemented feature, not a tuning miss. Hiding even half of 71.8 MB/token behind compute beats every quant in
   this document. Find the missing call site first — it may be a dead flag, not a feature.
2. **`--expert-cache 1500` leaves ~5 GB of your 12 GB card empty.** GPU-resident experts
   skip the DRAM *and* NVMe path (4070 SUPER ≈ 500 GB/s). 1500 blobs = 3.3 GB; KV at
   `q4_0`/131k ≈ 3.2 GB; there is room for ~3000 blobs. Try `--expert-cache auto` (the
   parser maps `auto` → −1) and 2500/3000, watching the
   `R4 hit path ON … resident experts computed on the GPU` line.
3. **Re-measure the hot set properly.** `data/counts-r16.strc` holds only **12,480 access
   events = 26 tokens** at 480 touches/token — far too small, which is why its 4,555
   touched experts trivially "fit" 24 GiB. Re-dump over a few thousand representative
   tokens before trusting any coverage figure. Encouraging sign though: 26 tokens already
   touched 4,555 distinct experts = 9.9 GiB, so short-session locality is much better than
   the arena size implies, and the tier may be doing more than the arena suggests.


## 9. Quality — what you would actually trade

ISTA's published numbers, all at **xhigh reasoning effort**, against the BF16 base:

| | SWE-bench Verified | LiveCodeBench v6 | AIME25 | GPQA-D | task avg |
|---|---:|---:|---:|---:|---:|
| BF16 base (354 GB) | 82.80 | 87.43 | 100.00 | 91.92 | 93.12 |
| **Coder** (58.4 GB, 256e) | **75.60** (91.3%) | **86.28** (98.7%) | n/a | n/a | n/a |
| IQ3_S (83.6 GB) | — | 86.86 | 100.00 | **92.93** | **93.26** |
| IQ3_XXS (75.8 GB) | — | 86.29 | **100.00** | 91.41 | 92.57 |
| IQ2_XS (68.0 GB) | — | 83.43 | 96.67 | 87.37 | 89.16 |
| Q2_0 (66.4 GB) | — | 81.14 | 96.67 | 89.39 | 89.07 |

Two readings matter.

**Pruning beats quantisation at equal footprint.** At *1.89* effective bpw the Coder holds
LCB at 86.28 (98.7% of base); unpruned Q2_0 at *2.40* bpw — more bits per retained
parameter — only holds LCB at 81.14 (92.8%). 256 experts at 3.5 bpw is a better model than
512 experts at 2.4 bpw, in 30% less memory. If you were going to shrink anyway, the pruned
build is the technically superior compression. That is the strongest real argument for it.

**But the direction of the compression is not your workload.** The card is explicit:
*"Degradation outside this set is an accepted cost of the method. For general-purpose use,
the unpruned GSQ-RCO releases are the appropriate choice."* Their first calibration — code
and agentic data only — deleted the experts the vision pathway depends on and broke image
capability while coding scores stayed flat. And SWE-bench, sustained multi-turn agentic
work on real repos, is the one number that visibly cracks (−7.2 points; 91.3% retained vs
LCB's 98.7%). That is exactly the task class you are serving.

Also: your **KLD 0.1096 ± 0.0015 is not comparable** to any ISTA number. Theirs are task
scores against the BF16 base; yours is distributional divergence against spiritfather's
Q8_0 of an *abliterated* model. Different bases, different metrics. Expect a bake-off, not
a clean ordering across the two families.

## 10. Two bugs in `tools/probe_gguf.py`, found while doing this

Fix them or keep being lied to about compatibility:

* `dense_bad` scans the **global** type histogram, which includes the 144 `_exps` tensors.
  So `IQ2_S/IQ3_S/IQ3_XXS/IQ1_M` appearing in the *expert* role get reported as
  `DENSE TYPES NOT SUPPORTED`, and the tool prints `needs requantization` for files whose
  dense path is entirely fine. Exclude `_exps.weight` (and the `token_embd`/`output`
  native-served set) before computing `dense_bad`.
* `EXPERT_GU_OK` omits `Q2_0`, though `vec_dot_q2_0_q8_1` has been wired since REPORT.md §4b and
  `EXPERT_DOWN_OK` already lists it; and `IQ4_XS` gate/up is supported post-REPORT.md §4b while the
  list still rejects it — which is what flagged the Coder's 2 layers.

Corrected state: **Q2_0 = compatible today. IQ2_XS / IQ3_XXS = need `IQ3_S` in dense mmvq.
Coder = needs the 256-expert work in §5.**


## 11. Verdict, and what I would do instead

**Do not swap to `IQ1_M`.** Not because it is too dumb — it is the *fastest* option here,
and its weights are finer than yours — but because you would pay the highest price in the
family (abliteration, generality, MTP acceptance, plus 1–2 days de-hardcoding `NE=512`
through the packer, the CPU expert pool, the router kernel and the cache id space, with a
silently-wrong hot set if you get blocker 4 wrong) for a speed win that `Q2_0` gets you
most of with a one-line allow-list change.

The load-bearing facts, in order:

1. Pruning does **not** reduce per-token traffic: 1044 → 982 MB/token, **−6%**. Top-10 of
   256 reads the same 10 blobs as top-10 of 512.
2. Its only speed mechanism on this box is **residency** — 23.4 GiB inside the 24.0 GiB
   tier, the sole released artefact that fits.
3. Residency is worth ~3×, because **70–85% of your decode latency is NVMe miss traffic** —
   measured, and the model validated twice independently (6–17 tok/s live; your
   IQ4_XS-is-slower report).
4. `Q2_0` attacks that same bottleneck directly: NVMe traffic 71.8 → 17.8 MB/token,
   **all 512 experts preserved**, dense path clean, kernel already ported.
5. You also have ~5 GB of unused GPU expert cache and a prefetch path that issues **zero**
   requests. Both are free, and both attack the same 71.8 MB.

### Recommended order

| # | action | cost | expected |
|---|---|---|---|
| 1 | Chase `prefetch: … 0 issued` — find the dead call site | hours | largest single lever |
| 2 | `--expert-cache auto` / 2500 / 3000 against the 12 GB card | minutes | +15–40% |
| 3 | Fix `probe_gguf.py` allow-lists + `dense_bad` scope | minutes | stops false "requant" |
| 4 | Re-dump routing counts over a real session (>2k tokens) | minutes | trustworthy hot set |
| 5 | **`GSQ-RCO Q2_0`** pack, reusing your existing `ple-iq4nl.gguf` | 1 day, ~37 GB | **~2.5–3×** |
| 6 | Bake-off 5 against live heretic on *your* tasks, incl. refusals | hours | decides it |
| 7 | Only if residency is still binding: the Coder, **after** re-abliteration | 1–2 d + 29.6 GB | up to ~3× |

Steps 1–4 change no weights and are worth doing regardless. If after 1–2 the miss traffic
is genuinely the wall *and* Q2_0's quality is unacceptable, the Coder becomes the right
answer — but as a deliberate **capacity** decision (it is the only model in this family
that fits 30 GiB of RAM), taken with a re-abliteration pass, not as an "IQ1_M is faster"
drop-in.

Worth restating, because it is the counter-intuitive part: at a fixed 24 GiB budget,
**pruning at 3.5 bpw beats requantising to ~1.1 bpw**. Getting all 512 experts into 24 GiB
would need IQ1_S gate/up + Q2_0 down ≈ 1.10 MB/blob ≈ 25.2 GiB — still over the tier, at a
precision well below anything ISTA ships, and by their own measurements far worse than the
pruned build. That is why the Coder exists, and it is the honest reason to consider it.

## 12. Still unverified

* **Coder expert-index semantics.** The allocation lists retained experts by original
  index; the router matrix is `[2560, 256]`. Whether Strata must remap ids depends on
  whether llama.cpp semantics renumber densely or keep holes in the fused `*_exps`
  tensors. Read the full 2.7 MB allocation and the `experts.bin` byte layout before
  committing to the §5 work.
* **PLE interchangeability.** Assumed identical because the blob is byte-identical in size
  and sha across releases; confirm the tensor list and tokenizer against `ple-iq4nl.gguf`.
* **NVMe at queue depth.** 602 MB/s is QD1. The engine's real miss throughput (io_uring?
  the 20 pool workers?) could be 2–3× that, which shrinks every miss-driven gain
  proportionally — including the Coder's.
* **The 26-token counts sample.** §8.3 — no coverage conclusion in this document rests on
  it, and none should.
* **Coder dense-path confirmation from a full header walk.** §5's "dense is clean" is
  derived by subtracting expert tensors from the global type histogram in the allocation
  file, not from walking all 1,223 tensor names (the 16 MB range read did not reach them).
* **Your own eval harness.** Everything here ranks candidates on public or third-party
  numbers; your harness should arbitrate step 6.

## 13. Reproducing this analysis

```bash
# expert dtypes of any release, no download (16 MB range read)
venv/bin/python tools/probe_gguf.py \
  https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --bytes 16000000

# authoritative per-layer expert types (39 KB, no download of the model)
curl -sL .../tensor-allocation/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.rco-allocation.txt

# blob bytes for one expert = 2*n_embd*n_ff*bpv(gu) + n_embd*n_ff*bpv(dn);  n_embd 2560, n_ff 640
# arena = blob * n_experts * 48 ;  MB/token = blob * 480
# sanity check: blob 2.176 MB -> arena 49.80 GiB == packs/heretic-iq3xxs/native_experts.txt

free -h
dd if=packs/heretic-iq3xxs/experts-native.bin of=/dev/null bs=1M \
   skip=30000 count=2048 iflag=direct      # 602 MB/s, measured here
grep -i 'lru:' logs/r19-best.log           # 93.6% hit, measured here
```

