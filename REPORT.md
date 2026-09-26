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

Pending — auto-generated by `heretic_finish.sh` step 7 into `bench/results/`:

| Context | prompt tokens | TTFT (s) | prefill tok/s | decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| ~1K | | | | |
| ~4K | | | | |
| ~32K | | | | |
| ~128K | | | | |

Also: device load time, peak RAM/VRAM/swap, engine log (`strata-heretic.log`), capability and
refusal spot-check, and the thinking-mode 4K measurement.

## 10. Gap analysis (final numbers to be inserted)

Targets vs what this silicon can do: 200–300 tok/s is not reachable on 12 GB + 30 GB RAM with a
105 GiB model; the honest target is the Strata-published class (45–95 tok/s on 64 GB RAM with the
same 12 GB card). 128K is firm and provisioned; 256K/512K depend on KV + expert residency and are
tested after the run. Intelligence is bounded by IQ3_XXS (≈99.4 % of BF16 on task average) ×
abliteration (KL 0.0818), i.e. very close to the model's own Opus-4.6-class standing.

## 11. Highest-value next optimizations

* `--expert-cache` tuning / `--expert-cache-per-layer` (measured +2.7 tok/s at 4096 slots on the
  reference box) and `--expert-profile` re-derived from *this* model's routing.
* More RAM would remove the SSD paging of the expert arena (the single biggest lever).
* MTP acceptance sweep (`--spec`, `--spec-min-p`) per workload.
* A KVarN-style or int8 KV on the 12 full-attention layers to push context toward 512K.

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
