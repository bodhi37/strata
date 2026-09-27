# MISSION — Autonomous Hillclimb: Qwen3.8-Flash-Next on Strata (30–50+ TPS Decode, 500–1000 TPS Prefill)

**Read this entire document before touching code or configs. It replaces all prior mission briefs.**

You are the optimization engineer running this system. Your directive is to **autonomously attack, innovate, and aggressively hillclimb** this local inference endpoint until the hardware is genuinely redlined.

The catastrophic zero-day bugs (4KB faulting, read amplification, prefill OOB corruption, IQ4_NL sign flip, GPU persistence failure) are **diagnosed and solved**.
You are taking a working, coherent, measured baseline (~6–9 tok/s median decode at 200 tokens, 10–14 tok/s at 150–300 tokens, 694 tok/s prefill) to **30–50+ tok/s decode across ALL agentic workloads** and **500–1000 tok/s prefill**, supporting **80k/128k/256k context with giant reasoning and answer budgets** in Pi/OpenAI/Anthropic harnesses.

Push until the silicon genuinely has nothing left to give. Treat any claim that "the hardware can't" as a challenge to be shattered by measurement.

---

## 0. Prime Directive: Autonomous, Aggressive Innovation

Do not wait for permission. Do not test one safe knob and give up. **Attack this from every possible optimization angle.**
Under current loads, the hardware is **practically idle**:
- **GPU VRAM:** 12 GiB card, yet running with only 800 expert slots (~1.7 GiB) — ~95% of VRAM (incl bandwidth) is unutilized or stranded.
- **Host RAM & DRAM Bandwidth:** 30 GiB DDR5 system RAM capable of ~60-80 GB/s bandwidth (I will OC to 6200MHZ next reboot, so even more), yet the engine is blocked on serial NVMe reads, idling the memory bus.
- **CPU Compute:** 12C / 24T Ryzen 9 9900X (AVX-512, VNNI, BF16) with 64 MiB L3 cache across 2 CCDs sitting idle while waiting on serial disk queues (PC has 11 case fans at max speed + AIO pump running at full, do not be scared of thermal throttling, I get <30C idle at max performance/pinned to 5.6ghz).
- **NVMe Bandwidth:** NVMe is capable of ~1.0 GB/s sequential reads (the SSD can do like 4+GB/S, or even double that), yet decode uses only ~35–190 MB/s because reads arrive 1–2 per layer at serial QD1.

YOU'RE ONLY USING 5% OF THE COMPUTERS POWER

Your job is to systematically eliminate every idle cycle and every bottleneck across the entire memory and compute hierarchy.

---

## 1. Ground Truth & Anti-Hallucination Guardrails

| Outdated Myth | Verified Ground Truth Today |
|---|---|
| "Engine does 0.4–1.0 tok/s; find 100x bug" | **Fixed.** Live baseline on `strata-r13-base.json` is **6.3–9.2 tok/s** decode (200 tokens across 6 varied topics) and **694 tok/s** prefill. |
| "Upstream is github.com/niko1221/Strata (404)" | Upstream is **`https://github.com/Niko1221/Strata`** (capital `N`, single `k`). Tracks up to engine v0.1.8. |
| "KV-cache cross-turn agent is active; don't touch" | **Finished & merged.** Cross-turn prefix reuse (`GEN ... KEEP <N>`, `sess_record`, `serve/test_prefix_cache.py`) is live in `serve/server.py`. Use it as a throughput weapon. |
| "Configs follow -v3 ... -v6" | Superseded. The active progression is `strata-r7-*.json` through `strata-r13-*.json`. |
| "Correctness is a follow-up" | **Correctness is a hard regression gate.** Coherence must be verified (`scratch/r11_coherence.py` or `scratch/r11_decode_bench.py`) before logging any speed win. Fast garbage = 0 tok/s. |
| "zram compression gives 2–4x RAM" | **Dead.** Payloads are high-entropy (lz4 1.000x, zstd 1.008x measured in R7). |
| "Packs are in `packs/heretic-iq3_m`" | Directories on disk are `packs/heretic-iq3xxs` (49.8 GiB arena), `packs/iq3_m` (57.9 GiB), and `packs/iq4_xs` (65.4 GiB). |
| "Run multiple engines concurrently" | **Never.** 30 GiB RAM cannot fit two engines. Two engines = instant OOM-kill of your shell/process. Always stop the running engine and wait 10s before launching another. |

---

## 2. Targets & Measured Baseline

| Metric | Minimum Gate | Target / North Star | Measured Baseline Today (IQ3_XXS, `strata-r13-base`) |
|---|---|---|---|
| **Decode (tok/s)** | **30–50** | **100** | **6.3 / 8.0 / 9.2** (min/median/max 6-topic 200-tok); 10.5–11.4 @150tok; ~14.1 @300tok |
| **Prefill (tok/s)** | **500** | **1000** | **694** warm at 11k ctx (R10.5). Peaks 845 MB/s on cold drive staging. |
| **Context** | **80k–128k** usable | **256k** | 131,072 ctx with `--kv int8` (or `--kv q4_0` / streaming). |
| **Generations** | Giant agentic budgets | Unlimited | Zero truncation, full reasoning depth across all topics. |
| **Coherence** | 100% coherent | Zero drift | Confirmed clean (sourdough, python merge, law, mRNA, sqrt(2), agentic). |

> **Measurement Rule:** A benchmark without completion length, prompt context length, topic set, cache state (cold vs warm), and exact config file is worthless.

---

## 3. The Bottleneck: The 48-Layer Serial QD1 Latency Wall

* One decode token routes to 480 expert blobs (~1.044 GB in IQ3_XXS, 2.18 MiB each) across **48 strictly serial layers** (layer $L$'s router depends on layer $L-1$'s hidden state).
* With our 24.5 GiB mlocked RAM tier, routing coverage is **~97.2%**. The remaining **2.8% (~29 MB/token)** misses to NVMe.
* While 29 MB/s is nothing for drive bandwidth, the misses arrive 1–2 per layer sequentially. Drive queue depth is **QD1–QD2**, paying full **2.8 ms per read**. 48 × 2.8 ms = **134 ms/window latency wall** (giving ~7–9 tok/s decode).
* Bandwidth tricks alone (`O_DIRECT`, chunked `kSplit`, parallel readers) cannot bypass a serial latency wall.
* **The Arithmetic Escape:**
  - At **$\ge 99.1\%$ resident coverage**, ~82% of layers have **zero misses**, collapsing window wait time from ~134 ms down to ~25 ms $\rightarrow$ **30–50+ tok/s immediate speedup**.
  - Achieving $\ge 99.1\%$ coverage requires **~34.8 GiB resident** of the 49.8 GiB arena. Host RAM cannot fit this alone (30 GiB total).
  - Therefore, the **only physical path to 30–50+ TPS is aggressive multi-tier co-residency (RAM tier + VRAM tier + host RAM reclaim)** combined with latency-hiding innovations.

---

## 4. Key Upstream Innovations (`Niko1221/Strata` v0.1.2–v0.1.8)

Upstream has landed breakthroughs that directly blow open our bottlenecks. Inspect and leverage them:

1. **VRAM Expert Cache Parity & Accuracy Confirmed (PR #21 / Issue #23):**
   - The startup warning *"GPU hit path is NOT CORRECT"* was a relic from an ancient bug. Upstream teacher-forced parity tests on 2,557 tokens show **95–98% identical top-1 predictions and identical perplexity (-0.005 ± 0.005 nats)**.
   - **Leverage:** The VRAM tier is safe! Scale `--expert-cache` from 800 (~1.7 GiB) up to 3,000–4,096 blobs (~6.5–8.9 GiB) with `--expert-cache-per-layer`.
2. **Q4_0 KV Cache with FWHT-256 Hadamard Rotation (`--kv q4_0`, PR #21):**
   - Orthonormal Hadamard transform of K/V/Q before 4-bit quantization halves KV footprint (576 B/token vs 1,056 B in INT8) while passing 5/5 needle tests up to 262k context.
   - **Leverage:** Frees 1.5–3+ GiB of VRAM at long contexts, which can be immediately dedicated to VRAM expert storage!
3. **KV Streaming for QSA Layers (`--kv-resident`, Engine 0.1.5):**
   - Stores KV in pinned RAM and streams selected attention blocks to a compact VRAM window (e.g. `--kv-resident 32768`). Frees up massive VRAM headroom for the expert cache.
4. **Prompt Lookup Drafter (`--suffix-draft`, Engine 0.1.7):**
   - Learned drafter for context repeats (code editing, agentic history, quotes) drafting up to 5 tokens. Boosts decode by +6–11% on agentic tasks without output drift.
5. **Short-Prompt Windows (`--short-read 64`, PR #10):**
   - Prompts $\le 64$ tokens route through decode verify windows instead of full batched prefill, cutting first-token latency by ~300 ms on short tool turns.
6. **Thread Pool Condvar Sleep & Sampling Bitmaps:**
   - Eliminates 100% core spinning between requests (PR #9) and removes $O(k \cdot n_{vocab} \cdot hlen)$ latency on sampled paths (commit `3e0836f`).

---

## 5. The Autonomous Hillclimbing Attack Vectors

Iterate, benchmark, and innovate aggressively across these fronts:

### Vector 1: Maximize Multi-Tier Residency (The Primary 30–50 TPS Lever)
- **VRAM Expert Tier Expansion:** Sizing `--expert-cache 2048` to `4096` with `--expert-cache-per-layer`. With 3,500 blobs in VRAM and 11,500 in RAM, total resident blobs reach ~15,000 / 24,576 ($\ge 99\%$ coverage).
- **Host RAM Reclaim:** Dense pool (~1,416 MiB) and token embedding (~644 MiB) are duplicated between host RAM and VRAM. Reclaim them on host to expand `--hot-ram-gib` from 24.5 to 26.5 GiB (+1,000 expert blobs).
- **Adopt Upstream `--kv q4_0` and `--kv-resident`:** Shrink KV cache in VRAM to free maximum memory for expert blobs at 80k–128k context.
- **Trace Profile Refresh:** Rebuild the profile binary from `data/counts-r10.strc` to ensure the most frequent routing pairs occupy the resident tiers.

### Vector 2: Compute Offloading & Overlap
- **Sweep `--pcie-frac` (0.55 $\rightarrow$ 0.80 $\rightarrow$ 1.0):** Offload missed experts to RTX 4070 SUPER PCIe execution rather than waiting on the CPU pool.
- **Dual-CCD Topology & Worker Locality:** Ryzen 9 9900X has two 6-core CCDs with split L3 cache. Test pinning reader threads to one CCD and compute workers to the other, or tuning `--pool-workers` (12 vs 16 vs 20).
- **Park Spin Tuning:** Benchmark `STRATA_PARK_SPIN_US=250` (`strata-r13-spin.json`) to eliminate context-switch latencies on CPU pool dispatch.

### Vector 3: I/O Latency Demolition
- **Prefill Admissions (`STRATA_PREFILL_ADMIT=1`):** Benchmark `strata-r13-admit.json`. Prefill sweeps ~53% of the arena; admitting these reads warms the cache for subsequent decode steps.
- **Coalesced Layer-Major Reads:** Misses currently issue as isolated 2.18 MiB reads. Group/coalesce contiguous blob offsets per layer into unified reads to cut multiple 2.8 ms seek latencies.
- **Speculative Amortization:** Benchmark `strata-r13-spec6.json` (`--spec 6`, `--spec-min-p 0.8`) and evaluate upstream `--suffix-draft 3` to increase tokens generated per verify window.

### Vector 4: Multi-Quant & Full-Context Validation
- **Support All 3 Quants:** Verify `IQ3_XXS`, `IQ3_M`, and `IQ4_XS`. Fix aliases in `LAUNCH-ALIAS.md` to point to real paths (`packs/iq3_m`, `packs/iq4_xs`).
- **Agentic Stress Testing:** Validate 80k/128k context prompts with deep reasoning chains (`reasoning_effort` enabled, large completion limits). Guarantee no truncation or mid-thought stalls.

---

## 6. Execution SOP & Tooling

1. **Launch Server:**
   `bash srv.sh <config.json> 8123` (Always bench on port 8123+ to avoid port 8111 conflicts).
2. **GPU Persistence Mode:**
   `echo 'Monster.8!!!' | sudo -S nvidia-smi -pm 1` (Must be active; verify if engine reports device busy).
3. **Engine Lifecycle:**
   - Wait 10 seconds after stopping an engine before starting a new one.
   - Stop cleanly: `bash srv.sh stop 8123` or kill by PID. Never use plain `pkill -f 'serve/server.py'`.
4. **Benchmarking & Validation:**
   - Step 1: Verify coherence: `python3 scratch/r11_coherence.py 8123`.
   - Step 2: 6-topic 200-tok decode sweep: `python3 scratch/r11_decode_bench.py 8123 200 warm`.
   - Step 3: High-context & reasoning eval: `python3 scratch/r11_decode_bench.py 8123 1000 warm`.
   - Log all configs, metrics, and observations into `REPORT.md`.

Do not hold back. Innovate, experiment, profile, iterate, and conquer the wall.
