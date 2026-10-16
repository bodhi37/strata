> [!NOTE]
> This branch (`orca-port`) is frozen and outdated — use `main` for the latest.

## `orca-port`: what this fork changes

Goal: serve the model on boxes that cannot hold it in RAM, via a VRAM + hot-RAM + NVMe expert tier, plus Orca support and `serve/` hardening.

### Expert serving (`src/core/expert_source.cpp`, `include/strata/core/expert_source.hpp`, `src/program/generate.cpp`, `src/core/expert_cache.cpp`)
- **Three tiers.** VRAM slice (non-redundant, past the hot tier) + mlocked hot RAM + NVMe via `pread` ring and parked reader pool. Startup helper: `run-engine-mlock.sh`.
- **Routing profiles from traces.** `--dump-profile` / `--dump-counts` + `tools/make_profile_from_counts.py`; `--expert-cache -2` as the no-cache baseline.
- **VRAM fill fixes.** Per-layer slot bases in `open_sized`; fill scans past per-layer quotas and verifies the first-admitted pair.
- **Read-amplification fix (6.4x).** Hot-tier guard on the ring-full fallback (2.4 → 6.1 tok/s on IQ3_XXS); whole-blob `pread` + `DONTNEED` + `MADV_HUGEPAGE` in `pin_hot`; two-pass dispatch (`begin_layer` / `wait_layer`) so resident experts compute while reads fly. Remaining wall is SSD QD1 latency (~2.8 ms × 48 layers), not bandwidth.
- **Adaptive LRU hot tier.** Seeded in profile rank order; zero-copy admit on decode miss; frequency-aware eviction; phase attribution.
- **`O_DIRECT` reads.** Twin `O_DIRECT` fds with thread-local aligned bounce buffers; avoids the per-read page-cache alloc/copy cycle (~39% CPU before). Buffered fallback preserved; `STRATA_NO_ODIRECT=1` disables it. Also surfaces `cudaErrorString` on `NativeEmbed` pin failure.
- **Elastic tier under memory pressure.** `cudaHostRegister` pins pages, so a 24 GiB tier plus KV pools could force swap and kill `cublasCreate`. Fix: register in ~2 GiB slices; `shed_pressure()` unregisters cold slices, `regrow_pressure()` adds one back when `MemAvailable` recovers. Server polls every 2 s (`STRATA_MEM_FLOOR_MIB=1536`) and sheds only between requests.

### Prefill (`src/prefill/prefill.cpp`, `src/prefill/kernels.cu`)

- **Tiling, not bigger chunks.** Chunk decoupled from scratch VRAM (chunk 16384, tile 768); buffers allocated per run and freed after, so decode is never starved. Warm 11k prefill 50 → 694 tok/s.
- **Fixed two correctness bugs.** Out-of-bounds write when chunk > tile (degenerate `!!!` output); GEMM state (`cuBLAS` handle, scratch) now created once in `init` instead of per prompt.

### CPU / CUDA kernels (`src/kernels/cpu/iq_avx512.cpp`, `src/kernels/cpu/expert.cpp`, `src/kernels/cuda/native_moe.cu`, `src/kernels/cuda/sampler.cu`)

- **AVX-512 `IQ4_NL` multi-token kernel** (~3x per-token ggml at `nt=5`); fixed a sign bug (`vpsignb` dropped weight signs, caught by `iq4nl_fuzz`).
- **Orca `IQ4_XS` decode-once kernel** (`row_dot_iq4xs`, `gu_rows_iq4xs`); fuzz-clean vs ggml AVX2, parity at activation-noise level. Uses a static host tier with prefill-admit off (LRU churn halved throughput in testing); `--expert-cache-per-layer` and `--dump-counts` supported.
- **`native_moe` combine + sampler.** Shared-memory weight broadcast, `fmaf`, `LDG.128` path; sampler caches unique token counts (bit-identical, much faster on long histories).

### Orca support (`tools/`, `strata-orca-*.json`, `src/program/generate.cpp`)

- **OOM-safe tools.** `make_native_head.py` and `probe_gguf.py` stream tens-of-GB shards instead of loading them; `repair_heretic_pack.py` converts `IQ4_XS` / `IQ4_NL` to BF16.
- **Endpoint example.** `strata-orca-iq4xs.json`: port 8104, 24 GiB hot tier, 1500-slot cache, 16k prefill, spec-4, `q4_0` KV, 128k ctx.

### `serve/` hardening (`serve/server.py`, `srv.sh`, `serve/test_server.py`)

- **No spurious 400s.** Over-budget `max_tokens` is clamped to room left; engine faults return a `finish_reason` (retryable or overflow), never a bare EOF.
- **Long-request tolerant.** 300 s watchdog; failover waits with heartbeats instead of failing fast; stale-output guard replaces a desynced engine; infra faults auto-replace the engine in the background.
- **Covered by 61 contract tests** (23 + 20 + 18).

---
<h1 align="center">Strata</h1>

<p align="center"><b>Run a 125-billion-parameter AI model on a normal gaming PC</b><br>
one NVIDIA card (12-24 GB) + 64 GB of RAM · Windows or Linux · one click to install</p>

Strata runs **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** - a large, smart AI model that
normally needs a server - on your own PC. It writes its answers at **60-95 tokens per second** (a token is about ¾
of a word): faster than you can read.

- **Private:** everything runs on your PC. Nothing is sent anywhere.
- **Works with your apps:** chat apps, coding agents and scripts that speak the OpenAI or Anthropic API just work.
- **Sees pictures** too, if you want (screenshots, photos, scanned pages).
- **Free and open source.**

> **Jump to:** [Is my PC enough?](#is-my-pc-enough) · [Install](#install-3-steps) · [How fast?](#how-fast-is-it) ·
> [Which model?](#which-model-should-i-pick) · [Using it](#using-it) · [Problems?](#something-went-wrong) ·
> [How it works](#how-does-it-work) · [All the details](docs/DETAILS.md)

---

## Is my PC enough?

| You need | |
| --- | --- |
| **Graphics card** | NVIDIA RTX 30, 40 or 50 series with **12 GB of VRAM or more** |
| **Memory (RAM)** | **64 GB** |
| **Free disk space** | ~80 GB (an SSD makes the first start much faster) |
| **System** | Windows 10/11, or Linux |

That's it. The only thing you install yourself is a current **NVIDIA driver**
([nvidia.com/drivers](https://www.nvidia.com/drivers) or the NVIDIA App). Everything else - Python, the engine, the
model - is set up for you.

## Install (3 steps)

**Windows**

1. [Download this project](https://github.com/Niko1221/Strata/archive/refs/heads/main.zip) and unzip it (or `git clone` it).
2. Double-click **`START-HERE.bat`**.
3. Answer 4 questions - or just press Enter each time for the recommended choice:
   - **Which model?** The original, or Swift 1.5 (a version that thinks shorter and answers sooner)
   - **Which size?** Q2_0, IQ2_XS, IQ3_XXS or IQ3_S - see [which model](#which-model-should-i-pick)
   - **How much context?** How much text it can keep in mind at once (it suggests one for your card)
   - **Images?** Whether it should also read pictures

Then it downloads everything (the model is ~70 GB, so the first time takes a while - you can stop and it picks up
where it left off) and **starts the model**. Your browser opens the Strata app at `http://127.0.0.1:8080`.

**Next time**, just double-click `START-HERE.bat` again: it starts right away, nothing is downloaded twice. Close its
window to stop the model.

**Linux:** run `./setup.sh` - same questions, same result.

## How fast is it?

Measured on an RTX 5070 (12 GB), a Ryzen 5 7600 and 64 GB of RAM:

| Size | Writes answers (short chat) | Writes answers (128K context) | Reads your prompt |
| --- | ---: | ---: | ---: |
| **Q2_0** | 95 tokens/s | 65 tokens/s | 539 tokens/s |
| **IQ2_XS** | 78 tokens/s | 52 tokens/s | 463 tokens/s |
| **IQ3_XXS** | 66 tokens/s | 45 tokens/s | 410 tokens/s |
| **IQ3_S** | 54 tokens/s | 42 tokens/s | 374 tokens/s |

- **Writes answers** = how fast the reply appears (tokens per second).
- **Reads your prompt** = how fast it takes in what you send (long documents, code, chat history).

A card with more VRAM is faster, because more of the model fits on the GPU: an RTX 3090 (24 GB) should do roughly
100-140 tokens per second. All measurements, long-context numbers and estimates for other cards are in the
[details](docs/DETAILS.md#speed-measured).

## Which model should I pick?

**The size** (the same model, compressed more or less):

| Size | Download | Speed | Quality | Pick it if... |
| --- | ---: | --- | --- | --- |
| **Q2_0** | 66 GB | fastest | good | you want speed |
| **IQ2_XS** | 68 GB | fast | better | you want a good all-rounder (**recommended**) |
| **IQ3_XXS** | 76 GB | slower | great | you want better answers (uses 43 GB of your 64 GB RAM) |
| **IQ3_S** | 84 GB | slowest | best: matches the full model on the published tests | you want the very best answers (original model only; uses 50 GB of your 64 GB RAM, so close other big programs) |

**The version:**

- **Qwen3.8-Flash-Next** - the original.
- **[Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)** - a fine-tune by UkisAI
  that thinks much shorter before answering, so you get the answer sooner, with about the same quality. Same speed per
  token. Its own license applies (see its page).

Not sure? Take **IQ2_XS**. You can add another one later with `START-HERE.bat --setup`.

## Using it

- **In the browser:** `http://127.0.0.1:8080` - the Strata app (it opens by itself when the model starts): **Chat**, a
  live **Monitor** of the model and your GPU/CPU/RAM, and **About** with the settings and addresses.
- **Chat in the terminal:** `.venv\Scripts\python chat.py`
- **Your apps and coding agents:** add it as an "OpenAI-compatible" provider with base URL
  **`http://127.0.0.1:8080/v1`**, any API key and any model name. Apps that use Anthropic's API: `http://127.0.0.1:8080/v1/messages`.
- **Thinking:** the model thinks before it answers. Choose **off, low, medium or high** - in the chat page menu, with
  `/think low` in `chat.py`, or with your app's "reasoning effort" setting. Off is fastest; high is best for hard questions.
- **Pictures:** in the chat page click **Picture**; in `chat.py` type `/image <path>`; in apps just attach them.
- **From your phone or another PC:** see the [details](docs/DETAILS.md#using-it) (set an API key first).

**Good to know:** it answers one request at a time. The first message of a chat is read in full (about 1 minute per
30,000 tokens); after that it keeps the conversation and reads only what is new, so follow-ups start in seconds.

## Something went wrong?

| What you see | What to do |
| --- | --- |
| `the NVIDIA driver is too old` | Update the driver (NVIDIA App or nvidia.com/drivers), restart the PC, run `START-HERE.bat` again. |
| It stopped during download or setup | Run `START-HERE.bat` again - it continues where it stopped. |
| `port 8080 is already in use` | Strata is already running - look for its window. |
| The first start takes minutes | Normal: it loads 35-43 GB into RAM. The next start is faster. |
| Slow, and the disk light is busy | Not enough free RAM: close other programs (browsers use a lot), or pick Q2_0 / IQ2_XS. |
| "prompt exceeds the context" | The conversation is longer than the context you chose: run `START-HERE.bat --setup` and pick more. |

More in the [full troubleshooting table](docs/DETAILS.md#troubleshooting). Still stuck? Open an issue and attach
`strata-<model>.log` from this folder.

## How does it work?

A model this big doesn't fit on a gaming graphics card. Strata splits the work between the parts of your PC:

<p align="center"><img src="docs/paper/tiers.svg" width="700" alt="how Strata splits the model between GPU, RAM and SSD"></p>

- **The GPU** runs the part of the model that is used for every word, plus the "experts" it needs most often.
- **The RAM** holds all 24,576 experts, and **the CPU** computes the few the GPU doesn't have - at the same time as the GPU.
- **The SSD** holds a big lookup table; the model reads a few rows of it per word.
- **A small helper inside the model guesses the next words**, and Strata checks several guesses at once. That makes
  it 1.6-1.8x faster than going word by word - and the answer is exactly the same.

The full story is in the [paper](docs/paper/Strata-Paper.pdf) and the [details](docs/DETAILS.md).

## Credits

- Model: [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team; compressed versions by
  [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF);
  [Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF) by UkisAI. Their licenses apply
  to the model files.
- Built with parts of [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) (MIT). Ideas from
  [Splash](https://github.com/incoai/splash), [ninfer](https://github.com/Neroued/ninfer) and
  [HyperQwen](https://github.com/syv-ai/HyperQwen). More in the [details](docs/DETAILS.md#credits-and-licenses).
