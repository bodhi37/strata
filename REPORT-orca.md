# REPORT: OrcaRouter Qwen3.8-Flash-Next IQ4_XS — download, port, first light (2026-09-30)

Branch: `orca-port` (7 commits on top of `r14-upstream` @ 3e66e23).
Endpoint: `http://127.0.0.1:8104/v1`, model `qwen3.8-flash-next-orca-iq4xs`, ctx 131072.
IQ3_XXS endpoint (port 8123, `strata-r23-wrapup.json`) untouched — verified no running
process was disturbed; engine binary change is backward-compatible (widens accepted
types, heretic's Q8_0/42 paths unchanged).

## 1. Download (x64) + rsync (local) — DONE, verified byte-identical

- x64 staging: `/mnt/1tbssd/ssh-downloads/aimodels/qwen38-fn-orcarouter-iq4xs/` (5 files).
- Gate was 403-GatedRepo until the user accepted the HF terms; a poll-loop retried
  until the gate cleared (curl went 403 → 302 to Xet CDN).
- `hf_xet` could not install on x64 (PEP 668 externally-managed env, no venv);
  download ran on system `huggingface_hub 1.2.2` via HTTP fallback at ~90–115 MB/s.
  Killed a duplicate downloader + stale poller that were racing the keeper process.
- Local: `~/models/qwen3.8-flash-next-orcarouter-uncensored-iq4xs/`, two-stage rsync
  (stage1: shard3+MTP+mmproj; stage2: shards 1+2) at ~30 MB/s over Tailscale.

| file | bytes |
|---|---|
| IQ4_XS-00001-of-00003.gguf | 44,766,155,936 (+445,644,800 appended by consolidate_straddle → 45,211,800,736) |
| IQ4_XS-00002-of-00003.gguf | 44,735,995,008 |
| IQ4_XS-00003-of-00003.gguf | 7,971,004,256 (+471,859,200 appended → 8,442,863,456) |
| MTP-draft.gguf | 4,135,893,632 |
| mmproj F16 | 907,543,296 |
| **total** | **~97 GiB (~9 GB smaller than heretic IQ3_XXS 5-shard)** |

- Shards hardlinked into `~/models/strata/model/` (nlink=2, no extra disk).
- NOTE: consolidate_straddle appended straddler tensors to shards 1+3, so local
  copies no longer byte-match x64. Same procedure as heretic pipeline. Do NOT
  re-rsync over them.

## 2. Model layout (probed)

- `general.name = Qwen3.8 Flash Next Abliterated`, arch `qwen4exp`, 48 blocks,
  512 experts (top-10 routed), embed 2560, **native ctx 262144**.
- 3-shard split: SH1 = metadata + token_embd + output(Q6_K) + early layers +
  PLE table; SH2 = layers ~10–42 (804 tensors); SH3 = tail layers 42–47.
  Layer 10 straddles SH1/SH2, layer 42 straddles SH2/SH3 (fixed by
  `consolidate_straddle.py` → `model/layer-experts-override-orca-iq4xs.json`).
- Expert quants: gate/up **IQ4_XS** (GPU-only path: CUDA `native_iq4_xs_mmvq`,
  zero CPU support), down **IQ4_NL**. Dense mmvq supports both.
- PLE table `per_layer_token_embd.weight` [160, 320001536] is **factory IQ4_NL**
  inside SH1 (same 320,001,536-row geometry as heretic's requanted table).
- PLE key `blk.1.ple_key.weight` [2560, 10240] is **IQ4_XS** (heretic: Q8_0).

## 3. OOM root causes found + fixed (the "keeps OOMing" bugs)

1. `tools/make_native_head.py` did `Path.read_bytes()` on the full 42 GB shard
   to copy a 521 MB tensor. → Rewrote as seek + 64 MB chunked stream copy.
2. `tools/probe_gguf.py::parse()` did `path.read_bytes()` on the whole file. →
   Header-only parse with a growing buffer (16 MB start, doubling to max filesize).
3. `finish_orca.sh` used `g.tensors.keys()` / `in g.tensors`, but gguf_reader
   exposes `tensors` as a list of TensorInfo → silently-wrong shard detection
   (would have pointed --native at a shard without token_embd). Fixed to
   `{t.name for t in g.tensors}`.
- `iq_pack.py`, `make_ple_iq4nl.py`, `repair_heretic_pack.py` already use
  `np.memmap` — audited safe.

## 4. Port artifacts (all built)

- `ple/head-native-orca-iq4xs.gguf` (521,476,096 B, byte-exact): arch guard keys
  + output.weight, both from SH1.
- `packs/orca-iq4xs/`: `index.txt` (1079 rows), `native_experts.txt` (48 layers,
  gu=23/down=20, 65.4 GB served by mmap offsets — no copy), `dense.bin`
  (1.38 GiB canonical BF16 after repair), `tokenizer/` (vocab 248320).
- `packs/orca-iq4xs/experts-native.bin` (65.4 GB, materialized on first launch
  at 0.33 GiB/s — one-time cost, now on disk).
- `strata-orca-iq4xs.json` (committed, force-added over gitignore like older
  winning configs): port 8104, ctx 131072, expert-cache 1500, hot-ram 24.0,
  mmap-experts, pool-workers 20, prefill 16384, spec 4 / min-p 0.8, MTP,
  kv q4_0, suffix-draft 3, prompt-cache 2, STRATA_RSPLIT=1.
- PLE decision: `--ple-gguf` points at **Orca SH1 itself** (engine loader
  supports multi-tensor shards; table verified identical geometry to heretic's).
  No 28.8 GB copy, no cross-model contamination.

## 5. Repair gap closed: IQ4_XS + IQ4_NL → BF16 dequant

`repair_heretic_pack.py` only knew Q8_0→BF16 and aborted on Orca
(`output_hc_down.weight is IQ4_XS`), leaving all **363 canonical-contract
tensors** (288 hc_* + 72 ssm_* + output_hc_* + ple_value) unserved.
Added vectorized numpy dequants transcribed from ggml `dequantize_row_iq4_xs`
+ `dq_iq4_xs` (CUDA kernel) with the kvalues_iq4nl codebook and RNE BF16
rounding shared with the Q8_0 path. Unit-verified (all-ones synthetic blocks
decode to exactly 1.0, correct shapes). Full run: 266×IQ4_XS + 97×IQ4_NL
repaired, 0 errors, idempotent on re-run (363 already canonical).

## 6. Engine changes (2 commits, both backward-compatible)

1. `src/program/generate.cpp`: native PLE key check widened from hardcoded
   `{42, 8}` to `native_mmvq_supported()` — the callee was already type-generic.
2. `src/kernels/cuda/ple.cu` (`ple_block` guard): same widening (was the
   request-time `"native key requires Q2_0/Q8_0"` error).
- Rebuilt via `build-engine.sh` (sm_89, CUDA 13.3), engine/strata refreshed.

## 7. First light (2026-09-30 ~18:4x +10:00)

- `bash srv.sh strata-orca-iq4xs.json 8104` → UP in 70 s.
- `READY 2+2=4` prompt → `finish: stop`, exact content match, 40 tokens.
- Engine-reported: req1 prefill 7.6 tok/s cold / decode 7.8 tok/s;
  req2 (prefix reuse) prefill 11.9 / **decode 14.7 tok/s**, MTP 27/30 (90%).
- Already above IQ3_XXS median (~9) on request 2 with zero tuning.
- Baseline bench `scratch/r11_decode_bench.py 8104 200 warm` launched →
  `logs/bench-orca-baseline.log` (pending at time of writing).

## 8. Next: hillclimb plan (not yet started)

1. Read baseline bench; confirm coherence on all 6 topics (no `!!!` garbage).
2. First easy win: Orca is ~9 GB smaller → push `--hot-ram-gib` / expert-cache
   coverage past heretic's 97.2% toward ≥99% (the serial-QD1 latency wall analysis
   in PROMPT-hillclimb-inference.md).
3. Knobs in order: expert-cache 1500→3000+, pool-workers, pcie-frac, kv q4_0 +
   kv-resident window, spec/suffix-draft, prefill admit, park-spin, profile
   refresh from Orca routing traces (`--dump-counts` → `make_profile_from_counts`).
4. 128k longctx bench (`r14_longctx_bench.py`), then 256k (native ctx, YaRN if needed).
5. Pi harness: add `qwen3.8-flash-next-orca-iq4xs` on :8104 to `~/.pi/agent/models.json`
   (131072 ctx, maxTokens 131072); verify 8123 entry untouched.
6. Targets: EASY ≥20–30 tok/s decode / 500–1000 prefill; STRETCH 50–80 / 1000–2000.

## 9. Hillclimb R1 (2026-09-30 ~19:00 +10:00) — baseline + first blood

Branch `orca-port` + 1 commit (kernel) + uncommitted `expert_cache.cpp` fix / static config
at time of writing. Engine was found DEAD on arrival (zombie `engine/strata`, server
`RuntimeError: the engine process ended` after 4 decode runs of the pending
`bench-orca-baseline.log`); no OOM in dmesg/journal — likely segfault/assert on the 5th
request. Restarted clean via `guard-launch.sh`.

### 9.1 Baseline (config `strata-orca-iq4xs.json` = r19-best tuning on Orca)

Coherence: clean on SHORT/MEDIUM/LONG-PREFILL/LONG-TASK (correct answers, no `!!!`).

| bench | result |
|---|---|
| decode 200 tok × 6 topics (client wall, incl ~7–11 s prefill each) | cooking 7.57, coding 6.08, law 6.16, medicine 2.53, math 3.51, agentic 5.85 → **min 2.53 / median 6.08 / max 7.57** |
| engine decode-only (warm short topics) | 8.7–12.2 tok/s; 14.7 with prefix reuse |
| prefill 10.8k × 6 unique (`pf_multi.py`) | 309 / 325 / 146 / 94 / 248 / 125 → **min 94 / median 197 / max 325** |
| prefill 6.4k (coherence LONG) | 208–225 tok/s |
| short-prompt prefill (60–70 tok) | 6–11 tok/s (7–12 s fixed cost per request!) |

Engine anatomy of a warm 200-tok decode: 245–382 expert reqs/token, 91–95% from RAM,
25–57 DISK blobs/token × 2.66 MB = 66–152 MB/token at ~1 GB/s; ringwait 185–730 ms/window;
pool drain 33–137 ms/window with **gu 22–88 ms (IQ4_XS, ggml fallback) vs down 11–48 ms
(IQ4_NL, AVX-512)**; MTP 1.6–3.0 tok/window. Prefill borrows 1068 cache slots —
chunk 16384 is real (no halving message).

### 9.2 The LRU-admit pathology (the run-to-run variance explained)

The 6.4k-token coherence LONG runs stream ~54k blobs through the 9679-slot LRU tier
(57k admitted/evicted) with `STRATA_PREFILL_ADMIT` default-ON + dynamic tier (no
`STRATA_STATIC_TIER`). Consequences measured in this session:
- decode after LONG prefill: ringwait 1440–1810 ms/window (vs 185–460 warm);
- prefill samples degrade across the run: 309 → 325 → 146 → 94 → 248 → 125
  (each 10.8k sweep evicts the profile hot set; the next sample re-reads it);
- decode topic spread 2.53–7.57 (each topic's working set evicts the last).
REPORT §0.8/0.9 saw the same on heretic (admit hurts short-mix, helps long-conv).
Fix under test: `strata-orca-static.json` = static tier + admit OFF.

### 9.3 Correction to §8 item 2

Orca IQ4_XS is ~11 GB *larger* than heretic IQ3_XXS (arena 61 GB vs 49.8 GB, blob
2.66 MB vs 2.18 MB), not smaller — coverage at equal tier size is *worse*
(9679 blobs = 39% of 24576 experts vs 12k = 49% on heretic). The ≥99% gate needs
~15–16k resident blobs ≈ 40+ GiB; resident ceiling is ~11.2k (9679 RAM + 1500 VRAM).
More RAM is the only coverage lever; everything else is latency-hiding.

### 9.4 Code changes (this session)

1. **IQ4_XS decode-once CPU kernel** (`iq_avx512.cpp`, `iq_avx512.hpp`): Orca's gate/up
   (type 23, Q8_K acts) had no AVX-512 path — every verify token re-decoded the nibble
   LUT via ggml AVX2. New `row_dot_iq4xs`/`gu_rows_iq4xs`/`iq4xs_gu_rows_nt` decodes each
   32-value sub-block once across nt tokens (same structure as R10's IQ4_NL kernel).
   `scratch/iq4xs_fuzz.cpp`: **200 trials × nt 1..8, bitwise-identical to ggml** (max
   rel/abs 0.0). `native_expert_parity` on Orca layers 0–5: cpu rel 1.2–1.4e-2
   (activation-noise class, same as heretic), 0 failures. Wired via `iq512_supported(23)`
   for the gu path only (nt ≥ 2; single-token keeps ggml, which ties at nt = 1).
2. **Per-layer VRAM fill**: default fill piles slots by global rank (~3% GPU hits per
   R4.2g); `--expert-cache-per-layer` (31 slots/layer at 1500) added to static config,
   plus two latent-bug fixes: `open_sized` never initialized `layer_next_` per-layer
   bases (all layers admitted from slot 0 → `verify_slot` failed at byte 0), and the
   fill loop `break`→`continue` under per-layer quotas (shared-mode `break` preserved).
3. **Static config** (`strata-orca-static.json`, force-added): static tier, admit OFF,
   per-layer VRAM cache, `--dump-counts data/counts-orca-r1.strc` for an Orca-native
   profile refresh. Separate log `logs/strata-orca-static.log`.

## 10. Hillclimb R2 — static + per-layer A/B (2026-09-30 ~19:20–19:40)

Config `strata-orca-static.json` (static host tier, admit OFF, per-layer VRAM 1341/1500
filled, IQ4_XS kernel, dump-counts). Same bench sequence. All coherent.

| bench | static-per-layer | baseline (dynamic) |
|---|---|---|
| coherence | clean (SHORT 18.8 s, MEDIUM 97.9 s, LONG-P 44.5 s, LONG-T 90.5 s) | clean (16.4 / 32.4 / 38.3 / 43.3 s) |
| decode 200 tok | 3.97/3.51/2.02/3.87/2.38/5.78 → **2.02/3.87/5.78** | **2.53/6.08/7.57** |
| prefill 10.8k ×6 | 181/287/188/290/188/111 → **111/188/290** | **94/197/325** |

Verdict: per-layer VRAM fill is a **regression** here (−36% decode median, −5% prefill
median). Cause: the shipped fill is already profile-ranked (top slice), not arrival-order —
R4.2g's 3%-hit figure was arrival-order; per-layer trades 159 globally-hot pairs for
deeper-rank ones (1341/1500 filled) and loses total coverage. Static tier did stabilize
prefill (no progressive 309→94 decay; bimodal 185/290 pattern instead), but fresh
long-prompts still sweep ~3 blobs/token. Profile overlap measured: r14-heretic top-9679
∩ Orca top-9679 = **98%** (router untouched by abliteration) — no profile-refresh win
available; SKIP Orca-native profile. Next: static-SHARED (isolate fill policy).


## 11. Hillclimb R3 — defrag (the 3x drive win, 2026-09-30 ~19:55)

`qdbench` on the Orca arena: **376 MB/s, 7.4 ms/read QD1** vs R22's 816 on heretic.
Root cause: **77,075 btrfs extents + `compress=zstd:3` mount option** strangling every
2.66 MB blob into ~3.5 seeks + decompress attempts. Fix: `btrfs property set ...
compression none` + `btrfs filesystem defragment -f` → **14,654 extents** →
**1115 MB/s QD1 (2.5 ms/read), 1656 MB/s QD8**. Same running engine, immediate re-bench:

| bench | pre-defrag (static-shared) | post-defrag |
|---|---|---|
| decode 200 tok | 3.26 / 4.56 / 5.73 | **8.05 / 9.13 / 11.16** (+100% median, tight spread) |
| prefill 10.8k x6 | 111 / 188 / 290 | **487 / 519 / 521** (+176% median, rock-stable) |

**Prefill 500 tok/s TARGET MET (median 519).** Lesson: the "drive ceiling" in REPORT
§0.8 was measuring filesystem fragmentation, not silicon — 1.1 GB/s QD1 random is real
now (1656 MB/s QD8). Decode now pool-bound (gu 207 ms/window at nt≈1 fallback; MTP 1.6
tok/window) — next: pool-workers, spec, cache size.

## 12. Hillclimb R4 — knob A/B (pool-workers, spec, cache size; 2026-09-30 ~20:05–20:30)

All on static-shared + defrag + kernel base (decode 200 tok: 8.05/9.13/11.16):

| knob | decode 200 tok | verdict |
|---|---|---|
| pool-workers 20 → 12 | 3.84 / 4.47 / 6.62 | **REJECT**: parallelism wins over contention (−51% median) |
| spec 4 → 6 | 3.08 / 6.53 / 7.63 | **REJECT**: bigger windows fetch more miss bytes without acceptance gain (−28%) |
| expert-cache 1500 → 1600 | (bench running) | boots, 1574 MiB VRAM free (vs 1828); borrow unchanged (1068 slots) |
