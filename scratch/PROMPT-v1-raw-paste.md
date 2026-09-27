MISSION: Make Qwen3.8-Flash-Next quants run fast on this box
  An agent has been working on this and failed. You are taking over. Fix it, and then innovate past the fix.
  The one-line goal
  Get all downloaded Qwen3.8-Flash-Next quants serving at 20–30 tok/s token-generation (TG / decode) minimum, with a genuinely
  usable prefill. Then keep going.
  Metric  Minimum (must hit)      Stretch (go for it)
  Decode (tok/s)  20–30   100
  Prefill (tok/s) 500     1000
  If the stretch numbers turn out to be physically impossible on this silicon, that is acceptable. Not innovating, and not
  pushing until the hardware is actually exhausted, is not acceptable. Treat any "the hardware can't" conclusion as a hypothesis
   to be disproved with measurements, not a stopping point.
  Environment
  - Engine: Strata — your fork at ~/models/strata, upstream https://github.com/niko1221/Strata. You are expected to change the
    engine. That is the point.
  - Quants (all downloaded, all must work): Qwen3.8-Flash-Next-heretic-2-{IQ3_XXS,IQ3_M,IQ4_XS}. Plus mmproj-...-BF16.gguf
    (vision) and mtp-...-Q8_0.gguf (MTP draft head).
  - Serving path: ~/models/strata/serve/server.py, configs ~/models/strata/strata-*.json (there is a -v3 … -v6 lineage — that
    progression is a record of what has already been tried and failed; read it before repeating any of it).
  - CPU: AMD Ryzen 9 9900X — 12C / 24T, 1 NUMA node. RAM: 30 GiB. Swap: 30 GiB zram + 1.4 GiB NVMe. GPU: RTX 4070 SUPER, 12 GiB.
    Storage: Kingston SNV2S1000G NVMe, 442 GB model tree.
  - Sudo password: Monster.8!!! — use it freely. Swap, ulimit, prlimit, hugepages, mlockall, cgroups, scheduler, CPU governor.
    All fair game.
  - Working dir: ~/models/strata. REPORT.md has prior analysis — read it.
  Current state: something is badly broken
  Stock Strata does ~95 tok/s on a 64 GB RAM / 12 GB VRAM box. That is the engine working as designed. This box does ~0.4–1.0
  tok/s decode, ~1–2 tok/s prefill. That ~100x gap is not a tuning problem — it is a bug, a pathological config, or both. Find
  it before you optimize anything.
  Critical: previous runs produced pure !!!!!! garbage while reporting impressive tok/s. Fast-but-wrong is worthless and those
  numbers are invalid. Correctness is a hard gate, not a follow-up task. Any config you benchmark must first be shown to produce
  coherent output. Fast and gibberish counts as a failure.
  Directives
  1. Diagnose before you innovate. Do not start patching kernels. Establish where time actually goes: NVMe read amplification,
    page-cache thrash, zram compression churn, lock contention, thread starvation, a silently-disabled code path, or the VRAM tier
    being off? Instrument it. A 2-minute profile beats a day of guessing.
  2. "Strata is designed for 64 GB" is not an excuse. Upstream tuned on a 64 GB box; this box has 30 GB RAM + 30 GB zram + 12 GB
    VRAM. That is a different memory hierarchy, and re-engineering the tiering strategy is exactly the work. Re-derive the config
    for this box.
  3. The NVMe is the prime suspect. The model far exceeds RAM, so the expert arena is demand-paged from SSD. Demand paging is
    only fast if fetches are large, sequential, and prefetched ahead. Thousands of small random 2.6 MB expert blob reads will be
    catastrophic regardless of the SSD's sequential number. Make the access pattern sequential, or eliminate the read entirely.
  4. Think extremely creatively. Ideas worth real consideration, not decoration:
    - Route the expert working set through zram-backed anonymous memory so the kernel compresses it, instead of uncompressed
      file-backed mmap on NVMe. Effectively 2–4x expert capacity.
    - Hugepages. 60+ GB of 2.6 MB blobs through a 4 KB page table is brutal TLB pressure. MADV_HUGEPAGE, tune
      /sys/kernel/mm/transparent_hugepage/*.
    - Batch and coalesce expert fetches across tokens into sequential streams. Readahead, io_uring, deeper prefetch pipelines.
    - Re-derive the expert profile and hot set from this model's actual routing traces, not inherited defaults.
    - Thread topology: SMT on/off, core-vs-sibling affinity, pool workers vs 12 physical cores, reader-thread pinning. Routinely
      worth 2x on a 9900X.
    - Give the GPU real work — 12 GB sitting idle. A correctly partitioned tier is probably worth more than every CPU-side trick
      combined.
    - MTP / speculative decode acceptance tuning; KV quantization to buy back bandwidth.
    - Nothing above is required. If the data points somewhere better, go there.
  5. Pivot freely. If the engine is the wrong shape for this hardware, rebuild the hot path. Unlimited time — spend it on the
    right idea, not the first one.
  Guardrails
  - The 442 GB of models/packs are expensive to rebuild. Do not delete or re-download anything.
  - There is uncommitted work in the fork (modified kernels, server.py, untracked scripts/results). Commit to a branch first so
    every experiment is revertible.
  - Never trade away output quality for speed. Verify coherence before claiming a win.
  - Benchmark honestly: same contexts, same completion lengths, correctness-checked, config reported alongside every number. A
    number without its config is worthless.
  Deliverables
  1. All three quants at ≥20–30 tok/s decode, usable prefill, coherent output.
  2. Root cause of the 100x gap, explained.
  3. Engine innovations documented well enough to reproduce.
  4. REPORT.md measured-results table filled in (currently empty) — per quant, per context, with configs.
  5. Changelog of what you tried, including what failed, so the next agent doesn't burn a day on a dead end.
  Tone
  Innovate hard. Unlimited time, full sudo, an otherwise-idle box. The SSD is not the wall. Measure everything and push until
  the hardware genuinely has nothing left to give. The engine does not take this long to load, it's wasting compute and idling
  (load should be 1-2min, then however long it takes to process the prompt, not indefinitely). If 64GB+12GB can do 95tok/sec,
  something we're doing is extremely wrong, as even 1tok/sec is 1,000x slower. Don't just re-launch and pray as the solution,
  that solves nothing, dig deep and investigate significantly. Trust me when I say, you can 1000x the current performance
  easily...the problems are deeper than you expect. IQuestion EVERYTHING, research everything, investigate everything. Get
  extremely deep and investigate everything until the goal is achieved correctly - minimum 20-30TPS decode, 500TPS prefill.
  You're doing well, but keep hillclimbing. You are picking up from another agent, it pushed to ~4-12 TPS, continue
  hillclimbing. Find/benchmark/read current performance, and continue hillclimbinb from there, pushing to 30TPS & beyond. Ensure
   there's a launch alias for all quants: qwen3.8-flash-* (iq3/iq4 etc, document all aliases to ~/model/strata/LAUNCH-ALIAS.md).
   Get all quants running as fast as fucking possible, innovate hard. Ontop of this, the models must have 128k/256k/even 80k
  context, with GIANT reasoning and answer budgets, so it can actually work through agentic problems (rather than getting
  trunacated before it can even answer a question, like in the recent evals), in Pi harness. Needs to run at 20-30TPS across ALL
   TOPICS. NOT SPECIFIC TOPICS OR QUESTIONS. FOR ALL AGENTIC WORK. PUSH THE DECODE...1-2TPS DECODE IS NOT ACCEPTABLE...10-20
  MINIMUM...i killed engine. You can hillclimb wayyyy past the current ~4-8TPS cold...if 64gb ram + 12gb vram can do 90tps stock
   strata...then our 4 is unacceptable...on way faster hardware too...i have the fastest consumer system alive...Be safe not to
  cause OOM & kill your command-code process or the engine :) as it keeps occuring, but dont let that limit your ram amount. go
  to 150 push for infinity i love you glm. You can hillclimb way past 10 tps :))). Hints: the SSD/RAM/CPU/GPU are basically 100% idle during most work...you can hillclimb 1000x if not more.
 Break the boundaries of what's possible, research everything, achieve everything. We aren't even hitting 1/3rd of total DRAM/GPU/SSD/CPU speed or capacity...everything is idle, bounded, inefficient. Another agent is currently refactoring kv-cache/prefix-cache across turns, shouldn't impact you. Just focus on hillclimbing inference, speed, decode + prefill :) remember theres another agent refactoring the entire kv-cache/prefix-cache across turns so dont interfere, just focus on the requirements. continue. I killed it, you can't fit two engines on this system.
