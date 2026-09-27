# Task: Seamless KV-cache & prefix-cache reuse across turns in Strata

You are working in `~/models/strata` — a custom CUDA inference engine (fork of
nikko1221/strata) serving a hybrid Qwen3.8-Flash-Next architecture (36 GDN
linear-attention layers + 12 QSA full-attention layers, 512-expert MoE, a
SSD-resident PLE n-gram module, and an MTP draft head for speculative decode).
It exposes OpenAI/Anthropic-compatible HTTP APIs from `serve/server.py`, which
talks over stdin/stdout to a resident `engine/strata` binary built from `src/`.

## The mission

The model and inference are correct and fast. There is exactly one problem
left, and it is your only focus: **the engine drops all state between turns
and re-prefills the entire conversation on every request.** At 131K context
this is ~1 minute per 30K tokens before the first token of an answer that
differs from the previous one by a handful of tokens. Without carried cache,
long-horizon and agentic workloads are not viable on this engine.

Your goal: make conversation state persist across turns so that each turn
prefills **only the new tokens**, and any reuse decision is provably correct —
the cache may affect speed, never output correctness. A user with a standard
OpenAI client (which resends full history every turn) should experience
seamless multi-turn conversation where turn N's prefill is O(new tokens), and
a non-continuing request (system prompt change, edited history, new
conversation, error recovery) must still work — degrading to full prefill is
acceptable on a miss; a corrupted or misaligned session is not.

## Scope discipline: there is a parallel agent doing hillclimbing work

Another agent is concurrently tuning inference throughput (prefill chunking
knobs, expert-cache tiers, `STRATA_*` sweep env vars, the `strata-r*-*.json`
bench configs, `hot_sweep.sh`, decode perf). **Do not touch, tune, revert, or
"fix" any of that.** Stay strictly on cache-across-turns:

- Do not modify prefill performance constants, expert-tier behavior, pool
  workers, mlock setup, or any sweep/bench configuration.
- You will inevitably work in files the other agent also touches
  (`src/prefill/`, `src/program/generate.cpp`). Keep your edits there minimal,
  semantically orthogonal, and structured so they can't conflict with knob
  tuning — e.g. new modules/files where feasible, additive protocol fields
  rather than rewrites of hot paths.
- Never run or interpret the perf sweep benchmarks as your success metric.
  Your metric is cache correctness and hit behavior, not tok/s.

## Phase 1 — understand the system deeply before designing anything

Study these and be able to explain each one before writing any code:

1. **The existing attempt ("plan v0.3 P8b").** There is a fragile
   cross-turn reuse mechanism already: the server in `serve/server.py`
   (the `Service` class: `asm_ids`/`asm_valid`/`last_render` state, the
   seam-detection in `prepare()`, the session bookkeeping in `run()`) sends
   `GEN <max_new> KEEP <n> <ids>` and the engine in `src/program/generate.cpp`
   (the KEEP parse, reuse gate, suffix prefill, `sess_len` reporting in the
   `DONE` line) skips `session_zero` and prefills only the suffix when
   `keep_n == sess_len`. Understand exactly how it works end to end, and then
   **audit it skeptically**: production logs (`logs/r10-iq3xxs.log`) show the
   cumulative prefill counter growing by the full prompt length every request —
   it has essentially never engaged in practice. Find out why, and audit
   whether the fast path, if it did engage, would produce a session whose
   internal state is *bit-identical* to a full clean prefill of the same
   canonical token stream. Pay special attention to:
   - the boundary between tokens the engine generated last turn (including
     the stop token) and tokens the server re-encodes from the new render —
     who owns that seam token, and does the assembled token stream equal the
     canonical chat-template render exactly?
   - what the engine's committed live state actually is at end-of-generation
     (trace what `Verifier::commit` commits in `src/core/verify.cpp` vs. what
     `sess_len` claims in `generate.cpp`) versus what the server assumes.
   - whether the engine verifies *content* identity of the kept prefix or only
     a length, and what happens if the server's and engine's views diverge.
   - the mock engine path (`--engine mock`) and the `Engine` protocol in
     server.py — does the current machinery still support it?
2. **Session state lifecycle.** `src/core/session.cpp`, `src/core/layer.cpp`
   (QSA state allocation, `qsa_state_zero`), `src/core/verify.cpp`
   (window commit, snapshots of indexer tail / PLE history), and
   `session_zero` in `generate.cpp`. Know exactly what state exists: 12 QSA
   paged KV pools (+ page tables, indexer state, `step` buffers), the 36 GDN
   recurrence/conv states, PLE history and `ple_prev`, and the MTP drafter's
   own QSA arena. Note which pieces can rewind and which are strictly
   sequential/append-only, and what invariant currently lets rejected draft
   cells be safely overwritten instead of freed.
3. **Prefill.** `src/prefill/prefill.cpp` (`Prefill::run(tokens, n, pos0)`) —
   how chunked prefill leaves state consistent with the token path, and how it
   interacts with the MTP drafter's `on_chunk` and the expert staging ring.
4. **The serving protocol.** The full line protocol (`READY`, `GEN`, `GENI`,
   `T`, `DONE`, `ERR`, `QUIT`) in `generate.cpp` + `StrataEngine` in
   `server.py`, the FIFO single-session model, and how `sess_len` is
   validated. Note there are no sequence/session ids today — the single
   resident session *is* the cache.
5. **The planning discipline.** `include/strata/plan/plan.hpp` and
   `src/plan/plan_main.cpp`: fixed startup allocation (`session_bytes`),
   the refusal-to-overcommit philosophy, the zero-mid-run-allocation rule
   (CUDA graphs bake pointers; a captured `kv_append` with a frozen `pos`
   writes every token to cell 0 — read the warning in
   `include/strata/kernels/qsa.hpp`), and the VRAM envelope (12 GB card,
   ~3.35 GiB expert tier, `--kv int8`, fixed `max_cells`).

## Phase 2 — study industry precedence

Before designing, read how mature engines solved this, and internalize *why*
each made its choices — the transferable ideas, not the code:

- **vLLM — PagedAttention + Automatic Prefix Caching.** KV blocks mapped to
  content hashes in a global hash table, so identical prefix blocks share
  physical memory across requests; copy-on-write on divergence. Key ideas:
  the cache is keyed by *token content*, not by "the previous request", and
  cache hit/miss is a block-level decision, not all-or-nothing.
  See docs.vllm.ai → "Automatic Prefix Caching" design doc.
- **SGLang — RadixAttention.** A radix tree over block hashes manages
  prefix reuse and eviction across requests; compare against vLLM's flat
  hash table and think about which mental model fits a single-sequence,
  strictly-extendable engine.
- **llama.cpp — llama-server prompt caching.** `--cache-prompt` keeps a
  slot's KV and computes the longest common *token* prefix with the next
  request to skip prefill; `--cache-reuse` extends this to
  context-shift/branching via slot save/restore (`llama_state_seq_save_file`
  / restore). Key ideas: token-ID-level prefix matching with explicit
  verification, and making reuse a first-class, testable engine capability
  rather than a server-side string heuristic.
- **The universal invariant.** In every mature design, correctness never
  depends on the cache: a hit must be verified against actual token identity,
  a miss must be indistinguishable from a cold start, and a partial hit must
  leave state exactly as a full prefill would have. Notice also that in all
  of these systems the *server* decides policy but the *engine* owns and
  validates the cache — contrast with Strata's current split, where the
  server reconstructs token identity from rendered text and the engine
  trusts a bare length.

## Phase 3 — design before code

Write your design down and check it against these questions:

- **Who owns the token stream of record?** Today the server owns a rendered
  text string and re-derives tokens from it each turn, while the engine owns
  live state with a length only. Should the canonical token stream — the one
  whose KV is actually resident — be owned and *content-verified* by the
  engine, with the server merely proposing an extension? What is the minimal
  protocol surface that lets the engine accept or reject a proposed
  continuation based on evidence it computed itself?
- **What is the unit of reuse, and what is verifiable?** The GDN/PLE state is
  strictly sequential — it cannot rewind, only extend. Given that, what does
  a "block" even mean here, and what granularity of reuse is actually
  achievable vs. what vLLM/SGLang get away with via true paging? Be honest
  about this constraint and design the verification so a mismatch anywhere
  falls back cleanly.
- **What is the true committed-length semantics at end of turn?** Trace
  exactly which positions have valid KV/indexer/GDN/PLE state when the last
  token of a generation is emitted, and make the engine's reported state and
  the server's cached token stream refer to the *same* stream, with one
  unambiguous owner for the stop token at the seam.
- **What invalidates a cache, and how does each invalidation behave?** Enumerate
  them: vision requests, engine errors, client disconnects, server restarts,
  template/history changes, context-window pressure, engine-side capacity
  (max_cells / max-context overflow). For each, the correct outcome is
  graceful degradation — decide whether stale state must be zeroed or can be
  overwritten, and make sure a reused session and a fresh prefill are
  indistinguishable in output.
- **What fits in the envelope?** Any prefix bookkeeping (hashes, token-id
  records, snapshots) must live inside the plan discipline: sized at startup,
  no mid-run allocations, graph-capturable (device-driven positions through
  fixed buffers), and it must not disturb the prefill staging-ring/expert-tier
  interaction that the other agent is tuning.
- **How do you test it?** The repo's `tests/` is a stub and the mock engine
  path is currently broken by the existing attempt — restoring a mock/test
  path that exercises the cache is in scope. Design tests that can prove
  the hard property: for a multi-turn conversation, the engine's outputs
  with cache reuse are *identical* to outputs from a from-scratch full
  prefill of the same canonical token stream (this is checkable directly —
  think about how to compare at the logit/output level, and consider what
  `bench_endpoint.py` and `scratch/multiturnbench.py` already measure).

## Phase 4 — implement, then prove it

- Implement the refactor in coherent, reviewable pieces. Prefer making reuse
  *unconditional, automatic, and verified* over adding opt-in knobs: mature
  engines do prefix caching by default and simply fall back on miss.
- Instrument honestly: the engine should report (in `DONE` or a log line)
  tokens reused vs. tokens prefilled, and whether the cache engaged —
  the current failure mode (a feature that silently never engages) is the
  thing to eliminate, so hit/miss must be observable at a glance.
- Validate end-to-end: multi-turn conversations through the real OpenAI
  endpoint must show near-zero prefill on continuation turns and identical
  outputs vs. full re-prefill; misses (edited history, new conversation,
  vision, errors) must be correct and transparent; the mock engine path and
  any unit tests you add must pass; `scratch/multiturnbench.py` is the
  quickest harness for the headline behavior.
- Leave the hillclimb agent's surface untouched. If a shared file must
  change, keep the diff additive and explain the boundary in your final
  report.

## Definition of done

1. Turn 2+ of a standard multi-turn conversation prefills only genuinely new
   tokens (verified in logs/metrics, not assumed).
2. Output equivalence: reused-session outputs are bit-for-bit (or argued to
   be, at the state level) identical to full-prefill outputs for the same
   canonical token stream.
3. Every miss path degrades gracefully to a correct full prefill; nothing
   can produce a session whose state disagrees with the token stream it
   claims to hold.
4. The cache engages reliably in practice — observable reuse/prefill counters
   confirm it — including through the plain `chat.py` / OpenAI-client path.
5. Mock engine and tests restored/passing; no perf-knob files touched.
