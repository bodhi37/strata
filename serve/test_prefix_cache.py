#!/usr/bin/env python3
"""plan v0.4 P8c: unit tests for the server side of the cross-turn prefix cache.

Runs the real `Service` (prepare/run seam assembly, cache bookkeeping, invalidation) against
`MockEngine` and the `ByteTokenizer` test stand-in - no GPU, no model pack.  Self-asserting:

    python3 serve/test_prefix_cache.py
"""
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from server import ByteTokenizer, IM_END, MockEngine, Service                      # noqa: E402
from frontend import ChatTemplate                                                  # noqa: E402

# The template re-normalizes the assistant block exactly like the model's own: reasoning is
# rendered inside a '<think>\\n\\n...\\n\\n</think>' wrapper even when the RAW generation had
# different whitespace (or none) - the drift the seam assembly must skip over.
TEMPLATE = (
    "{%- for m in messages -%}"
    "{%- if m.role == 'system' -%}<|im_start|>system\n{{ m.content }}<|im_end|>"
    "{%- elif m.role == 'user' -%}{{ '\\n' }}<|im_start|>user\n{{ m.content }}<|im_end|>"
    "{%- elif m.role == 'assistant' -%}{{ '\\n' }}<|im_start|>assistant\n<think>\n\n{{ m.reasoning_content or '' }}"
    "\n\n</think>\n\n{{ m.content }}<|im_end|>"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{ '\\n' }}<|im_start|>assistant{{ '\\n' }}{%- endif -%}"
)

SYS = "You are Strata."
U1 = "What is 2+2?"
U2 = "And 3+3?"
U3 = "Thanks."


def make_service(script="Sure.", max_new_default=64):
    tok = ByteTokenizer()
    with tempfile.NamedTemporaryFile("w", suffix=".jinja", delete=False) as f:
        f.write(TEMPLATE)
        tpl_path = f.name
    svc = Service(MockEngine(tok, script, max_context=32768), tok, ChatTemplate(tpl_path))
    svc.default_max_new = max_new_default
    return svc


def turn(svc, messages, max_new=64, cancel=None):
    ids, thinking, keep, render = svc.prepare(messages, None, {}, max_new)
    out = []
    for kind, ev in svc.run(ids, thinking, None, max_new, {}, cancel or threading.Event(),
                            keep=keep, render_text=render):
        out.append(ev)
    return ids, out, keep


def assistant_msg(text, reasoning=""):
    m = {"role": "assistant", "content": text}
    if reasoning:
        m["reasoning_content"] = reasoning
    return m


def last_text(events):
    texts = [e.text for e in events if hasattr(e, "text") and getattr(e, "kind", "") == "content"]
    return "".join(texts)


def test_turn1_full_prefill():
    svc = make_service()
    ids, _, keep = turn(svc, [{"role": "system", "content": SYS}, {"role": "user", "content": U1}])
    assert keep == 0
    assert svc.asm_valid and svc.asm_ids
    assert svc.asm_ids[:len(ids)] == ids                       # the cached stream starts with the prompt
    assert svc.asm_ids[-1] == svc.tok.encode(IM_END, parse_special=True)[0]   # ...and ends with the stop token
    print("ok: turn 1 is a full prefill and caches prompt + emitted (stop token included)")


def test_turn2_reuses_and_seam_is_clean():
    svc = make_service()
    _, _, _ = turn(svc, [{"role": "system", "content": SYS}, {"role": "user", "content": U1}])
    asm_after_1, keep_expected = list(svc.asm_ids), len(svc.asm_ids)
    ids2, _, keep2 = turn(svc, [{"role": "system", "content": SYS}, {"role": "user", "content": U1},
                         assistant_msg("Four."), {"role": "user", "content": U2}])
    assert keep2 == keep_expected, "turn 2 must propose the whole cached stream"
    assert ids2[:keep_expected] == asm_after_1, "the assembled stream must extend the cached record verbatim"
    # the seam: the last cached token is the previous reply's <|im_end|>, and the suffix starts
    # with the canonical "\n<|im_start|>" continuation - no duplicated stop token
    im_end = svc.tok.encode(IM_END, parse_special=True)[0]
    assert ids2[keep_expected - 1] == im_end
    assert ids2[keep_expected] == 10, "the suffix must start with the newline after the closer"
    assert ids2[keep_expected + 1] == svc.tok.encode("<|im_start|>", parse_special=True)[0]
    tail = svc.tok.encode("\n<|im_start|>user\n" + U2 + "<|im_end|>\n<|im_start|>assistant\n", parse_special=True)
    assert ids2[keep_expected:] == tail, "the suffix must be the canonical continuation after the closer"
    eng = svc.engine
    assert eng.last_keep == keep_expected, "the engine must receive the keep proposal"
    assert eng.last_prompt == ids2
    assert eng.last["sess"] == len(ids2) + eng.last["generated"]
    print("ok: turn 2 proposes keep = cached length; the seam has no duplicated stop token")


def test_three_turn_chain():
    svc = make_service()
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": U1}]
    _, _, _ = turn(svc, msgs)                                   # turn 1: full prefill
    msgs = msgs + [assistant_msg("Answer 0."), {"role": "user", "content": U2}]
    ids2, _, keep2 = turn(svc, msgs)                            # turn 2: extends turn 1
    assert keep2 > 0 and ids2[:keep2] == list(svc.asm_ids[:keep2])
    asm_after_2 = list(svc.asm_ids)                             # turn 2's full cached stream
    msgs = msgs + [assistant_msg("Answer 1."), {"role": "user", "content": U3}]
    ids3, _, keep3 = turn(svc, msgs)                            # turn 3: extends turn 2
    assert keep3 == len(asm_after_2), "turn 3 must carry turn 2's full stream"
    assert ids3[:keep3] == asm_after_2
    im_end = svc.tok.encode(IM_END, parse_special=True)[0]
    # blocks: system + user1 + asst1 + user2 + asst2 + user3 = 6 closers
    assert ids3.count(im_end) == 6
    print("ok: turn 3 chains on turn 2's cache with a clean seam")


def test_new_conversation_is_a_miss():
    svc = make_service()
    _, _, _ = turn(svc, [{"role": "system", "content": SYS}, {"role": "user", "content": U1}])
    ids, _, keep = turn(svc, [{"role": "system", "content": SYS}, {"role": "user", "content": "A different question"}])
    assert keep == 0
    print("ok: a new conversation falls back to a full prefill")


def test_edited_history_is_a_miss():
    svc = make_service()
    _, _, _ = turn(svc, [{"role": "system", "content": SYS}, {"role": "user", "content": U1}])
    before = list(svc.asm_ids)
    ids, _, keep = turn(svc, [{"role": "system", "content": SYS}, {"role": "user", "content": U1 + " (edited)"}])
    assert keep == 0
    assert ids != before
    print("ok: edited history falls back to a full prefill")


def test_tool_json_drift_still_hits():
    """The old +/-32 size heuristic missed on tool-call JSON re-serialization drift; the new seam
    does not measure anything, so the hit survives arbitrary drift inside the assistant block."""
    svc = make_service()
    _, _, _ = turn(svc, [{"role": "system", "content": SYS}, {"role": "user", "content": U1}])
    asm = list(svc.asm_ids)
    big = assistant_msg("x" * 500)                    # render far longer than the raw generation
    ids, _, keep = turn(svc, [{"role": "system", "content": SYS}, {"role": "user", "content": U1},
                              big, {"role": "user", "content": U2}])
    assert keep == len(asm), "drift inside the assistant block must not break the seam"
    assert ids[:len(asm)] == asm
    print("ok: arbitrary render drift inside the assistant block still hits")


def test_length_cut_drops_cache():
    svc = make_service(script="one two three four five", max_new_default=64)
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": U1}]
    ids, events, _ = turn(svc, msgs, max_new=3)
    done = [e for e in events if isinstance(e, dict)][0]
    assert done["finish"] == "length"
    assert not svc.asm_valid, "a length-cut turn has no canonical continuation; the cache must drop"
    print("ok: a length-cut turn drops the cache")


def test_cancel_drops_cache():
    svc = make_service()
    cancel = threading.Event()
    cancel.set()
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": U1}]
    _, events, _ = turn(svc, msgs, cancel=cancel)                 # pre-cancelled
    done = [e for e in events if isinstance(e, dict)][0]
    assert not svc.asm_valid, "a cancelled request must not feed the cache"
    print("ok: a cancelled request drops the cache")


def test_endoftext_stop_drops_cache():
    """Only <|im_end|> has a canonical continuation the seam can encode.  A turn that ends on
    <|endoftext|> must drop the cache even though it is a 'stop' finish - otherwise the next
    turn would assemble ...endoftext, im_end... and diverge from the canonical render forever."""
    svc = make_service()
    eot = svc.tok.encode("<|endoftext|>", parse_special=True)[0]
    svc.engine.script = svc.tok.encode("plain ending", parse_special=True) + [eot]
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": U1}]
    ids, events, _ = turn(svc, msgs)
    done = [e for e in events if isinstance(e, dict)][0]
    assert done["finish"] == "stop" and svc.engine.script[-1] == eot
    assert not svc.asm_valid, "an <|endoftext|>-terminated turn must not feed the cache"
    ids2, _, keep2 = turn(svc, msgs + [assistant_msg("reply."), {"role": "user", "content": U2}])
    assert keep2 == 0
    print("ok: an <|endoftext|>-terminated turn drops the cache and the next turn re-prefills")


def test_mock_engine_multi_turn_no_typeerror():
    """The original P8b broke `--engine mock`: Service.run passed keep= to a mock without the
    parameter.  A full three-turn mock conversation must run clean."""
    svc = make_service()
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": U1}]
    for i in range(3):
        turn(svc, msgs)
        msgs = msgs + [assistant_msg(f"mock reply {i}"), {"role": "user", "content": f"turn {i}"}]
    assert svc.engine.last_keep >= 0
    print("ok: mock engine serves multi-turn (keep kwarg accepted)")


def test_done_cache_counters_flow_through():
    svc = make_service()
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": U1}]
    ids, events, keep = turn(svc, msgs)
    done = [e for e in events if isinstance(e, dict)][0]
    assert "cache_reused" in done and "cache_prefilled" in done
    assert done["cache_reused"] == 0 and done["cache_prefilled"] == len(ids) - 1
    print("ok: engine cache counters surface in the done payload")


def test_concurrent_prepare_is_race_free():
    """prepare() runs OUTSIDE the FIFO; two interleaved prepares must not cross-wire their
    token streams (the original P8b stashed pend_ids on `self`, so request A could send
    request B's assembled tokens)."""
    import itertools
    svc = make_service()
    turn(svc, [{"role": "system", "content": SYS}, {"role": "user", "content": U1}])
    msgs_a = [{"role": "system", "content": SYS}, {"role": "user", "content": U1},
              assistant_msg("Four."), {"role": "user", "content": U2}]
    msgs_b = [{"role": "system", "content": SYS}, {"role": "user", "content": "different first question"}]
    pa, pb = svc.prepare(msgs_a, None, {}, 64), svc.prepare(msgs_b, None, {}, 64)   # B prepares after A
    out_a = list(svc.run(*pa[:2], None, 64, {}, threading.Event(), keep=pa[2], render_text=pa[3]))
    out_b = list(svc.run(*pb[:2], None, 64, {}, threading.Event(), keep=pb[2], render_text=pb[3]))
    done_b = [e[1] for e in out_b if e[0] == "done"][0]
    assert done_b["cache_reused"] == 0, "b's disjoint prompt must not reuse a's session"
    assert svc.last_render == pb[3], "the anchor must be b's render, not a's"
    print("ok: interleaved prepares keep their own token streams (no cross-wiring)")


if __name__ == "__main__":
    for fn in [test_turn1_full_prefill, test_turn2_reuses_and_seam_is_clean, test_three_turn_chain,
               test_new_conversation_is_a_miss, test_edited_history_is_a_miss, test_tool_json_drift_still_hits,
               test_length_cut_drops_cache, test_cancel_drops_cache, test_endoftext_stop_drops_cache,
               test_mock_engine_multi_turn_no_typeerror, test_done_cache_counters_flow_through,
               test_concurrent_prepare_is_race_free]:
        fn()
    print("all prefix-cache unit tests passed")
