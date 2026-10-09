#!/usr/bin/env python3
"""Deep-prefill boundary drill for the strata endpoint.

Generates a filler prompt near a target token count, sends TURN 1 completely, then sends TURN 2 with the same
prefix plus a small suffix and zero gap - the session-turn boundary that used to die in cublasCreate under
memory pressure (deep context + right after a long decode).  Prints per-turn status + token counts, exits
non-zero on any failure.
"""
import json, sys, time, urllib.request

BASE = "http://100.87.70.9:8126"
KEY = open("/home/bodhi/.config/qwen-serve/api-key").read().strip()

UNIT = "Reference line %d: 0123456789abcdef the quick brown fox jumps over the lazy dog and parks the tractor.\n"

def text_for_tokens(n_tokens):
    per = len(UNIT % 123456) + 14    # ~53.4 chars per entry incl. line break
    n = max(1, int(n_tokens * 3.8 / per))
    return "".join(UNIT % i for i in range(n))

def call(label, content, max_tokens=32):
    body = {"model": "qwen3.8-flash-next-gsq-rco-abliterated-iq3s",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens, "reasoning_effort": "none", "stream": False}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        b = json.loads(r.read())
    u = b.get("usage", {})
    print(f"{label}: {r.status} prompt_tokens={u.get('prompt_tokens')} "
          f"completion_tokens={u.get('completion_tokens')} wall={time.time()-t0:.0f}s", flush=True)
    return u.get("prompt_tokens")

if __name__ == "__main__":
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 78000
    turns = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    txt = text_for_tokens(target)
    n =call("turn 1 (deep prefill)", txt)
    for t in range(2, turns + 1):
        txt2 = txt + f"\n\nNow, out of the materials above:-reply with exactly: OK-{t}."
        call(f"turn {t} (boundary, no gap)", txt2)
    print("DRILL COMPLETE")
