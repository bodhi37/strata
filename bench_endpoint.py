#!/usr/bin/env python3
"""bench_endpoint.py - honest throughput / latency measurement of the Strata endpoint.

Sends one long-context request per target size, greedily decodes a fixed number of tokens,
and reports prefill, TTFT and decode throughput.  Token counts come from the server's own
`usage` chunk, so the reported prompt size is the real one, not an estimate.

  python3 bench_endpoint.py --port 8101 --contexts 1024,4096,32768,131072
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request

BASE = ("def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n\n"
        "The quick brown fox jumps over the lazy dog.  A distributed system reaches consensus\n"
        "when a quorum of replicas agrees on a value despite crashes and message loss; the\n"
        "safety property must hold across every possible interleaving of events.\n\n")


def make_prompt(target_tokens: int) -> str:
    approx = max(64, target_tokens) * 4
    body = (BASE * (approx // len(BASE) + 1))[:approx]
    return ("Read the material below, then reply with the single word OK.\n\n" + body)


def one(url: str, model: str, prompt: str, max_tokens: int, effort: str, timeout: float):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "stream": True, "reasoning_effort": effort}
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer local"})
    t0 = time.time()
    tfirst = None
    n = 0
    usage = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except ValueError:
                continue
            if d.get("usage"):
                usage = d["usage"]
            delta = (d.get("choices") or [{}])[0].get("delta", {})
            if delta.get("content") or delta.get("reasoning_content"):
                if tfirst is None:
                    tfirst = time.time()
                n += 1
    t1 = time.time()
    return t0, tfirst, t1, n, usage


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8101)
    ap.add_argument("--model", default="qwen3.8-flash-next-heretic-2-iq3xxs-strata")
    ap.add_argument("--contexts", default="1024,4096,32768,131072")
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--effort", default="none", choices=["none", "low", "medium", "high"])
    ap.add_argument("--timeout", type=float, default=7200)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    url = f"http://{a.host}:{a.port}/v1"
    rows = []
    for ctx in [int(x) for x in a.contexts.split(",") if x.strip()]:
        prompt = make_prompt(ctx)
        print(f"\n== target ~{ctx} tokens ==", flush=True)
        try:
            t0, tf, t1, n, usage = one(url, a.model, prompt, a.max_tokens, a.effort, a.timeout)
        except Exception as e:
            print("  FAILED:", e, flush=True)
            rows.append({"target": ctx, "error": repr(e)})
            continue
        pt = (usage or {}).get("prompt_tokens")
        ct = (usage or {}).get("completion_tokens", n)
        ttft = (tf - t0) if tf else float("nan")
        dec = (t1 - tf) if tf else float("nan")
        prefill = (pt / ttft) if (pt and ttft and ttft > 0) else float("nan")
        decode = ((ct - 1) / dec) if (dec and dec > 0) else float("nan")
        row = {"target": ctx, "prompt_tokens": pt, "completion_tokens": ct,
               "ttft_s": ttft, "prefill_tok_s": prefill, "decode_tok_s": decode,
               "wall_s": t1 - t0}
        rows.append(row)
        print(f"  prompt={pt} tok  TTFT={ttft:.2f}s  prefill={prefill:.1f} tok/s  "
              f"decode={decode:.2f} tok/s  out={ct}  wall={t1 - t0:.1f}s", flush=True)
    print("\n" + json.dumps(rows, indent=1))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
