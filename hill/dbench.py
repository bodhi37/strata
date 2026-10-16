#!/usr/bin/env python3
"""dbench.py - R27 decode bench: diverse prompts, long greedy outputs, median decode tok/s.
Usage: hill/dbench.py PORT [ROUNDS]   (server already up)."""
import json, os, statistics, sys, time, urllib.request
HOST = os.environ.get("STRATA_HOST", "127.0.0.1")
kp = os.path.expanduser("~/.config/qwen-serve/api-key")
KEY = open(kp).read().strip() if os.path.exists(kp) else "local"
PROMPTS = [
    "Write a Python implementation of a thread-safe LRU cache with TTL expiry, with docstrings and a short usage example.",
    "Explain how TCP congestion control works (slow start, congestion avoidance, fast retransmit, fast recovery) to a new engineer.",
    "Write a short story (about 400 words) about a lighthouse keeper who discovers the light has been signalling someone.",
    "Solve step by step: a train leaves at 3pm at 80 km/h, another leaves the same station at 4pm at 110 km/h on the same track. When and where does the second catch the first? Then generalise the formula.",
    "Compare PostgreSQL and SQLite for a small SaaS backend: durability, concurrency, operations, backups, and when to switch.",
    "Translate the following idea into a detailed project plan with milestones: building a home weather station that uploads data to a dashboard.",
]
def run(port, prompt, max_tokens):
    body = {"model": "m", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "reasoning_effort": "none"}
    req = urllib.request.Request(f"http://{HOST}:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t0 = time.time(); tf = None; n = 0; usage = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line[6:] == "[DONE]":
                continue
            d = json.loads(line[6:])
            usage = d.get("usage") or usage
            delta = (d.get("choices") or [{}])[0].get("delta", {})
            if delta.get("content") or delta.get("reasoning_content"):
                tf = tf or time.time()
    t1 = time.time()
    ct = (usage or {}).get("completion_tokens", 0)
    return ct, (ct - 1) / (t1 - tf) if tf and ct > 1 else float("nan")
port = int(sys.argv[1]); rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 1
mt = int(os.environ.get("MAXTOK", "320"))
rates = []
for r in range(rounds):
    for i, p in enumerate(PROMPTS):
        ct, rate = run(port, p, mt)
        rates.append(rate)
        print(f"  p{i} ctok={ct} decode={rate:.2f}", flush=True)
print(f"DECODE median {statistics.median(rates):.2f}  mean {statistics.mean(rates):.2f}  min {min(rates):.2f}  max {max(rates):.2f}  (n={len(rates)})")
