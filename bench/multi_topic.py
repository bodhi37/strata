#!/usr/bin/env python3
"""R8 acceptance: decode across many topics, long generations, coherence spot-checks."""
import json, time, urllib.request, sys

URL = "http://127.0.0.1:8111/v1"

TOPICS = [
    ("cooking", "Explain how sourdough starters work, what feeds the yeast, and why the ratio of flour to water matters. Be thorough.", 300),
    ("coding", "Write a Python function that merges two sorted lists without duplicates, then explain its complexity. Include edge cases.", 300),
    ("law", "Describe how a congressional bill becomes law in the United States, including committee stages and veto overrides.", 300),
    ("medicine", "Explain how mRNA vaccines work, from injection to immune response, in detail.", 300),
    ("finance", "Explain how compound interest works and derive the formula for monthly mortgage payments.", 300),
    ("history", "Describe the causes and consequences of the fall of the Western Roman Empire.", 300),
    ("physics", "Explain the principle of superposition in quantum mechanics and how it differs from classical wave superposition.", 300),
    ("agentic", "You are debugging a web service with 500 errors under load. Walk through a systematic diagnosis plan: metrics to check, hypotheses to test, and fixes for each finding.", 300),
    ("poetry", "Write a short essay on why iambic pentameter became the dominant meter in English poetry.", 300),
    ("math", "Prove that the square root of 2 is irrational, then explain the generalization to sqrt(n) for non-square n.", 300),
]

def ask(prompt, maxt):
    body = {"model": "x", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": maxt, "reasoning_effort": "none"}
    req = urllib.request.Request(URL + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    ct = d["usage"]["completion_tokens"]
    return dt, ct, d["choices"][0]["message"]["content"]

rows = []
for name, prompt, maxt in TOPICS:
    dt, ct, txt = ask(prompt, maxt)
    rate = ct / dt
    # crude coherence probe: not empty, has sentence structure, no degenerate repetition
    bad_rep = any(t * 8 in txt for t in ("!", "?", "the the the"))
    coherent = len(txt) > 200 and not bad_rep and txt.count("\n\n\n") < 3
    rows.append((name, dt, ct, rate, coherent))
    print(f"{name:9s}: wall {dt:6.1f}s  {ct} tok  {rate:6.2f} tok/s  {'OK' if coherent else 'CHECK'}  | {txt[:70].replace(chr(10),' ')!r}")

rates = [r[3] for r in rows]
print(f"\nmin {min(rates):.2f}  median {sorted(rates)[len(rates)//2]:.2f}  max {max(rates):.2f} tok/s")
