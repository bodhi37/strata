#!/usr/bin/env python3
"""R22: repeated-sample prefill benchmark with UNIQUE prompts.

The single-shot prefill numbers in REPORT.md swing 2x run to run on this box (measured warmup 54-89 s across
four launches), so one sample cannot decide an A/B.  Each prompt here starts with a random nonce so the
engine's conversation cache never resumes it, and the caller gets min/median/max over K samples instead of a
single number.  max_tokens is 4 so the 4 decode tokens are noise against a 12k prefill.
"""
import json, time, urllib.request, sys, random, statistics

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8123
NTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 12000
K = int(sys.argv[3]) if len(sys.argv) > 3 else 6

WORDS = ("lighthouse keeper horizon lens mechanics tide logbook storm brass gear mercury barometric "
         "pressure dusk fog beacon gale buoy rope canvas flare chart reef compass").split()

para = ("The lighthouse keeper surveyed the horizon: lens mechanics, tide tables, and the logbook of "
        "storms. He adjusted the brass gears, checked the mercury bath, and recorded the barometric "
        "pressure at dusk. ")


def post(payload, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 json.dumps(payload).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


rates = []
for k in range(K):
    # a unique first line defeats the conversation cache; a different body makes each sweep a bit different
    nonce = " ".join(random.choice(WORDS) for _ in range(12)) + f" {random.randrange(1 << 30):x}"
    body = para * (NTOK // 46 + 1)
    prompt = nonce + " " + body + "\n\nSummarize the keeper's routine in one sentence."
    t0 = time.time()
    r = post({"model": "x", "messages": [{"role": "user", "content": prompt}],
              "max_tokens": 4, "temperature": 0})
    dt = time.time() - t0
    u = r.get("usage", {})
    pt = u.get("prompt_tokens", 0)
    rate = pt / dt
    rates.append(rate)
    print(f"  sample {k}: prompt={pt} tok  wall={dt:6.1f}s  prefill~={rate:6.1f} tok/s", flush=True)

med = statistics.median(rates)
print(f"prefill summary: min {min(rates):.0f}  median {med:.0f}  max {max(rates):.0f}  (n={K})")
