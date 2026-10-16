#!/usr/bin/env python3
"""dlt.py - long-session prefix hits: a ~65K-token agent context read once, then requests that each add a fresh user
message of d tokens (corpus text) to that same context; per d the engine's prompt line -> time and effective tok/s
(context / time), the number a long agent session sees.  hill/dlt.py --tag T --cfg CFG [--srv ...]"""
import argparse, json, os, re, subprocess, sys, time, random, urllib.request
sys.path.insert(0, "/home/user/strata-unify/tools"); sys.path.insert(0, "/home/user/strata/hill")
import agent_turns_bench as atb
from hc import KEY
from pb import corpus
PP = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms")
ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--cfg", required=True)
ap.add_argument("--srv", default="/home/user/strata/srv-uni.sh"); ap.add_argument("--sizes", default="128,256,512,1024,2048")
a = ap.parse_args(); cfg = json.load(open(a.cfg))
log = cfg["log"] if os.path.isabs(cfg["log"]) else os.path.join(cfg.get("cwd", "/home/user/strata"), cfg["log"])
subprocess.run(["bash", a.srv, "stop"], capture_output=True)
print(subprocess.run(["bash", a.srv, os.path.abspath(a.cfg), "8126"], capture_output=True, text=True).stdout.strip()[-60:], flush=True)
base = atb.turns()[3][1]          # server-option.3, ~65K tokens
text = corpus(); rng = random.Random(9)
def req(msgs):
    body = {"model": cfg.get("model_name", "m"), "messages": msgs, "tools": atb.TOOLS, "max_tokens": 8, "temperature": 1.0,
            "reasoning_effort": "none", "stream": False}
    at = os.path.getsize(log)
    r = urllib.request.Request("http://127.0.0.1:8126/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t0 = time.time(); urllib.request.urlopen(r, timeout=3600).read(); w = time.time() - t0; time.sleep(0.3)
    m = PP.findall(open(log, "rb").read()[at:].decode("utf-8", "replace"))
    return (tuple(map(int, m[-1])) if m else None), w
print("base:", req(base), flush=True)
res = []
for rep in range(2):
    for d in [int(x) for x in a.sizes.split(",")]:
        s = rng.randrange(0, len(text) - d * 4)
        msg = base + [{"role": "user", "content": f"[note {rep}-{d}] Also consider this code:\n" + text[s:s + int(d * 3.2)]}]
        m, w = req(msg)
        if not m: print("  no line", flush=True); continue
        n, reu, rd, ms = m; res.append((d, n, reu, rd, ms))
        print(f"  d={d:5d}: {n} tok, reused {reu}, read {rd} in {ms} ms -> effective {n/ms*1000:7.0f} tok/s (wall {w:.1f}s)", flush=True)
tot_n = sum(r[1] for r in res); tot_ms = sum(r[4] for r in res)
print(f"[dlt] {a.tag}: effective {tot_n/tot_ms*1000:.0f} tok/s over {len(res)} hits", flush=True)
