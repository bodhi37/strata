#!/usr/bin/env python3
"""gcmp.py BASE.json "name|mutation" ... : the same fixed prompts, greedy (temperature 0, an identity check only - never a
bench), through each arm; prints whether every arm's text matches the first arm's."""
import json, subprocess, sys, urllib.request
sys.path.insert(0, "/home/user/strata/hill")
from hc import KEY
base = json.load(open(sys.argv[1]))
prompts = ["Write a Python function that merges two sorted lists, then explain its complexity.",
           "Explain in detail how a CPU cache hierarchy works, with examples.",
           "List ten creative names for a coffee shop and a one-line pitch for each."]
outs = {}
for spec in sys.argv[2:]:
    name, mut = spec.split("|", 1)
    c = json.loads(json.dumps(base)); a = c["args"]; e = c["env"]
    exec(mut)
    c["log"] = f"/home/user/strata/logs/gc-{name}.log"
    p = f"/home/user/strata/hill/cfg-gc-{name}.json"; json.dump(c, open(p, "w"), indent=1)
    subprocess.run(["bash", "/home/user/strata/srv-uni.sh", "stop"], capture_output=True)
    r = subprocess.run(["bash", "/home/user/strata/srv-uni.sh", p, "8126"], capture_output=True, text=True)
    print(name, r.stdout.strip()[-50:], flush=True)
    outs[name] = []
    for pr in prompts:
        body = {"model": c.get("model_name", "m"), "messages": [{"role": "user", "content": pr}], "max_tokens": 200,
                "stream": False, "reasoning_effort": "none", "temperature": 0}
        req = urllib.request.Request("http://127.0.0.1:8126/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
        outs[name].append(json.loads(urllib.request.urlopen(req, timeout=600).read())["choices"][0]["message"].get("content") or "")
ref = sys.argv[2].split("|")[0]
for n in outs:
    same = [a == b for a, b in zip(outs[ref], outs[n])]
    print(f"{n} vs {ref}: identical {sum(same)}/{len(same)}", flush=True)
    for i, s in enumerate(same):
        if not s:
            k = next((j for j in range(min(len(outs[ref][i]), len(outs[n][i]))) if outs[ref][i][j] != outs[n][i][j]), -1)
            print(f"   prompt {i}: first difference at char {k}: {outs[ref][i][max(0,k-30):k+30]!r} / {outs[n][i][max(0,k-30):k+30]!r}")
