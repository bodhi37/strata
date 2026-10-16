#!/usr/bin/env python3
"""pfcheck.py - prefill equivalence: the same long prompt through several arms (env overrides on a base config), each
dumping the final residual rows every 64th position (STRATA_PREFILL_DUMP_R); compares every arm against the first."""
import json, os, struct, subprocess, sys, time, urllib.request
import numpy as np
sys.path.insert(0, "/home/user/strata/hill")
from hc import KEY
from pb import corpus
base = json.load(open(sys.argv[1]))
arms = [a.split("|", 1) for a in sys.argv[2:]]          # name|python-mutation (a, e, c)
text = corpus()
prompt = "[pfcheck fixed prompt]\nSummarize what this code does:\n\n" + text[200000:200000 + 48000]
D = 4 * 2560
dumps = {}
for name, mut in arms:
    c = json.loads(json.dumps(base)); a = c["args"]; e = c["env"]
    exec(mut)
    dump = f"/home/user/strata/data/pfdump-{name}.bin"
    if os.path.exists(dump): os.remove(dump)
    e["STRATA_PREFILL_DUMP_R"] = dump
    c["log"] = f"/home/user/strata/logs/pfc-{name}.log"
    cfgp = f"/home/user/strata/hill/cfg-pfc-{name}.json"
    json.dump(c, open(cfgp, "w"), indent=1)
    srv = "/home/user/strata/srv-uni.sh"
    subprocess.run(["bash", srv, "stop"], capture_output=True)
    r = subprocess.run(["bash", srv, cfgp, "8126"], capture_output=True, text=True)
    print(name, r.stdout.strip()[-60:], flush=True)
    body = {"model": c.get("model_name", "m"), "messages": [{"role": "user", "content": prompt}], "max_tokens": 8,
            "stream": False, "reasoning_effort": "none", "temperature": 0}
    req = urllib.request.Request("http://127.0.0.1:8126/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t0 = time.time(); out = json.loads(urllib.request.urlopen(req, timeout=3600).read()); wall = time.time() - t0
    txt = out["choices"][0]["message"].get("content") or ""
    raw = open(dump, "rb").read()
    rows = {}
    rec = 8 + D * 4
    for i in range(0, len(raw) - rec + 1, rec):
        pos = struct.unpack_from("<q", raw, i)[0]
        rows[pos] = np.frombuffer(raw, dtype=np.float32, count=D, offset=i + 8)
    dumps[name] = rows
    print(f"  {name}: {len(rows)} rows, wall {wall:.1f}s, greedy continuation {txt!r}", flush=True)
ref = arms[0][0]
for name, _ in arms[1:]:
    common = sorted(set(dumps[ref]) & set(dumps[name]))
    a = np.stack([dumps[ref][p] for p in common]); b = np.stack([dumps[name][p] for p in common])
    rel = np.abs(a - b).max(axis=1) / (np.abs(a).max(axis=1) + 1e-30)
    bitwise = int((a == b).all(axis=1).sum())
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-30)
    print(f"{name} vs {ref}: {len(common)} rows, bitwise-equal rows {bitwise}, rel max-diff median {np.median(rel):.2e} max {rel.max():.2e}, cos min {cos.min():.5f} median {np.median(cos):.5f}")
    for lo in range(0, max(common) + 1, 2048):
        m = [i for i, p in enumerate(common) if lo <= p < lo + 2048]
        if m: print(f"    pos {lo:6d}+: rel med {np.median(rel[m]):.2e} cos min {cos[m].min():.5f} bitwise {int((a[m] == b[m]).all(axis=1).sum())}/{len(m)}")
