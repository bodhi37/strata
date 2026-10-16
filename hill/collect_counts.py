#!/usr/bin/env python3
"""collect_counts.py - drive the training corpus through a running engine started with --dump-counts.

  hill/collect_counts.py --cfg hill/cfg-collect.json [--max-tokens 400] [--restart]
"""
import argparse, json, os, subprocess, sys, time
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "hill"))
from xb import chat  # noqa: E402
from train_prompts import TRAIN  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cfg", required=True)
ap.add_argument("--port", type=int, default=8126)
ap.add_argument("--max-tokens", type=int, default=400)
ap.add_argument("--restart", action="store_true")
a = ap.parse_args()
cfgp = a.cfg if os.path.isabs(a.cfg) else os.path.join(REPO, a.cfg)
model = json.load(open(cfgp)).get("model_name", "m")
host = os.environ.get("STRATA_HOST", "127.0.0.1")
if a.restart:
    subprocess.run(["bash", os.path.join(REPO, "srv.sh"), "stop"], capture_output=True, timeout=120)
    r = subprocess.run(["bash", os.path.join(REPO, "srv.sh"), cfgp, str(a.port)], capture_output=True, text=True, timeout=900)
    print(r.stdout.strip(), flush=True)
t0 = time.time()
for i, p in enumerate(TRAIN):
    wall, d = chat(host, a.port, model, p, a.max_tokens)
    n = d.get("usage", {}).get("completion_tokens")
    print(f"[{i+1}/{len(TRAIN)}] {n} tok in {wall:.1f}s  (elapsed {time.time()-t0:.0f}s)", flush=True)
