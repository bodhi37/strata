#!/usr/bin/env python3
"""coact.py - can layer L's routed set predict layer L+1's MISSES?  (cross-layer prefetch feasibility)

Learns co-activation counts C[L][e][f] = #(e routed at L and f routed at L+1, same token) from a TRAIN trace,
then on an EVAL trace predicts, for each window and layer L+1, the non-resident experts whose score
sum_{e in routed(L)} C[L][e][f] / count(L,e) passes --thr (top --m per window-layer), and reports
recall (of the real misses) and precision (of the predicted reads).

  hill/coact.py TRAIN_TRACE EVAL_TRACE --profile data/profile.bin --vram-slots 1700 --ram-gib 25
"""
import argparse, os, struct, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("train"); ap.add_argument("eval")
ap.add_argument("--profile", required=True)
ap.add_argument("--vram-slots", type=int, default=1700)
ap.add_argument("--ram-gib", type=float, default=25)
ap.add_argument("--m", type=int, default=4)
a = ap.parse_args()
NL, NE = sim.NL, sim.NE
bb = sim.blob_bytes()
b = open(a.profile, "rb").read(); _, nl, ne, sl, nr = struct.unpack("<5I", b[4:24])
pr = np.frombuffer(b, "<u2", 2 * nr, 24).reshape(-1, 2)
rank = [int(l) * NE + int(e) for l, e in pr]
vram, ram = sim.build_tiers(rank, bb, a.vram_slots, int(a.ram_gib * 2**30), True)
res = (vram | ram).reshape(NL, NE)

C = np.zeros((NL - 1, NE, NE), np.float32)
cnt = np.zeros((NL, NE), np.float32)
for T, acc, sfx, layers in sim.load_trace(a.train):
    for l in range(NL - 1):
        if l not in layers or l + 1 not in layers: continue
        x = layers[l].reshape(-1, 10).astype(int); y = layers[l + 1].reshape(-1, 10).astype(int)
        for t in range(len(x)):
            C[l][np.ix_(x[t], y[t])] += 1
            cnt[l][x[t]] += 1
P = C / np.maximum(cnt[:-1, :, None], 1)

hit = miss = pred = good = 0
for T, acc, sfx, layers in sim.load_trace(a.eval):
    for l in range(NL - 1):
        if l not in layers or l + 1 not in layers: continue
        x = layers[l].reshape(-1, 10).astype(int)
        actual = set(int(f) for f in np.unique(layers[l + 1]) if not res[l + 1][f])
        score = np.zeros(NE)
        for t in range(len(x)):
            score += P[l][x[t]].sum(0)
        score[res[l + 1]] = -1
        top = [int(f) for f in np.argsort(-score)[:a.m * max(1, len(x))] if score[f] > 0]
        pred += len(top); good += len(actual & set(top)); miss += len(actual)
print(f"m={a.m}: real misses {miss}, predicted reads {pred}, recall {good / max(miss, 1):.1%}, "
      f"precision {good / max(pred, 1):.1%}")
