#!/usr/bin/env python3
"""la_study.py - offline study of cross-layer routing prediction for disk-miss prefetch.

Reads the STRATA_DUMP_RIN dump (per pool call: layer, n_tok, k, ids, fp16 router input) and the BF16 routers
from the pack, simulates a per-layer LRU (seeded by frequency) to find the misses, and reports how many of
the misses of layer L+d are predicted by router_{L+d}(x_L) top-k', and how many non-resident reads that costs.
"""
import sys, numpy as np, collections, argparse

ap = argparse.ArgumentParser()
ap.add_argument("dump")
ap.add_argument("--pack", default="/home/user/strata/packs/sc117-iq3s")
ap.add_argument("--slots", type=int, default=307)
ap.add_argument("--max-win", type=int, default=100000)
a = ap.parse_args()

H, NE, NL = 2560, 512, 48
# routers
idx = {}
for line in open(a.pack + "/index.txt"):
    if line.startswith("#"): continue
    f = line.split()
    idx[f[0]] = f
dense = np.memmap(a.pack + "/dense.bin", dtype=np.uint8, mode="r")
R = np.zeros((NL, NE, H), np.float32)
for l in range(NL):
    f = idx[f"blk.{l}.ffn_gate_inp.weight"]
    off, n = int(f[3]), int(f[4])
    w = np.frombuffer(dense[off:off + n].tobytes(), dtype=np.uint16).astype(np.uint32) << 16
    R[l] = w.view(np.float32).reshape(NE, H)

# parse dump into windows: list of per-layer (ids set list per row, x rows)
raw = np.fromfile(a.dump, dtype=np.uint8)
p = 0
wins = []
cur = None
while p + 12 <= len(raw):
    layer, nt, k = np.frombuffer(raw[p:p + 12].tobytes(), dtype=np.int32)
    p += 12
    ids = np.frombuffer(raw[p:p + 4 * nt * k].tobytes(), dtype=np.int32).reshape(nt, k)
    p += 4 * nt * k
    if p + 2 * nt * H > len(raw): break
    x = np.frombuffer(raw[p:p + 2 * nt * H].tobytes(), dtype=np.float16).reshape(nt, H).astype(np.float32)
    p += 2 * nt * H
    if layer == 0 and (cur is None or cur["L"][0] is not None and len(cur["ids"][0]) and cur.get("done0")):
        pass
    if layer == 0 and (cur is None or cur["last"] != 0):
        cur = {"ids": [[] for _ in range(NL)], "x": [[] for _ in range(NL)], "last": 0, "L": [None]}
        wins.append(cur)
    cur["ids"][layer].append(ids)
    cur["x"][layer].append(x)
    cur["last"] = layer
    if len(wins) > a.max_win: break
wins = [w for w in wins if all(len(w["ids"][l]) for l in range(NL))]
for w in wins:
    for l in range(NL):
        w["ids"][l] = np.concatenate(w["ids"][l])
        w["x"][l] = np.concatenate(w["x"][l])
print(f"{len(wins)} windows, {sum(len(w['ids'][0]) for w in wins)} rows")

# frequency seed + LRU sim -> misses per (win, layer)
freq = np.zeros((NL, NE))
for w in wins:
    for l in range(NL):
        for e in np.unique(w["ids"][l]):
            freq[l, e] += 1
caches = []
for l in range(NL):
    od = collections.OrderedDict()
    for e in np.argsort(-freq[l])[:a.slots][::-1]:
        od[int(e)] = 1
    caches.append(od)
misses = []  # per win: list per layer of set
resident_at = []  # resident set snapshot (before the layer) for cost of predictions
for w in wins:
    ml, rl = [], []
    for l in range(NL):
        c = caches[l]
        rl.append(set(c.keys()))
        s = set(int(e) for e in np.unique(w["ids"][l]))
        m = set()
        for e in s:
            if e in c:
                c.move_to_end(e)
            else:
                m.add(e)
                c[e] = 1
                c.move_to_end(e)
                if len(c) > a.slots: c.popitem(last=False)
        ml.append(m)
    misses.append(ml)
    resident_at.append(rl)
tot_m = sum(len(m) for ml in misses for m in ml)
tot_rows = sum(len(w["ids"][0]) for w in wins)
print(f"misses {tot_m} = {tot_m / tot_rows:.1f}/token, {tot_m / len(wins):.1f}/window")

def topk(logits, k):
    return np.argpartition(-logits, k - 1, axis=1)[:, :k]

KS = (10, 16, 24, 32)
for d in (1, 2, 3):
    hit = dict.fromkeys(KS, 0); cost = dict.fromkeys(KS, 0)
    for wi, w in enumerate(wins):
        for l in range(NL - d):
            t = l + d
            lg = w["x"][l] @ R[t].T
            order = np.argsort(-lg, axis=1)[:, :max(KS)]
            res = resident_at[wi][t]; mis = misses[wi][t]
            for kk in KS:
                pred = set(order[:, :kk].ravel().tolist())
                nonres = pred - res
                cost[kk] += len(nonres)
                hit[kk] += len(nonres & mis)
    m_d = sum(len(ml[t]) for ml in misses for t in range(d, NL))
    for kk in KS:
        print(f"depth {d} k'={kk:2d}: recall of misses {hit[kk] / max(1, m_d):.3f}  reads {cost[kk] / len(wins):.1f}/win "
              f"(precision {hit[kk] / max(1, cost[kk]):.3f}, misses {m_d / len(wins):.1f}/win)", flush=True)
