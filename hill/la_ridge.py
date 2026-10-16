#!/usr/bin/env python3
"""la_ridge.py - learned cross-layer miss predictor study.

Fits, per layer pair (L -> L+d), a ridge map M from the layer-L router input x_L to layer L+d's router logits
(target R[L+d] @ x_{L+d}), shrunk toward the raw router R[L+d] (the zero-training predictor).  Train on the first
part of the dump, test on the rest; reports recall of simulated-LRU misses vs reads issued per window.
"""
import numpy as np, collections, argparse

ap = argparse.ArgumentParser()
ap.add_argument("dump")
ap.add_argument("--pack", default="/home/user/strata/packs/sc117-iq3s")
ap.add_argument("--slots", type=int, default=307)
ap.add_argument("--train", type=float, default=0.7)
ap.add_argument("--lams", default="1e9,1000,100,10")
ap.add_argument("--depth", type=int, default=1)
ap.add_argument("--save", default="")
a = ap.parse_args()
H, NE, NL = 2560, 512, 48

idx = {}
for line in open(a.pack + "/index.txt"):
    if line.startswith("#"): continue
    f = line.split(); idx[f[0]] = f
dense = np.memmap(a.pack + "/dense.bin", dtype=np.uint8, mode="r")
R = np.zeros((NL, NE, H), np.float32)
for l in range(NL):
    f = idx[f"blk.{l}.ffn_gate_inp.weight"]
    off, n = int(f[3]), int(f[4])
    R[l] = (np.frombuffer(dense[off:off + n].tobytes(), dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(NE, H)

raw = np.fromfile(a.dump, dtype=np.uint8)
p = 0
wins = []          # per window: list per layer of (ids [r,k], x [r,H] f16)
cur = None
last = -1
while p + 12 <= len(raw):
    layer, nt, k = (int(v) for v in np.frombuffer(raw[p:p + 12].tobytes(), dtype=np.int32))
    if p + 12 + 4 * nt * k + 2 * nt * H > len(raw): break
    p += 12
    ids = np.frombuffer(raw[p:p + 4 * nt * k].tobytes(), dtype=np.int32).reshape(nt, k); p += 4 * nt * k
    x = np.frombuffer(raw[p:p + 2 * nt * H].tobytes(), dtype=np.float16).reshape(nt, H); p += 2 * nt * H
    if layer == 0 and last != 0:
        cur = [[[], []] for _ in range(NL)]
        wins.append(cur)
    cur[layer][0].append(ids); cur[layer][1].append(x)
    last = layer
wins = [w for w in wins if all(len(w[l][0]) for l in range(NL))]
for w in wins:
    for l in range(NL):
        w[l] = (np.concatenate(w[l][0]), np.concatenate(w[l][1]))
wins = [w for w in wins if len({len(w[l][0]) for l in range(NL)}) == 1]
nw = len(wins); ntr = int(nw * a.train)
print(f"{nw} windows ({ntr} train), {sum(len(w[0][0]) for w in wins)} rows", flush=True)

# LRU sim over ALL windows in order (misses and resident snapshot per (win, layer))
freq = np.zeros((NL, NE))
for w in wins[:ntr]:
    for l in range(NL):
        freq[l, np.unique(w[l][0])] += 1
caches = []
for l in range(NL):
    od = collections.OrderedDict()
    for e in np.argsort(-freq[l])[:a.slots][::-1]: od[int(e)] = 1
    caches.append(od)
misses, resid = [], []
for w in wins:
    ml, rl = [], []
    for l in range(NL):
        c = caches[l]
        rl.append(np.fromiter(c.keys(), dtype=np.int64))
        m = []
        for e in np.unique(w[l][0]).tolist():
            if e in c: c.move_to_end(e)
            else:
                m.append(e); c[e] = 1
                if len(c) > a.slots: c.popitem(last=False)
        ml.append(m)
    misses.append(ml); resid.append(rl)

d = a.depth
KS = (6, 8, 10, 12, 16, 20)
lams = [float(v) for v in a.lams.split(",")]
res = {lam: {kk: [0, 0] for kk in KS} for lam in lams}
mtot = 0
save = {}
for L in range(NL - d):
    t = L + d
    Xtr = np.concatenate([w[L][1] for w in wins[:ntr]]).astype(np.float32)
    Ztr = np.concatenate([w[t][1] for w in wins[:ntr]]).astype(np.float32) @ R[t].T
    G = Xtr.T @ Xtr
    B = Xtr.T @ Ztr
    tr = np.trace(G) / H
    for lam in lams:
        if lam >= 1e8: M = R[t].T
        else:
            l2 = lam * tr / 100.0
            M = np.linalg.solve(G + l2 * np.eye(H, dtype=np.float32), B + l2 * R[t].T)
        if a.save and lam == lams[-1]: save[L] = M.astype(np.float16)
        for wi in range(ntr, nw):
            lg = wins[wi][L][1].astype(np.float32) @ M
            order = np.argsort(-lg, axis=1)[:, :max(KS)]
            rs = set(resid[wi][t].tolist()); ms = set(misses[wi][t])
            for kk in KS:
                pred = set(order[:, :kk].ravel().tolist()) - rs
                res[lam][kk][0] += len(pred & ms); res[lam][kk][1] += len(pred)
    mtot += sum(len(misses[wi][t]) for wi in range(ntr, nw))
    if L % 8 == 7: print(f"  layer {L} done", flush=True)
nt = nw - ntr
for lam in lams:
    for kk in KS:
        h, c = res[lam][kk]
        print(f"lam {lam:g} k'={kk:2d}: recall {h / mtot:.3f} reads {c / nt:.1f}/win precision {h / max(1, c):.3f} (misses {mtot / nt:.1f}/win)")
if a.save:
    np.savez(a.save, **{str(k): v for k, v in save.items()})
