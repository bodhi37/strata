#!/usr/bin/env python3
"""sim4.py - can a window's MISSES be predicted at window start from its tokens? (the H4 prefetch question)

Replays a token-annotated route trace (STRATA_ROUTE_TRACE from engine/strata-h4: per-layer records, the -1 footer,
then a -2 record with the window's T tokens) through the engine's tiers (layered VRAM + small static + exact LRU)
and scores an online predictor: for each row's token, the experts that token was routed to before (per layer, the
top `m` by count).  Predicted misses = predictions not resident when the window starts.

  hill/sim4.py data/trace-h4.bin [--m 4] [--warm 0.5]
"""
import argparse, os, sys
from collections import defaultdict, Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim import NL, NE, blob_bytes, load_counts, ranking_from_counts, build_tiers  # noqa: E402
from sim2 import FreqLRU, GiB  # noqa: E402


def load(path):
    raw = np.fromfile(path, dtype="<i2")
    wins, cur, i, n = [], {}, 0, len(raw)
    while i + 3 <= n:
        layer, nt, k = int(raw[i]), int(raw[i + 1]), int(raw[i + 2])
        i += 3
        if layer == -1:
            wins.append([nt, cur, None])
            cur = {}
            continue
        if layer == -2:
            u = raw[i:i + 2 * nt].view(np.uint16).astype(np.int64)
            toks = u[0::2] | (u[1::2] << 16)
            if wins and wins[-1][2] is None:
                wins[-1][2] = toks
            i += 2 * nt
            continue
        if not (0 <= layer < NL) or nt <= 0 or k <= 0 or i + nt * k > n:
            break
        if layer == 0 and cur:
            cur = {}
        cur[layer] = raw[i:i + nt * k].astype(np.int64).reshape(nt, k)
        i += nt * k
    return [w for w in wins if w[2] is not None and len(w[1]) == NL]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--counts", default="data/counts-t1-train.strc")
    ap.add_argument("--m", default="2,4,8")
    ap.add_argument("--warm", type=float, default=0.5, help="score only the windows after this fraction")
    ap.add_argument("--lru", type=float, default=22.9)
    ap.add_argument("--static", type=float, default=2.0)
    a = ap.parse_args()
    bb = blob_bytes()
    wins = load(a.trace)
    print(f"{len(wins)} windows with tokens")
    c = load_counts(a.counts)
    rank = ranking_from_counts(c)
    vram, ram = build_tiers(rank, bb, 2208, int(a.static * GiB), True)
    plen = {int(b): (int(b) + 4095) // 4096 * 4096 for b in bb}
    nl = Counter(int(b) for b in bb)
    den = sum(n * plen[b] for b, n in nl.items())
    for m in [int(x) for x in a.m.split(",")]:
        pools = {b: FreqLRU(int(a.lru * GiB * n * plen[b] / den // plen[b])) for b, n in nl.items()}
        table = defaultdict(Counter)        # (token, layer) -> expert counts
        start = int(len(wins) * a.warm)
        miss = pred = hit = acc_rows = 0
        for ep, (T, layers, toks) in enumerate(wins):
            score = ep >= start
            # predictions at window start
            P = {}
            for l in range(NL):
                s = set()
                for r in range(T):
                    tc = table.get((int(toks[r]), l))
                    if tc:
                        for e, _ in tc.most_common(m):
                            s.add(l * NE + e)
                P[l] = {i for i in s if not (vram[i] or ram[i] or i in pools[int(bb[l])].od)}
            for l in range(NL):
                ids = layers[l]
                u = np.unique(ids + l * NE)
                pool = pools[int(bb[l])]
                for idx in u:
                    idx = int(idx)
                    if vram[idx] or ram[idx] or pool.access(idx, ep):
                        continue
                    if score:
                        miss += 1
                        if idx in P[l]:
                            hit += 1
                if score:
                    pred += len(P[l])
                for r in range(min(T, ids.shape[0])):
                    tc = table[(int(toks[r]), l)]
                    for e in ids[r]:
                        tc[int(e)] += 1
            if score:
                acc_rows += T
        print(f"m={m}: misses {miss} ({miss / max(acc_rows, 1):.2f}/row)  predicted reads {pred} "
              f"({pred / max(acc_rows, 1):.2f}/row)  caught {hit}  recall {hit / max(miss, 1):.1%}  "
              f"precision {hit / max(pred, 1):.1%}", flush=True)


if __name__ == "__main__":
    main()
