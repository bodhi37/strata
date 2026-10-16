#!/usr/bin/env python3
"""sim2.py - static/LRU split sweep with fixed vs exact LRU slot sizes, train/test split by time.

  hill/sim2.py TRACE [--split 0.5] [--counts C.strc] [--vram 1700] [--total 25]

Ranking = counts file if given, else the routed counts of the trace's first `split` windows; scored on the rest.
LRU policy = the engine's (lowest hit count within a 512-slot walk from the tail, LRU tie-break, admit on miss).
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim import NL, NE, blob_bytes, load_trace, load_counts, ranking_from_counts, build_tiers  # noqa: E402

GiB = 1 << 30


class FreqLRU:
    """class-aware: one pool per blob-size class when exact, else one pool"""
    def __init__(self, nslots):
        from collections import OrderedDict
        self.od = OrderedDict()   # idx -> count, LRU order (first = tail)
        self.n = nslots

    def access(self, idx, epoch):
        od = self.od
        if idx in od:
            c, _ = od.pop(idx)
            od[idx] = (min(c + 1, 4095), epoch)
            return True
        if self.n <= 0:
            return False
        if len(od) >= self.n:
            best, bc = None, 1 << 30
            for j, (key, (c, ep)) in enumerate(od.items()):
                if j >= 512:
                    break
                if ep == epoch:
                    continue
                if c < bc:
                    best, bc = key, c
                    if c <= 1:
                        break
            if best is None:
                return False
            del od[best]
        od[idx] = (1, epoch)
        return False


def run(wins, vram, ram, bb, lru_bytes, exact, cls_share):
    if exact:
        pools = {}
        for b, share in cls_share.items():
            pools[b] = FreqLRU(int(lru_bytes * share // ((b + 4095) // 4096 * 4096)))
        pool_of = lambda l: pools[int(bb[l])]
    else:
        one = FreqLRU(int(lru_bytes // int(bb.max() // 4096 * 4096 + 4096)) if lru_bytes else 0)
        pool_of = lambda l: one
    miss = mbytes = acc_tok = 0
    for ep, (T, acc, sfx, layers) in enumerate(wins):
        acc_tok += (acc + 1) if acc >= 0 else 1
        for l, ids in layers.items():
            for idx in np.unique(ids.astype(np.int64) + l * NE):
                if vram[idx] or ram[idx]:
                    continue
                if lru_bytes and pool_of(l).access(int(idx), ep):
                    continue
                miss += 1
                mbytes += int(bb[l])
    return miss / acc_tok, mbytes / acc_tok / 1e6, acc_tok / len(wins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--split", type=float, default=0.5)
    ap.add_argument("--counts", default="")
    ap.add_argument("--vram", type=int, default=1700)
    ap.add_argument("--total", type=float, default=25.0)
    ap.add_argument("--lru", default="0,4,8,12,16,20")
    ap.add_argument("--density", action="store_true")
    ap.add_argument("--exact-only", action="store_true")
    ap.add_argument("--eq", action="store_true", help="class share = equal slot count per layer")
    a = ap.parse_args()
    bb = blob_bytes()
    wins = load_trace(a.trace)
    cut = int(len(wins) * a.split)
    train, test = wins[:cut], wins[cut:]
    if a.counts:
        c = np.zeros(NL * NE)
        for p in a.counts.split(","):
            x = load_counts(p)
            c += x / x.sum()
    else:
        c = np.zeros(NL * NE)
        for (T, acc, sfx, layers) in train:
            for l, ids in layers.items():
                np.add.at(c, ids.astype(np.int64) + l * NE, 1)
    rank = ranking_from_counts(c, bb if a.density else None)
    print(f"windows train {len(train)} test {len(test)}; total RAM {a.total} GiB, vram {a.vram}")
    for lg in [float(x) for x in a.lru.split(",")]:
        vram, ram = build_tiers(rank, bb, a.vram, int((a.total - lg) * GiB), True)
        # class shares for exact pools: miss traffic bytes by size class, from train windows
        share = {}
        for (T, acc, sfx, layers) in train:
            for l, ids in layers.items():
                for idx in np.unique(ids.astype(np.int64) + l * NE):
                    if not (vram[idx] or ram[idx]):
                        share[int(bb[l])] = share.get(int(bb[l]), 0) + int(bb[l])
        tot = sum(share.values()) or 1
        for b in set(int(x) for x in bb):
            share[b] = share.get(b, 0) / tot
        if a.eq:
            nl = {}
            for x in bb:
                nl[int(x)] = nl.get(int(x), 0) + 1
            den = sum(n * ((b + 4095) // 4096 * 4096) for b, n in nl.items())
            share = {b: n * ((b + 4095) // 4096 * 4096) / den for b, n in nl.items()}
        for exact in ([True] if a.exact_only else [False, True]) if lg else [False]:
            m, mb, tpw = run(test, vram, ram, bb, int(lg * GiB), exact, share)
            print(f"lru {lg:5.1f} GiB {'exact' if exact else 'fixed'}: static {ram.sum():5d} | miss/tok {m:5.2f} "
                  f"MB/tok {mb:6.1f} tok/win {tpw:.2f} -> disk ceiling {4200 / max(mb, 1e-9):5.0f} tok/s", flush=True)


if __name__ == "__main__":
    main()
