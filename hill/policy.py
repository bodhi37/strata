#!/usr/bin/env python3
"""policy.py - RAM-tier replacement policies over a route trace (VRAM static on top, per-layer quota).

The trace is split in time: the profile (static seed + VRAM tier) is learned from windows [0, split) and the
policy is scored on [split, end) only - so nothing is graded on data it was built from.

  hill/policy.py TRACE --vram-slots 1700 --ram-gib 25 [--split 0.5]
"""
import argparse, os, sys
from collections import OrderedDict, defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim  # noqa: E402

NL, NE = sim.NL, sim.NE


def window_sets(wins, vram):
    out = []
    for T, a, sfx, layers in wins:
        s = []
        for l in sorted(layers):
            u = np.unique(layers[l].astype(np.int64) + l * NE)
            s.extend(int(i) for i in u if not vram[i])
        out.append((a + 1 if a >= 0 else 1, s))
    return out


class Static:
    def __init__(self, seed, cap): self.s = set(seed[:cap])
    def access(self, i): return i in self.s
    def end_window(self): pass


class LRU:
    def __init__(self, seed, cap):
        self.cap = cap; self.d = OrderedDict((i, 1) for i in reversed(seed[:cap]))
    def access(self, i):
        if i in self.d: self.d.move_to_end(i); return True
        self.d[i] = 1
        if len(self.d) > self.cap: self.d.popitem(last=False)
        return False
    def end_window(self): pass


class StaticLRU:
    def __init__(self, seed, cap, lru_cap):
        self.st = set(seed[:cap - lru_cap]); self.l = LRU([], lru_cap)
    def access(self, i): return i in self.st or self.l.access(i)
    def end_window(self): pass


class DecayLFU:
    """the engine's R8.1 shape: count (seeded), halved every `half` windows; evict min count (LRU tie-break)"""
    def __init__(self, seed, cap, seed_count=8.0, half=8):
        self.cap = cap; self.c = {}; self.t = {}; self.now = 0; self.half = half; self.w = 0
        for r, i in enumerate(seed[:cap]): self.c[i] = seed_count; self.t[i] = -r
        self.f = 1.0
    def access(self, i):
        self.now += 1
        if i in self.c:
            self.c[i] += self.f; self.t[i] = self.now; return True
        if len(self.c) >= self.cap:
            v = min(self.c, key=lambda k: (self.c[k], self.t[k]))   # O(n) - fine offline
            del self.c[v]; del self.t[v]
        self.c[i] = self.f; self.t[i] = self.now
        return False
    def end_window(self):
        self.w += 1
        self.f *= 2 ** (1.0 / self.half)   # equivalent to halving everyone, without touching them
        if self.f > 1e12:
            for k in self.c: self.c[k] /= self.f
            self.f = 1.0


class ARC:
    def __init__(self, seed, cap):
        self.c = cap; self.p = 0
        self.t1 = OrderedDict(); self.t2 = OrderedDict((i, 1) for i in reversed(seed[:cap]))
        self.b1 = OrderedDict(); self.b2 = OrderedDict()
    def _replace(self, i):
        if self.t1 and (len(self.t1) > self.p or (i in self.b2 and len(self.t1) == self.p)):
            k, _ = self.t1.popitem(last=False); self.b1[k] = 1
        else:
            k, _ = self.t2.popitem(last=False); self.b2[k] = 1
    def access(self, i):
        c = self.c
        if i in self.t1: del self.t1[i]; self.t2[i] = 1; return True
        if i in self.t2: self.t2.move_to_end(i); return True
        if i in self.b1:
            self.p = min(c, self.p + max(1, len(self.b2) // max(1, len(self.b1))))
            self._replace(i); del self.b1[i]; self.t2[i] = 1; return False
        if i in self.b2:
            self.p = max(0, self.p - max(1, len(self.b1) // max(1, len(self.b2))))
            self._replace(i); del self.b2[i]; self.t2[i] = 1; return False
        if len(self.t1) + len(self.b1) == c:
            if len(self.t1) < c: self.b1.popitem(last=False); self._replace(i)
            else: self.t1.popitem(last=False)
        elif len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2) >= c:
            if len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2) >= 2 * c: self.b2.popitem(last=False)
            self._replace(i)
        self.t1[i] = 1
        return False
    def end_window(self): pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--vram-slots", type=int, default=1700)
    ap.add_argument("--ram-gib", type=float, default=25)
    ap.add_argument("--split", type=float, default=0.5)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    bb = sim.blob_bytes()
    wins = sim.load_trace(a.trace)
    k = int(len(wins) * a.split)
    c = np.zeros(NL * NE)
    for T, acc, sfx, layers in wins[:k]:
        for l, ids in layers.items(): np.add.at(c, ids.astype(np.int64) + l * NE, 1)
    rank = sim.ranking_from_counts(c, bb)
    rank += [i for i in range(NL * NE) if c[i] == 0]
    vram, _ = sim.build_tiers(rank, bb, a.vram_slots, 0, True)
    seed = [i for i in rank if not vram[i]]
    cap = int(a.ram_gib * 2**30 / (float(bb.mean()) + 2048))      # packed-size slots on average
    test = window_sets(wins[k:], vram)
    acc_tok = sum(t for t, _ in test)
    pols = {
        "static": lambda: Static(seed, cap),
        "static+lru4": lambda: StaticLRU(seed, cap, int(cap * 4 / a.ram_gib)),
        "static+lru8": lambda: StaticLRU(seed, cap, int(cap * 8 / a.ram_gib)),
        "static+lru12": lambda: StaticLRU(seed, cap, int(cap * 12 / a.ram_gib)),
        "lru(seeded)": lambda: LRU(seed, cap),
        "arc(seeded)": lambda: ARC(seed, cap),
        "lfu-decay8": lambda: DecayLFU(seed, cap, 8.0, 8),
        "lfu-decay64": lambda: DecayLFU(seed, cap, 8.0, 64),
    }
    for name, mk in pols.items():
        if a.only and name not in a.only.split(","): continue
        p = mk(); miss = 0
        for t, s in test:
            for i in s:
                if not p.access(i): miss += 1
            p.end_window()
        print(f"{name:14s} miss/tok {miss / acc_tok:6.1f}   ({len(test)} windows, {acc_tok} tok, cap {cap})", flush=True)


if __name__ == "__main__":
    main()
