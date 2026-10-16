#!/usr/bin/env python3
"""sim3.py - cache-aware routing: a routed expert at top-k rank >= R is DROPPED when it would miss.
Tiers as the engine's hybrid: VRAM per-layer + packed static + exact LRU (equal slots per layer)."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim import NL, NE, blob_bytes, load_trace, ranking_from_counts, build_tiers
from sim2 import FreqLRU, GiB
tr = sys.argv[1]; total = float(sys.argv[2]); lru_g = float(sys.argv[3]); Rs = [int(x) for x in sys.argv[4].split(",")]
bb = blob_bytes(); wins = load_trace(tr); cut = len(wins) // 2; train, test = wins[:cut], wins[cut:]
c = np.zeros(NL * NE)
for (T, a, s, L) in train:
    for l, ids in L.items(): np.add.at(c, ids.astype(np.int64) + l * NE, 1)
rank = ranking_from_counts(c)
vram, ram = build_tiers(rank, bb, 1700, int((total - lru_g) * GiB), True)
nl = {}
for x in bb: nl[int(x)] = nl.get(int(x), 0) + 1
den = sum(n * ((b + 4095) // 4096 * 4096) for b, n in nl.items())
for R in Rs:
    pools = {b: FreqLRU(int(lru_g * GiB * n * ((b + 4095) // 4096 * 4096) / den // ((b + 4095) // 4096 * 4096))) for b, n in nl.items()}
    miss = acc_tok = dropped = routed = 0
    for ep, (T, acc, sfx, L) in enumerate(test):
        acc_tok += (acc + 1) if acc >= 0 else 1
        for l, ids in L.items():
            a = ids.reshape(-1, 10).astype(np.int64) + l * NE
            routed += a.size
            must = set(np.unique(a[:, :R]).tolist())
            opt = set(np.unique(a[:, R:]).tolist()) - must
            pool = pools[int(bb[l])]
            for idx in must:
                if vram[idx] or ram[idx] or pool.access(idx, ep): continue
                miss += 1
            for idx in opt:
                if vram[idx] or ram[idx]: continue
                if idx in pool.od: pool.access(idx, ep); continue
                dropped += 1
    print(f"R={R:2d}: miss/tok {miss/acc_tok:5.2f}  dropped expert-uses/tok {dropped/acc_tok:5.2f}  (routed/tok {routed/acc_tok:.0f})", flush=True)
