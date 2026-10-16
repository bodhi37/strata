#!/usr/bin/env python3
"""belady.py - clairvoyant (optimal) eviction over VRAM-static + a fully dynamic RAM cache: the policy ceiling."""
import sys, os, heapq, struct
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim
trace, prof, vslots, ram_gib = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
bb = sim.blob_bytes()
b = open(prof, "rb").read(); _, nl, ne, sl, nr = struct.unpack("<5I", b[4:24])
rank = [int(l) * 512 + int(e) for l, e in np.frombuffer(b, "<u2", 2 * nr, 24).reshape(-1, 2)]
vram, _ = sim.build_tiers(rank, bb, vslots, 0, True)
wins = sim.load_trace(trace)
seq = []   # (window, set of idx)
acc = 0
for w, (T, a, sfx, layers) in enumerate(wins):
    acc += a + 1
    s = set()
    for l, ids in layers.items():
        for i in np.unique(ids.astype(np.int64) + l * 512):
            if not vram[i]: s.add(int(i))
    seq.append(s)
# next-use index per access
nxt = {}
future = [dict() for _ in seq]
last = {}
for w in range(len(seq) - 1, -1, -1):
    for i in seq[w]:
        future[w][i] = last.get(i, 1 << 30)
        last[i] = w
cap_slots = int(ram_gib * 2**30 / 2.05e6)
cache = {}; heap = []; miss = 0
for w, s in enumerate(seq):
    for i in s:
        nu = future[w][i]
        if i not in cache:
            miss += 1
        cache[i] = nu
        heapq.heappush(heap, (-nu, i))
    while len(cache) > cap_slots:
        nu, i = heapq.heappop(heap)
        if i in cache and cache[i] == -nu: del cache[i]
print(f"Belady: RAM {ram_gib} GiB (~{cap_slots} blobs) + VRAM {vslots}: miss/tok {miss/acc:.1f} over {acc} tok")
