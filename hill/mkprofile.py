#!/usr/bin/env python3
"""mkprofile.py - STRP expert profile from one or more STRC count files (summed, each normalised first).

  hill/mkprofile.py OUT.bin data/counts-a.strc[,data/counts-b.strc] [--density] [--fill-all]

--density ranks by routed count per blob byte (the greedy knapsack order for byte-budgeted tiers; this pack's
blobs run 1.5-2.7 MB by layer).  --fill-all appends never-seen experts (smallest first) so the tiers can always
fill to capacity instead of stopping at the last observed pair.
"""
import argparse, os, struct, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim import blob_bytes, load_counts, NL, NE  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("counts")
ap.add_argument("--density", action="store_true")
ap.add_argument("--fill-all", action="store_true")
a = ap.parse_args()
bb = blob_bytes()
c = np.zeros(NL * NE)
for p in a.counts.split(","):
    x = load_counts(p)
    c += x / x.sum()
key = c / np.repeat(bb, NE) if a.density else c
order = np.argsort(-key, kind="stable")
rank = [int(i) for i in order if c[i] > 0]
if a.fill_all:
    rest = [int(i) for i in np.argsort(np.repeat(bb, NE), kind="stable") if c[i] == 0]
    rank += rest
with open(a.out, "wb") as f:
    f.write(b"STRP")
    f.write(struct.pack("<5I", 0, NL, NE, len(rank), len(rank)))
    f.write(np.array([(i // NE, i % NE) for i in rank], dtype="<u2").tobytes())
print(f"wrote {a.out}: {len(rank)} pairs ({int((c > 0).sum())} observed)")
