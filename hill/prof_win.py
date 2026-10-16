#!/usr/bin/env python3
"""prof_win.py - per-window GPU timeline breakdown from an nsys sqlite export (kernel trace incl. graph nodes)."""
import sqlite3, sys, collections, re
db = sys.argv[1]
c = sqlite3.connect(db)
names = dict(c.execute("select id, value from StringIds"))
ks = c.execute("select start, end, shortName, streamId, gridX, blockX from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
def short(n):
    n = names.get(n, str(n))
    return n
print(len(ks), "kernels; streams:", collections.Counter(k[3] for k in ks).most_common(5))
# windows: split at sampler_kernel (one per window) on the main stream
win = []
cur = []
for k in ks:
    cur.append(k)
    if 'sampler_kernel' in short(k[2]):
        win.append(cur); cur = []
print(len(win), "windows")
agg = collections.Counter(); wall = []; busy = []; waitt = []
cnt = 0
for w in win[5:-2]:
    t0, t1 = w[0][0], w[-1][1]
    wall.append((t1 - t0) / 1e6)
    b = 0; wt = 0
    for k in w:
        d = (k[1] - k[0]) / 1e6
        n = short(k[2])
        if 'wait_flag' in n: wt += d
        else: b += d
        agg[n] += d
    busy.append(b); waitt.append(wt); cnt += 1
import statistics as st
print(f"per window: wall {st.mean(wall):.1f} ms, kernel-busy {st.mean(busy):.1f} ms, wait_flag {st.mean(waitt):.1f} ms (n={cnt})")
for n, v in agg.most_common(40):
    print(f"  {v / cnt:7.3f} ms/win  {n[:100]}")
# big gaps (>= 50 us) by neighbouring kernels
pairs = collections.Counter(); tot = collections.Counter()
for w in win[5:-2]:
    for a, b in zip(w, w[1:]):
        g = (b[0] - a[1]) / 1e3
        if g >= 50:
            key = (short(a[2])[:36], short(b[2])[:36], a[3], b[3]); pairs[key] += 1; tot[key] += g
print("big gaps:")
for k_, v in tot.most_common(8): print(f"  {v / cnt / 1e3:6.2f} ms/win n/win={pairs[k_] / cnt:4.1f} {k_}")
