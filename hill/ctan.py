#!/usr/bin/env python3
"""ctan.py CALLTRACE [windows-ms] : per pool call {layer, n_tok, gpu_wait, plan, actq, pass0, ringwait, pass1, cpu_jobs, misses}"""
import sys
import numpy as np
r = np.fromfile(sys.argv[1], dtype="<f4"); r = r[:len(r) // 10 * 10].reshape(-1, 10)
r = r[r[:, 1] <= 16]        # decode calls only (prompt chunks are thousands of tokens)
L, nt, gw, pl, aq, p0, rw, p1, nj, nm = r.T
n = len(r)
print(f"{n} decode pool calls, n_tok mean {nt.mean():.2f}, cpu entries/call {nj.mean():.1f}, misses/call {nm.mean():.2f}")
tot = gw + pl + aq + p0 + rw + p1
for name, v in [("gpu_wait", gw), ("plan", pl), ("actq", aq), ("pass0 resident", p0), ("ringwait", rw), ("pass1 missed", p1)]:
    print(f"  {name:15s} mean {v.mean()*1000:7.1f} us   total share {v.sum()/tot.sum()*100:5.1f}%")
has = nm > 0
print(f"calls with a miss: {has.mean()*100:.1f}%  ringwait when missing: mean {rw[has].mean()*1000:.0f} us, p50 {np.median(rw[has])*1000:.0f}, p90 {np.percentile(rw[has],90)*1000:.0f}")
print(f"  pass0 when missing {p0[has].mean()*1000:.0f} us, pass1 {p1[has].mean()*1000:.0f} us; misses per missing call {nm[has].mean():.2f}")
for k in (1, 2, 3, 4):
    m = nm == k
    if m.any(): print(f"  {k} miss(es): {m.sum()} calls, ringwait mean {rw[m].mean()*1000:.0f} us, pass0 {p0[m].mean()*1000:.0f} us")
