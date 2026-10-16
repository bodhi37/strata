#!/usr/bin/env python3
"""calls.py trace.bin - summarize STRATA_CALL_TRACE (per pool call timings)."""
import numpy as np, sys
a = np.fromfile(sys.argv[1], dtype=np.float32); a = a[:len(a) // 10 * 10].reshape(-1, 10)
L, nt, wait, plan, actq, p0, rw, p1, nj, nm = a.T
n = len(a)
win = (L == 0).sum()   # approx calls at layer 0 (x groups)
print(f"{n} calls; per call ms: wait {wait.mean():.3f} plan {plan.mean():.3f} actq {actq.mean():.3f} pass0 {p0.mean():.3f} "
      f"ringwait {rw.mean():.3f} pass1 {p1.mean():.3f}; cpu entries {nj.mean():.1f} miss entries {nm.mean():.2f}")
tot = wait + plan + actq + p0 + rw + p1
print(f"sum per call {tot.mean():.3f} ms")
for lo, hi in ((0, 0), (1, 1), (2, 2), (3, 4), (5, 100)):
    m = (nm >= lo) & (nm <= hi)
    if m.sum() == 0: continue
    print(f"  misses {lo}-{hi}: {m.mean() * 100:5.1f}% of calls  wait {wait[m].mean():.3f} pass0 {p0[m].mean():.3f} ringwait {rw[m].mean():.3f} pass1 {p1[m].mean():.3f}  total {tot[m].mean():.3f}")
print("share of total time:", {k: round(float(v.sum() / tot.sum()) * 100, 1) for k, v in dict(wait=wait, plan=plan, actq=actq, pass0=p0, ringwait=rw, pass1=p1).items()})
