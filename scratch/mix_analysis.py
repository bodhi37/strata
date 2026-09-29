import struct, sys
import numpy as np
p = sys.argv[1] if len(sys.argv)>1 else "data/counts-r16.strc"
blob = open(p,'rb').read()
nl, ne, _ = struct.unpack("<3I", blob[4:16])
c = np.frombuffer(blob, dtype="<u8", count=nl*ne, offset=16).astype(np.float64)
total = c.sum()
print(f"{nl}x{ne} entries={total:.0f} touched={(c>0).sum()}")
order = np.argsort(-c); cs = c[order]; cum = np.cumsum(cs)
B32 = 2176000            # IQ3_XXS blob bytes (from make_profile default)
# measure the true iq2xxs blob size ratio from the pack index later; assume bpw ratio
for name, b3, b2 in [("IQ2_XXS 1.72bpw", 3.44, 1.72), ("IQ2_XS 2.31bpw", 3.44, 2.31), ("IQ2_S 2.6bpw",3.44,2.60)]:
    s3 = b3/8.0*4915200.0
    s2 = b2/8.0*4915200.0
    for budget_gib in (26.0, 28.0, 30.0):
        B = int(budget_gib*2**30)
        best = None
        for h in range(0, nl*ne+1, 256):
            hb = h*s3
            if hb > B: break
            m = min(int((B-hb)/s2), nl*ne-h)
            cov = (cum[h+m-1] if h+m>0 else 0.0)/total
            if best is None or cov > best[0]: best = (cov, h, m)
        cov,h,m = best
        # miss bytes per token for 281 requests/token under this pack
        nres = nl*ne-h-m
        missfrac = 1.0 - (cum[h+m-1] if h+m>0 else 0.0)/total
        print(f"{name:>16} budget {budget_gib:5.1f} GiB -> best h={h:6d} iq3 + {m:6d} iq2 resident "
              f"cov={100*cov:6.2f}% missfrac={100*missfrac:5.2f}% missMB/tok@281req={missfrac*281*s2/2**20:6.2f}")
print("--- all-IQ3_XXS reference ---")
for budget_gib in (26.0, 28.0):
    n = int(budget_gib*2**30/ (3.44/8*4915200.0))
    cov = cum[n-1]/total; miss = 1-cov
    print(f"budget {budget_gib} GiB -> {n} blobs cov={100*cov:.2f}% missMB/tok={miss*281*2.176e6/2**20:.2f}")
