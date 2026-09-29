"""R16: where do the misses live?  Answers two questions the tier design needs:
  1. per-layer: if we hold the top-K non-resident experts for a layer in a side buffer, what share of
     that layer's tail traffic do they capture?  (a "miss cache" prefetched during idle drive time)
  2. global: bytes/token as a function of resident blob count, on the exact counts the engine sees.
"""
import struct, sys
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "data/counts-r10.strc"
BLOB = 2176000
with open(path, "rb") as f:
    blob = f.read()
assert blob[:4] == b"STRC", blob[:4]
nl, ne, _ = struct.unpack("<3I", blob[4:16])
c = np.frombuffer(blob, dtype="<u8", count=nl * ne, offset=16).astype(np.float64).reshape(nl, ne)
total = c.sum()
print(f"{path}: {nl}x{ne}, {total:.0f} routed entries, mean {c.mean():.1f}/expert")

flat = c.reshape(-1)
order = np.argsort(-flat)
for n_res in (12000, 14000, 16000, 18000):
    if n_res >= flat.size:
        continue
    resident = np.zeros(flat.size, bool)
    resident[order[:n_res]] = True
    cov = flat[resident].sum() / total * 100
    miss_bytes = flat[~resident].sum() * BLOB
    print(f"  resident {n_res:6d} ({n_res*BLOB/2**30:5.2f} GiB): coverage {cov:5.2f}%  "
          f"miss {100-cov:5.2f}% = {miss_bytes/1e9:6.1f} GB per 6.03M requests")

# per-layer tail concentration: for each layer, hold the top-K non-resident experts in RAM
for K in (4, 8, 12, 16, 32):
    captured = 0.0
    buf_bytes = 0
    for l in range(nl):
        row = c[l]
        o = np.argsort(-row)
        if K >= ne:
            continue
        tail = row[o[K:]]
        cap = row[o[:K]].sum()
        # "non-resident" proxy: the tail below rank K within the layer (a strict upper bound on what a
        # side buffer of K blobs could capture for THIS layer)
        captured += cap
        buf_bytes += K * BLOB
    print(f"  side buffer top-{K:2d}/layer ({buf_bytes/2**30:5.2f} GiB): captures {captured/total*100:5.2f}% "
          f"of ALL layer traffic (upper bound; resident set already covers much of it)")

# realistic version: global top-N resident (the real tier), then per-layer side buffer for misses
for n_res in (12000, 14000):
    resident = np.zeros(flat.size, bool)
    resident[order[:n_res]] = True
    res = resident.reshape(nl, ne)
    for K in (4, 8, 16):
        cap = 0.0
        for l in range(nl):
            row = c[l].copy()
            row[res[l]] = -1.0          # resident experts are not part of the miss population
            o = np.argsort(-row)
            cap += row[o[:K]].sum()
        miss_total = c[~res].sum()
        print(f"  resident {n_res} + side top-{K}/layer: captures {cap/miss_total*100:5.1f}% of misses "
              f"({cap/total*100:5.2f}% of all traffic)")
