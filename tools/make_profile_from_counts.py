#!/usr/bin/env python3
"""make_profile_from_counts.py - turn a `strata --dump-counts` histogram into a STRP routing profile.

`--dump-counts` writes `STRC`, then `u32 n_layers, u32 n_expert, u32 pad`, then `n_layers * n_expert`
uint64 routing counts.  `--expert-profile` wants `STRP`, then `u32 version, n_layers, n_expert, slots,
n_ranked`, then `n_ranked` (u16 layer, u16 expert) pairs by DESCENDING frequency.

The difference matters on a small-RAM box: a profile that only carries a RANK cannot say how much of the
traffic the top N experts actually carry, and that coverage curve is the number every tier size has to be
chosen against.  This tool prints it, so the tier is sized from measurement.

  python3 tools/make_profile_from_counts.py routing.bin expert-profile.bin --blob-bytes 2176000
"""
from __future__ import annotations

import argparse
import struct
import sys

import numpy as np


def load(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        blob = f.read()
    if blob[:4] != b"STRC":
        raise SystemExit(f"{path}: not a STRC counts file (magic {blob[:4]!r})")
    nl, ne, _ = struct.unpack("<3I", blob[4:16])
    need = nl * ne * 8
    if len(blob) - 16 < need:
        raise SystemExit(f"{path}: expected {need} B of counts, got {len(blob) - 16}")
    return nl, ne, np.frombuffer(blob, dtype="<u8", count=nl * ne, offset=16).astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("counts")
    ap.add_argument("out")
    ap.add_argument("--blob-bytes", type=int, default=2176000)
    ap.add_argument("--max-ranked", type=int, default=24000)
    a = ap.parse_args()

    nl, ne, c = load(a.counts)
    total = c.sum()
    order = np.argsort(-c)
    cumulative = np.cumsum(c[order])
    touched = int((c > 0).sum())
    print(f"{a.counts}: {nl}x{ne}, {total:.0f} routed entries, {touched} experts touched "
          f"({100.0 * touched / c.size:.1f}%)")

    # the coverage curve is the tier-sizing input: bytes resident -> fraction of routing traffic served
    print(f"  {'blobs':>7} {'GiB':>7} {'% experts':>10} {'% traffic':>10}")
    for n in (2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 24576):
        if n <= c.size:
            print(f"  {n:>7} {n * a.blob_bytes / 2**30:>7.2f} {100.0 * n / c.size:>9.1f}%"
                  f" {100.0 * cumulative[n - 1] / total:>9.2f}%")

    ranked = [int(i) for i in order if c[i] > 0][: a.max_ranked]
    with open(a.out, "wb") as f:
        f.write(b"STRP")
        f.write(struct.pack("<5I", 0, nl, ne, len(ranked), len(ranked)))
        f.write(np.array([(i // ne, i % ne) for i in ranked], dtype="<u2").tobytes())
    print(f"wrote {a.out}: {len(ranked)} ranked pairs, built for {len(ranked)} slots")
    return 0


if __name__ == "__main__":
    sys.exit(main())
