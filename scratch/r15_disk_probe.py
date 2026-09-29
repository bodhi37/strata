#!/usr/bin/env python3
"""R15 disk probe: separate three hypotheses that the campaign has been conflating.

  H1  the NVMe itself caps at ~0.5 GB/s on 2 MiB reads          (hardware limit)
  H2  btrfs compressed-inode / page-cache path caps it           (filesystem limit)
  H3  the engine's QD1 serial access pattern caps it             (software limit)

They have completely different fixes, so we measure all three in one pass:
  buffered  / O_DIRECT, at queue depth 1 and 8 and 16, at the engine's real
  blob size (2.18 MiB) and the engine's real blob offsets.

Read-only. Never writes. Safe next to a running engine, but it does contend
for the drive, so keep --secs small while an engine is live.
"""
import argparse
import os
import random
import statistics
import threading
import time

BLOB = 2181120  # 2.18 MiB, the engine's expert blob size (256B aligned)
ALIGN = 4096    # O_DIRECT alignment requirement


def diskstats():
    with open("/proc/diskstats") as fh:
        for line in fh:
            f = line.split()
            if f[2].startswith("nvme") and f[2].count("p") == 0:
                return int(f[5]), int(f[3])  # sectors read, reads completed
    return 0, 0


def bench(path, total, offsets, direct, qd, secs):
    """Read random blobs with qd concurrent threads; return stats."""
    flags = os.O_RDONLY | (os.O_DIRECT if direct else 0)
    fd = os.open(path, flags)
    stop = time.time() + secs
    mb = [0.0]
    lat = []
    lk = threading.Lock()

    def worker(seed):
        rng = random.Random(seed)
        import ctypes
        buf = (ctypes.c_char * (BLOB + ALIGN))()
        base = ctypes.addressof(buf)
        off_adj = (-base) % ALIGN
        loc_lat = []
        loc_mb = 0.0
        # O_DIRECT needs an aligned destination AND a 512B-aligned file offset.
        # Blob offsets are 256B aligned in the arena; round down to 4K and
        # read a 4K-padded window so the probe measures aligned reads.
        while time.time() < stop:
            blob = offsets[rng.randrange(len(offsets))]
            aoff = (blob // ALIGN) * ALIGN
            t0 = time.perf_counter()
            data = os.pread(fd, BLOB, aoff)
            t1 = time.perf_counter()
            n = len(data) if data else 0
            if n > 0:
                loc_lat.append((t1 - t0) * 1e3)
                loc_mb += n / 1e6
        with lk:
            lat.extend(loc_lat)
            mb[0] += loc_mb

    ths = [threading.Thread(target=worker, args=(i * 7919 + qd,))
           for i in range(qd)]
    wall0 = time.perf_counter()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.perf_counter() - wall0
    if not lat:
        return None
    return {
        "qd": qd,
        "reads": len(lat),
        "mbps": mb[0] / wall,
        "lat_med": statistics.median(lat),
        "lat_p99": sorted(lat)[max(0, int(len(lat) * 0.99) - 1)],
        "min_lat": min(lat),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--secs", type=int, default=6)
    ap.add_argument("--n", type=int, default=40000, help="blobs in sample set")
    ap.add_argument("--modes", default="direct",
                    help="comma list of direct,buffered. buffered allocates page "
                         "cache: only run it when no engine is live on this box.")
    args = ap.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    total = os.path.getsize(args.path)
    nblobs = total // BLOB
    rng = random.Random(1234)
    offsets = [rng.randrange(nblobs) * BLOB for _ in range(args.n)]

    print(f"file      : {args.path}")
    print(f"size      : {total/2**30:.1f} GiB  ({nblobs} blobs of {BLOB/2**20:.2f} MiB)")
    try:
        print(f"extents   : {os.popen('filefrag ' + args.path).read().strip().splitlines()[-1]}")
    except Exception:
        pass

    s0, r0 = diskstats()
    print("\nmode      qd   reads   MB/s    lat_med  lat_p99  lat_min   (ms)")
    for direct in (True if "direct" in modes else None,
                   False if "buffered" in modes else None):
        if direct is None:
            continue
        for qd in (1, 4, 8, 16):
            try:
                r = bench(args.path, total, offsets, direct, qd, args.secs)
            except PermissionError as e:
                print(f"{'O_DIRECT' if direct else 'buffered '} {qd:3d}   SKIP: {e}")
                continue
            if r is None:
                print(f"{'O_DIRECT' if direct else 'buffered '} {qd:3d}   no reads")
                continue
            mode = "O_DIRECT" if direct else "buffered"
            print(f"{mode:9s} {qd:3d} {r['reads']:6d}  {r['mbps']:6.0f}  "
                  f"{r['lat_med']:7.2f}  {r['lat_p99']:7.2f}  {r['min_lat']:7.2f}")

    s1, r1 = diskstats()
    print(f"\ndrive over probe: {(s1-s0)*512/2**30:.2f} GiB read, "
          f"{r1-r0} read ops, avg req {((s1-s0)*512/max(1,r1-r0))/1024:.0f} KiB")
    print("interpretation: MB/s that RISES with qd => QD1 is the limit (H3), "
          "drive has headroom.\nFlat across qd and both modes => drive/FS limit "
          "(H1/H2); buffered>O_DIRECT => btrfs is blocking the direct path (H2).")


if __name__ == "__main__":
    main()
