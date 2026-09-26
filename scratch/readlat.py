#!/usr/bin/env python3
"""Cold random-read latency microbench on the expert arena (as the engine's ring sees it)."""
import os, random, time

PATH = "/home/bodhi/models/strata/packs/heretic-iq3xxs/experts-native.bin"
BLOB = 2080 * 1024
N = 200
sz = os.path.getsize(PATH)
fd = os.open(PATH, os.O_RDONLY)
rnd = random.Random(1234)
offs = [rnd.randrange(0, sz - BLOB) // 512 * 512 for _ in range(N)]
buf = os.pread  # noqa
import ctypes
libc = ctypes.CDLL(None, use_errno=True)
FADV_DONTNEED = 4
def _fadvise(fd, off, ln, adv=4):
    libc.posix_fadvise(fd, ctypes.c_long(off), ctypes.c_long(ln), ctypes.c_int(adv))
os.posix_fadvise = _fadvise

def one(off, dontneed=True):
    b = bytearray(BLOB)
    t0 = time.perf_counter()
    n = os.preadv(fd, [b], off) if hasattr(os, "preadv") else os.pread(fd, BLOB, off)
    dt = time.perf_counter() - t0
    if dontneed:
        os.posix_fadvise(fd, off, n)
    return dt

# QD1 cold
ts = [one(o) for o in offs]
ts.sort()
print(f"QD1 cold:  mean {1000*sum(ts)/len(ts):.2f} ms  p50 {1000*ts[len(ts)//2]:.2f}  p95 {1000*ts[int(len(ts)*0.95)]:.2f}  max {1000*ts[-1]:.2f}")

# QD1 with page cache (no DONTNEED)
ts2 = [one(o, dontneed=False) for o in offs[:50]]
print(f"QD1 2nd (cached-or-not): mean {1000*sum(ts2)/len(ts2):.2f} ms")

# threaded QD8
import threading
def worker(chunk, out):
    for o in chunk:
        out.append(one(o))
chunks = [offs[i::8] for i in range(8)]
out = []
t0 = time.perf_counter()
ths = [threading.Thread(target=worker, args=(c, out)) for c in chunks]
[t.start() for t in ths]; [t.join() for t in ths]
dt = time.perf_counter() - t0
print(f"QD8: {len(out)} reads in {dt*1000:.0f} ms -> {len(out)/dt:.0f} reads/s, {len(out)*BLOB/dt/1e9:.2f} GB/s")
