import os, threading, time, ctypes
libc = ctypes.CDLL(None, use_errno=True)
PATH = "/home/bodhi/models/strata/packs/heretic-iq3xxs/experts-native.bin"
sz = os.path.getsize(PATH)
def fadv_dontneed(fd, off, ln):
    libc.posix_fadvise(fd, ctypes.c_long(off), ctypes.c_long(ln), ctypes.c_int(4))
def run(nthreads, block_mb, dontneed=False):
    bs = block_mb * 1024 * 1024
    fds = [os.open(PATH, os.O_RDONLY | os.O_DIRECT) for _ in range(nthreads)]
    total = [0]*nthreads
    def worker(w):
        fd = fds[w]; n = 0; buf = bytearray(bs)
        off = w * bs
        while off + bs <= sz:
            n += os.preadv(fd, [buf], off)
            if dontneed: fadv_dontneed(fd, off, bs)
            off += nthreads * bs
        total[w] = n
    ths = [threading.Thread(target=worker, args=(w,)) for w in range(nthreads)]
    t0 = time.perf_counter()
    [t.start() for t in ths]; [t.join() for t in ths]
    dt = time.perf_counter() - t0
    [os.close(f) for f in fds]
    gb = sum(total)/1e9
    print(f"seq QD{nthreads:2d} x{block_mb:2d}MB {'DONTNEED' if dontneed else 'cached  '}: {gb:.1f} GB in {dt:5.1f}s = {gb/dt:5.2f} GB/s")
for n, mb in [(1,16),(2,8),(4,4),(8,4),(12,2),(16,2),(24,2)]:
    run(n, mb)
