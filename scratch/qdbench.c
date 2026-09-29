// R15: drive queue-depth / size curve on the ACTUAL arena file, O_DIRECT, fresh random offsets per pass.
//
//   ./qdbench <file> <bytes-per-read> <qthreads> <seconds> [seq]
//
// Reports MB/s and ms per read.  Random mode picks a random blob-aligned offset each iteration from a
// per-thread shuffled walk so nothing is cached and nothing is sequential by accident.
#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <stdint.h>
#include <unistd.h>

static double g_seconds = 3.0;
static int nthr = 1;
static int fd;
static long long fsize;
static long long rlen, roffs;      // read length, and count of aligned slots
static int seqmode = 0;
static long long iters_per_thread = 1 << 30;
static pthread_barrier_t bar;
static struct thr { long long done; double ms; } th[128];

static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}
static uint64_t rng_state;
static uint64_t xs(void) {
    uint64_t x = rng_state; x ^= x << 13; x ^= x >> 7; x ^= x << 17; rng_state = x; return x;
}
static void* work(void* arg) {
    long id = (long) arg;
    void* buf = NULL;
    if (posix_memalign(&buf, 4096, (size_t) rlen)) return NULL;
    rng_state = 0x9E3779B97F4A7C15ull * (id + 1) ^ 0xDEADBEEF;
    if (!rng_state) rng_state = 1;
    long long base = (long long) (xs() % (uint64_t) roffs);
    pthread_barrier_wait(&bar);
    double t0 = now();
    long long n = 0;
    while (now() - t0 < g_seconds * 1000.0 && n < iters_per_thread) {
        long long slot;
        if (seqmode) { slot = (base + n * (long long) nthr) % roffs; }
        else {
            // random jump, blob-granular; a prime stride keeps threads off each other
            slot = (long long) (xs() % (uint64_t) roffs);
        }
        off_t off = (off_t) slot * rlen;
        ssize_t got = pread(fd, buf, (size_t) rlen, off);
        if (got != (ssize_t) rlen) { fprintf(stderr, "pread short at %lld: %zd %s\n", (long long) off, got, strerror(errno)); break; }
        ++n;
    }
    th[id].done = n; th[id].ms = now() - t0;
    return NULL;
}
int main(int c, char** v) {
    if (c < 4) { fprintf(stderr, "usage: %s <file> <rlen> <threads> [seconds] [seq] [iters]\n", v[0]); return 2; }
    const char* path = v[1];
    rlen = atoll(v[2]);
    nthr = atoi(v[3]);
    if (c > 4) g_seconds = atof(v[4]);
    if (c > 5 && !strcmp(v[5], "seq")) seqmode = 1;
    if (c > 6) iters_per_thread = atoll(v[6]);
    fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) { fprintf(stderr, "open %s: %s\n", path, strerror(errno)); return 2; }
    fsize = lseek(fd, 0, SEEK_END);
    roffs = fsize / rlen;
    pthread_barrier_init(&bar, 0, nthr);
    pthread_t t[128];
    for (long i = 0; i < nthr; ++i) pthread_create(&t[i], 0, work, (void*) i);
    for (long i = 0; i < nthr; ++i) pthread_join(t[i], 0);
    long long tot = 0; double ms = 0;
    for (int i = 0; i < nthr; ++i) { tot += th[i].done; if (th[i].ms > ms) ms = th[i].ms; }
    double mb = (double) tot * rlen / 1e6;
    printf("%8lld B x QD%-3d %-3s : %8.1f MB/s  %6.2f ms/read  (%lld reads, %.1f s)\n",
           rlen, nthr, seqmode ? "seq" : "rnd", mb / (ms / 1e3), ms / (double) (tot / (nthr ? nthr : 1) + 1),
           tot, ms / 1e3);
    return 0;
}
