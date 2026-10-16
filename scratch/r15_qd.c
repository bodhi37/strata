// r15_qd: zero-copy queue-depth sweep of the expert arena.
//
// The campaign's disk numbers came from Python, which allocates and memcpy's
// every 2 MiB read into a fresh `bytes` object - so its MB/s ceiling was the
// interpreter's, not the drive's. This reads straight into reused,
// O_DIRECT-aligned buffers from N threads, so MB/s here is what the engine
// could actually reach at that many concurrent readers.
//
// Reports both throughput and the latency distribution, because decode and
// prefill want opposite ends of that tradeoff:
//   decode  = latency at low QD (a miss blocks its layer)
//   prefill = throughput at high QD (thousands of independent blobs)
//
// build: gcc -O2 -pthread -o r15_qd r15_qd.c -lrt
// run:   ./r15_qd <arena.bin> [blob_bytes] [secs_per_arm]
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define ALIGN 4096
#define MAXTHREADS 96

static int g_fd = -1;
static long long g_nblobs = 0;
static long long g_blob = 2181120;
static volatile int g_stop = 0;
static int g_sequential = 0;

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

struct Stat { long long n; double bytes, min, sum; double *lat; long long nl; };
static struct Stat g_st[MAXTHREADS];

// comparison for qsort on doubles
static int cmpd(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static void *worker(void *arg) {
    long id = (long)arg;
    void *buf = NULL;
    if (posix_memalign(&buf, ALIGN, (size_t)g_blob)) return NULL;
    struct Stat *s = &g_st[id];
    s->min = 1e18;
    long long cap = 200000;
    s->lat = malloc(cap * sizeof(double));
    s->nl = 0;
    unsigned seed = (unsigned)(id * 2654435761u + 12345u);
    long long seq = id;                       // per-thread stride for "sequential"
    while (!g_stop) {
        long long b;
        if (g_sequential) {
            b = __sync_fetch_and_add(&seq, 1) % g_nblobs;   // locally increasing
        } else {
            seed = seed * 1103515245u + 12345u;
            b = (long long)((seed >> 8) % (unsigned)g_nblobs);
        }
        long long off = (b * g_blob) / ALIGN * ALIGN;
        double t0 = now_ms();
        ssize_t n = pread(g_fd, buf, (size_t)g_blob, off);
        double dt = now_ms() - t0;
        if (n <= 0) { if (errno == EINTR) continue; break; }
        s->n++; s->bytes += (double)n; s->sum += dt;
        if (dt < s->min) s->min = dt;
        if (s->nl < cap) s->lat[s->nl++] = dt;
    }
    free(buf);
    return NULL;
}

static void arm(const char *path, int nthr, int secs, int sequential, long long blob) {
    g_blob = blob;
    g_sequential = sequential;
    g_stop = 0;
    memset(g_st, 0, sizeof(g_st));
    g_fd = open(path, O_RDONLY | O_DIRECT);
    if (g_fd < 0) { perror("open"); return; }

    pthread_t th[MAXTHREADS];
    double t0 = now_ms();
    for (long i = 0; i < nthr; i++) pthread_create(&th[i], NULL, worker, (void *)i);
    struct timespec sl = { secs, 0 };
    nanosleep(&sl, NULL);
    g_stop = 1;
    for (int i = 0; i < nthr; i++) pthread_join(th[i], NULL);
    double wall = now_ms() - t0;
    close(g_fd);

    long long n = 0; double bytes = 0, sum = 0, mn = 1e18;
    static double all[400000]; long long na = 0;
    for (int i = 0; i < nthr; i++) {
        n += g_st[i].n; bytes += g_st[i].bytes; sum += g_st[i].sum;
        if (g_st[i].min < mn) mn = g_st[i].min;
        for (long long j = 0; j < g_st[i].nl && na < 400000; j++) all[na++] = g_st[i].lat[j];
        free(g_st[i].lat); g_st[i].lat = NULL;
    }
    if (!n) { printf("  QD%-3d  no reads\n", nthr); return; }
    qsort(all, na, sizeof(double), cmpd);
    double med = all[na / 2], p10 = all[(long)(na * 0.10)], p90 = all[(long)(na * 0.90)];
    double mbs = bytes / (wall / 1e3) / 1e6;
    // concurrency implied by Little's law: ops in flight = QD, so
    // effective per-op service time = wall*n ... we report mean latency too.
    printf("  %-9s QD%-3d %8.0f MB/s  lat p10 %6.2f med %6.2f p90 %6.2f min %5.2f ms   %lld reads\n",
           sequential ? "seq" : "random", nthr, mbs, p10, med, p90, mn, n);
    (void)sum;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <arena.bin> [blob] [secs]\n", argv[0]); return 2; }
    long long blob = argc > 2 ? atoll(argv[2]) : 2181120;
    int secs = argc > 3 ? atoi(argv[3]) : 5;
    struct stat sb;
    if (stat(argv[1], &sb)) { perror("stat"); return 2; }
    g_nblobs = sb.st_size / blob;
    printf("arena %s  %.1f GiB  %lld blobs of %.2f MiB   (O_DIRECT, zero-copy)\n",
           argv[1], sb.st_size / 1073741824.0, g_nblobs, blob / 1048576.0);

    printf("\n-- random blob reads (decode's access pattern) --\n");
    int qs[] = {1, 2, 4, 8, 16, 32, 48, 64};
    for (unsigned i = 0; i < sizeof(qs) / sizeof(qs[0]); i++)
        arm(argv[1], qs[i], secs, 0, blob);

    printf("\n-- sequential-ish reads (prefill sweeps in file-offset order) --\n");
    int ss[] = {1, 8, 32};
    for (unsigned i = 0; i < sizeof(ss) / sizeof(ss[0]); i++)
        arm(argv[1], ss[i], secs, 1, blob);

    printf("\n-- small reads, 128 KiB (== max_hw_sectors_kb: one segment) --\n");
    for (unsigned i = 0; i < sizeof(qs) / sizeof(qs[0]); i++)
        arm(argv[1], qs[i], secs, 0, 131072);
    return 0;
}
