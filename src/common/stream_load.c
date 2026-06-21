/* stream_load.c — controlled CPU memory-bandwidth load (STREAM-style) for Exp 2.
 *
 * Generates a sustained, intensity-controlled CPU->DRAM bandwidth load so we can measure
 * how it contends with GPU decode on Apple Silicon's shared unified-memory bus.
 *
 *   usage: ./stream_load <seconds> <n_threads> <array_mb> <duty_pct>
 *   prints: "GBPS <achieved-average>" plus BYTES/ELAPSED/SINK.
 *
 * Why duty-cycle intensity (not thread count): on the M4 a SINGLE P-core already
 * saturates ~92 GB/s of the ~120 GB/s bus, so more threads don't raise aggregate
 * bandwidth. Instead one P-core thread runs the kernel then sleeps a calibrated fraction;
 * duty_pct in (0,100] sweeps the achieved AVERAGE bandwidth from ~0 to ~92 GB/s. Using a
 * single thread also leaves the other P-cores free for the GPU-dispatch thread, isolating
 * bandwidth contention from P-core scheduling pressure. GBPS is bytes/WALL-time, i.e. the
 * true average load (what Exp2 plots decode against).
 *
 * Controls (RESEARCH.md §6): worker raises QoS to USER_INTERACTIVE (macOS P-core bias;
 * no hard affinity exists). arrays >> LLC (default 96 MB vs 16 MB L2) so traffic hits DRAM.
 * Pointer rotation makes pass N+1 depend on pass N (stops the compiler hoisting the loop);
 * the kernel is a bounded mean (a = 0.5*b + 0.5*c) so values stay in range (no inf/denorm).
 * Bytes counted with the STREAM convention 24 B/element (2 read + 1 write, ignores RFO).
 */
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <pthread/qos.h>
#include <time.h>

typedef struct {
    long n;            /* elements per array */
    double seconds;    /* run duration (wall) */
    double duty;       /* active duty cycle, percent (0,100] */
    double *a, *b, *c;
    double bytes;      /* out: bytes moved (active passes only) */
    double sink;       /* out: anti dead-code-elimination */
} targ;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void sleep_s(double s) {
    if (s <= 0) return;
    struct timespec rq;
    rq.tv_sec = (time_t)s;
    rq.tv_nsec = (long)((s - (double)rq.tv_sec) * 1e9);
    nanosleep(&rq, NULL);
}

static void *worker(void *p) {
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);  /* bias to P-cores */
    targ *t = (targ *)p;
    long n = t->n;
    double *a = t->a, *b = t->b, *c = t->c, *tmp;
    double t0 = now_s();
    long passes = 0;
    while (now_s() - t0 < t->seconds) {
        double p0 = now_s();
        for (long i = 0; i < n; i++) a[i] = 0.5 * b[i] + 0.5 * c[i];   /* bounded mean */
        tmp = c; c = b; b = a; a = tmp;     /* rotate: pass N+1 depends on pass N */
        passes++;
        if (t->duty < 100.0) {              /* throttle to a target average bandwidth */
            double pass_t = now_s() - p0;
            sleep_s(pass_t * (100.0 - t->duty) / t->duty);
        }
    }
    t->bytes = (double)passes * (double)n * 24.0;   /* 24 B/elem (STREAM convention) */
    t->sink = b[0] + b[n - 1];
    return NULL;
}

int main(int argc, char **argv) {
    double seconds = argc > 1 ? atof(argv[1]) : 3.0;
    int nthreads   = argc > 2 ? atoi(argv[2]) : 1;
    long mb        = argc > 3 ? atol(argv[3]) : 96;
    double duty    = argc > 4 ? atof(argv[4]) : 100.0;
    if (nthreads <= 0 || duty <= 0.0) { printf("GBPS 0.000\nBYTES 0\nELAPSED 0\nSINK 0\n"); return 0; }
    if (nthreads > 64) nthreads = 64;
    if (duty > 100.0) duty = 100.0;

    long n = (mb * 1024L * 1024L) / (long)sizeof(double);
    pthread_t th[64];
    targ ta[64];
    for (int k = 0; k < nthreads; k++) {
        ta[k].n = n; ta[k].seconds = seconds; ta[k].duty = duty;
        ta[k].a = malloc(n * sizeof(double));
        ta[k].b = malloc(n * sizeof(double));
        ta[k].c = malloc(n * sizeof(double));
        if (!ta[k].a || !ta[k].b || !ta[k].c) { fprintf(stderr, "alloc fail\n"); return 1; }
        for (long i = 0; i < n; i++) { ta[k].b[i] = 1.0; ta[k].c[i] = 2.0; ta[k].a[i] = 0.0; }
        ta[k].bytes = 0; ta[k].sink = 0;
    }
    double t0 = now_s();
    for (int k = 0; k < nthreads; k++) pthread_create(&th[k], NULL, worker, &ta[k]);
    for (int k = 0; k < nthreads; k++) pthread_join(th[k], NULL);
    double elapsed = now_s() - t0;

    double total = 0, sink = 0;
    for (int k = 0; k < nthreads; k++) { total += ta[k].bytes; sink += ta[k].sink; }
    printf("GBPS %.3f\nBYTES %.0f\nELAPSED %.3f\nSINK %.3e\n",
           total / 1e9 / elapsed, total, elapsed, sink);
    return 0;
}
