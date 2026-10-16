// aggregate DRAM read bandwidth, N threads, scattered 2 MB pages (mimics the expert tier)
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>
static unsigned char *big; static long long GB=12, PGSZ=2*1024*1024, npg, iters=4;
static int nth; static double t_ms;
static double now(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec*1e3+ts.tv_nsec/1e6;}
static void* work(void* arg){
  long id=(long)arg; unsigned long long acc=0;
  for(int it=0; it<iters; ++it)
    for(long long p=id; p<npg; p+=nth){
      const unsigned char* pg = big + p*PGSZ;
      __m512i a=_mm512_setzero_si512();
      for(long long o=0; o<PGSZ; o+=64) a=_mm512_add_epi8(a,_mm512_load_si512((const void*)(pg+o)));
      acc += (unsigned long long)_mm_cvtsi128_si32(_mm512_castsi512_si128(a));
    }
  return (void*)acc;
}
int main(int c, char**v){
  nth = c>1?atoi(v[1]):12;
  big = aligned_alloc(2*1024*1024, GB<<30); npg = (GB<<30)/PGSZ;
  memset(big,1,GB<<30);
  pthread_t th[64]; double t0=now();
  for(long i=0;i<nth;i++) pthread_create(&th[i],0,work,(void*)i);
  unsigned long long acc=0; for(long i=0;i<nth;i++){void*r;pthread_join(th[i],&r);acc+=(unsigned long long)r;}
  t_ms = now()-t0;
  printf("%2d threads: %.1f GB/s read (%.0f ms for %lld GiB, acc %llx)\n", nth, (double)(GB*iters)/ (t_ms/1000.0), t_ms, GB*iters, (unsigned long long)acc);
  return 0;
}
