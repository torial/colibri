/* Throughput of the integer-dot tiers: build it once per ISA and compare.
 *   make tests/bench_idot_tiers ARCH=x86-64-v3        # avx2
 *   make tests/bench_idot_tiers ARCH=skylake-avx512   # avx512bw (Xeon Gold 6132 class)
 *   make tests/bench_idot_tiers ARCH=native           # avx512-vnni where the CPU has it
 * Single thread by default (OMP_NUM_THREADS=1) shows the per-core vector cost; with the
 * whole team it shows whether the DRAM (or the other socket's) bandwidth is the wall.
 * MB sizes are of weights; one decode token reads each weight byte once. */
#include "../idot.h"
#include <time.h>
#include <omp.h>
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
int main(int argc,char **argv){
    int mb=argc>1?atoi(argv[1]):256;          /* weight bytes per matrix, MB */
    int I=4096; int rb=I/2;
    int O4=(int)((int64_t)mb*1000000/rb); O4&=~15;       /* int4 rows */
    int O8=(int)((int64_t)mb*1000000/I);  O8&=~15;       /* int8 rows */
    uint8_t *w4=aligned_alloc(64,(size_t)O4*rb); int8_t *w8=aligned_alloc(64,(size_t)O8*I);
    int8_t *x=aligned_alloc(64,I), *g8=aligned_alloc(64,(size_t)O8*I);
    float *y=malloc(sizeof(float)*(O8>O4?O8:O4)), *sc=malloc(sizeof(float)*O8);
    uint32_t r=1; for(size_t i=0;i<(size_t)O4*rb;i++){ r=r*1664525u+1013904223u; w4[i]=(uint8_t)(r>>24); }
    for(size_t i=0;i<(size_t)O8*I;i++){ r=r*1664525u+1013904223u; w8[i]=(int8_t)(r>>24); if(w8[i]==-128) w8[i]=-127; }
    for(int i=0;i<I;i++){ r=r*1664525u+1013904223u; x[i]=(int8_t)((int)(r>>24)%255-127); }
    for(int o=0;o<O8;o++) sc[o]=0.01f;
    int32_t xsum=0; for(int i=0;i<I;i++) xsum+=x[i];
    i8r16_pack(g8,w8,O8,I);
    printf("kernel=%s threads=%d  int4 %d rows x %d, int8 %d rows x %d\n",IDOT_KERNEL,omp_get_max_threads(),O4,I,O8,I);
    volatile int32_t sink=0; double t; int reps=mb<16?400:5;
    t=now(); for(int k=0;k<reps;k++){
        #pragma omp parallel for schedule(static)
        for(int o=0;o<O4;o++){ y[o]=(float)(dot_i4p_u(w4+(size_t)o*rb,x,I)-8*xsum); } sink+=(int32_t)y[0]; }
    t=(now()-t)/reps; printf("dot_i4p_u   : %7.2f ms  %6.1f GB/s\n",t*1e3,(double)O4*rb/t/1e9);
    t=now(); for(int k=0;k<reps;k++){
        #pragma omp parallel for schedule(static)
        for(int o=0;o<O8;o++){ y[o]=(float)dot_i8i8(w8+(size_t)o*I,x,I); } sink+=(int32_t)y[0]; }
    t=(now()-t)/reps; printf("dot_i8i8    : %7.2f ms  %6.1f GB/s\n",t*1e3,(double)O8*I/t/1e9);
    t=now(); for(int k=0;k<reps;k++){ matmul_q_idot_r16(y,x,(float[]){1.f},&xsum,g8,sc,1,I,O8); sink+=(int32_t)y[0]; }
    t=(now()-t)/reps; printf("int8 r16    : %7.2f ms  %6.1f GB/s\n",t*1e3,(double)O8*I/t/1e9);
    (void)sink; return 0;
}
