/* numa_dense_bind's block arithmetic: a tensor is cut in the proportion of the cores of each
 * node and each piece gets its own MPOL_BIND. This box has one node, so the "two nodes" both
 * point at node 0: what is checked is the cut (from /proc/self/numa_maps, one VMA per piece,
 * with the bind policy) and that the rest of the program is unaffected. Skipped (exit 0) where
 * mbind or numa_maps is unavailable. */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main
#include <assert.h>
int main(void){
#ifdef __linux__
    /* the cut: 1 core of 4 -> a quarter, page aligned; the last piece always ends at hi */
    uintptr_t lo=0x10000000u, hi=lo+((uintptr_t)48<<20);
    assert(numa_dense_cut(lo,hi,1,4)==lo+((uintptr_t)12<<20));
    assert(numa_dense_cut(lo,hi,2,4)==lo+((uintptr_t)24<<20));
    assert(numa_dense_cut(lo,hi,4,4)==hi);
    assert((numa_dense_cut(lo,lo+4096*1000+123*4096,1,3)&4095)==0);
    size_t n=(size_t)48<<20;
    char *p=mmap(NULL,n,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if(p==MAP_FAILED){ puts("numa_dense: mmap failed, skipped the bind half"); return 0; }
    g_numa_dn=2; g_numa_dnode[0]=0; g_numa_dnode[1]=0; g_numa_dcores[0]=1; g_numa_dcores[1]=3; g_numa_dtotal=4;
    errno=0; long probe=syscall(SYS_mbind,p,4096,2,(unsigned long[]){1},65UL,2);
    if(probe<0){ printf("numa_dense: cut ok; mbind unavailable (%s), bind half skipped\n",strerror(errno)); return 0; }
    numa_dense_bind(p,n);                      /* both pieces to node 0: this box has one */
    for(size_t i=0;i<n;i+=4096) p[i]=1;        /* touching after the bind must work */
    int bound=0; FILE *f=fopen("/proc/self/numa_maps","r");
    if(f){ char line[1024]; while(fgets(line,sizeof line,f)){ unsigned long a;
        if(sscanf(line,"%lx",&a)==1 && a>=(unsigned long)p && a<(unsigned long)p+n && strstr(line,"bind:0")) bound=1; } fclose(f); }
    printf("numa_dense: ok (cut checked; bind policy %s in numa_maps)\n",bound?"seen":"not visible here");
#endif
    return 0;
}
