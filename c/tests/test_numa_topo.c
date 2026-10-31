/* numa_topo.h against fake sysfs trees: a 2-socket Xeon numbered round robin (even cpus on
 * node 0, SMT sibling = cpu+8) and one numbered node by node (sibling = cpu+4), a taskset-like
 * mask, and a single node. No real NUMA hardware needed. */
#define _GNU_SOURCE
#include "../numa_topo.h"
#include <assert.h>
#include <sys/stat.h>
#include <unistd.h>

static char root[256];
static void put(const char *rel, const char *text){
    char path[512]; snprintf(path,sizeof path,"%s/%s",root,rel);
    for(char *p=path+strlen(root)+1;*p;p++) if(*p=='/'){ *p=0; mkdir(path,0755); *p='/'; }
    FILE *f=fopen(path,"w"); assert(f); fputs(text,f); fclose(f);
}
static void cpu(int c,const char *sib){ char rel[128]; snprintf(rel,sizeof rel,"devices/system/cpu/cpu%d/topology/thread_siblings_list",c); put(rel,sib); }
static void fresh(void){ snprintf(root,sizeof root,"/tmp/numa_topo_test_%d_%ld",getpid(),(long)random()); mkdir(root,0755); }
static void rmtree(void){ char cmd[600]; snprintf(cmd,sizeof cmd,"rm -rf %s",root); (void)!system(cmd); }

int main(void){
    char places[1024]; coli_topo t;
    /* 16 cpus = 8 cores x 2 SMT, 2 nodes, even cpus on node 0 (the round robin numbering) */
    fresh();
    put("devices/system/node/node0/cpulist","0,2,4,6,8,10,12,14\n");
    put("devices/system/node/node1/cpulist","1,3,5,7,9,11,13,15\n");
    char sib[32];
    for(int c=0;c<8;c++){ snprintf(sib,sizeof sib,"%d,%d\n",c,c+8); cpu(c,sib); cpu(c+8,sib); }
    /* node0 = {0,2,4,6,8,10,12,14}: cores with lowest cpu 0,2,4,6 (siblings 8,10,12,14) */
    assert(coli_topo_load(&t,root,NULL)==8);
    assert(t.nnodes==2 && t.node_cores[0]==4 && t.node_cores[1]==4);
    assert(coli_topo_places(&t,places,sizeof places)==8);
    assert(strcmp(places,"{0,8},{2,10},{4,12},{6,14},{1,9},{3,11},{5,13},{7,15}")==0);
    assert(t.core_node[0]==0 && t.core_node[3]==0 && t.core_node[4]==1 && t.core_node[7]==1);
    /* a mask that leaves node 1 with two cores and drops the SMT siblings of node 0 */
    unsigned char am[COLI_TOPO_MAX_CPUS]={0};
    for(int c=0;c<8;c+=2) am[c]=1;      /* cpus 0,2,4,6 */
    am[1]=am[9]=am[3]=1;                 /* node 1: cores {1,9} and {3} */
    assert(coli_topo_load(&t,root,am)==6);
    assert(t.node_cores[0]==4 && t.node_cores[1]==2);
    assert(coli_topo_places(&t,places,sizeof places)==6);
    assert(strcmp(places,"{0},{2},{4},{6},{1,9},{3}")==0);
    /* a buffer that is too small is reported, not truncated */
    assert(coli_topo_places(&t,places,8)==0 && places[0]==0);
    rmtree();

    /* numbered node by node, sibling = cpu+4, no SMT file for one cpu (treated as one thread) */
    fresh();
    put("devices/system/node/node0/cpulist","0-3,8-11\n");
    put("devices/system/node/node1/cpulist","4-7,12-15\n");
    for(int c=0;c<4;c++){ snprintf(sib,sizeof sib,"%d,%d\n",c,c+8); cpu(c,sib); cpu(c+8,sib); }
    for(int c=4;c<8;c++){ snprintf(sib,sizeof sib,"%d-%d\n",c,c); (void)sib; }
    for(int c=4;c<8;c++){ char s2[32]; snprintf(s2,sizeof s2,"%d,%d\n",c,c+8); cpu(c,s2); cpu(c+8,s2); }
    assert(coli_topo_load(&t,root,NULL)==8);
    assert(coli_topo_places(&t,places,sizeof places)==8);
    assert(strcmp(places,"{0,8},{1,9},{2,10},{3,11},{4,12},{5,13},{6,14},{7,15}")==0);
    assert(t.node_cores[0]==4 && t.node_cores[1]==4);
    rmtree();

    /* one node: nnodes==1 (the caller then does nothing) */
    fresh();
    put("devices/system/node/node0/cpulist","0-3\n");
    for(int c=0;c<4;c++){ snprintf(sib,sizeof sib,"%d\n",c); cpu(c,sib); }
    assert(coli_topo_load(&t,root,NULL)==4 && t.nnodes==1);
    rmtree();

    /* no sysfs at all */
    assert(coli_topo_load(&t,"/nonexistent-root",NULL)==0);
    int cl[8]; assert(coli_parse_cpulist("0-2,5,7-8\n",cl,8)==6 && cl[0]==0 && cl[3]==5 && cl[5]==8);
    puts("numa_topo: ok");
    return 0;
}
