/* numa_topo.h -- sockets, physical cores and SMT siblings, read from sysfs.
 *
 * WHY.  COLI_NUMA=1 interleaves the expert slabs over the nodes, which spreads the load on
 * the memory controllers but makes half of every thread's reads remote on a two-socket box.
 * The dense trunk (attention, shared experts, router) is different: every decode matmul is
 * `#pragma omp parallel for schedule(static)` over the output rows, so thread t reads rows
 * [t*O/T, (t+1)*O/T) of every tensor, every token, always the same ones. If the team is laid
 * out node by node (the first cores of node 0, then node 1, ...) and each tensor's byte range
 * is bound in the same proportions, every one of those reads is local. COLI_NUMA=2 does that:
 * this file supplies the layout (the OMP_PLACES string, the cores per node), colibri.c the
 * binding (numa_dense_bind).
 *
 * The layout is built from sysfs and not assumed: Linux numbers the CPUs of a 2-socket Xeon
 * either node by node (0-13,28-41 | 14-27,42-55) or round robin (even | odd), and the SMT
 * sibling of cpu i is i+28 on one BIOS and i+1 on another.
 *
 * Linux only; everywhere else coli_topo_load returns 0 and nothing changes. `root` is "/sys"
 * (a test passes a directory with the same shape). `allowed` is a 0/1 per CPU array, or NULL
 * for every CPU the tree lists (a cgroup / taskset mask goes here). */
#ifndef COLI_NUMA_TOPO_H
#define COLI_NUMA_TOPO_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define COLI_TOPO_MAX_CPUS  1024
#define COLI_TOPO_MAX_NODES 64
#define COLI_TOPO_MAX_SMT   8

typedef struct {
    int nnodes;                          /* nodes that have at least one usable core */
    int ncores;                          /* physical cores, ordered by (node, lowest cpu) */
    int node_id[COLI_TOPO_MAX_NODES];    /* the kernel's number of the i-th node */
    int node_cores[COLI_TOPO_MAX_NODES]; /* cores in the i-th node */
    int core_node[COLI_TOPO_MAX_CPUS];   /* index (not id) of core i's node */
    int core_ncpu[COLI_TOPO_MAX_CPUS];
    int core_cpu[COLI_TOPO_MAX_CPUS][COLI_TOPO_MAX_SMT];
} coli_topo;

/* "0-13,28,30-31" -> cpus; returns the count (<= max). */
static inline int coli_parse_cpulist(const char *s, int *out, int max)
{
    int n = 0;
    while (s && *s) {
        while (*s == ' ' || *s == ',' || *s == '\n') s++;
        if (*s < '0' || *s > '9') break;
        char *e; long a = strtol(s, &e, 10), b = a;
        if (*e == '-') { s = e + 1; b = strtol(s, &e, 10); }
        for (long c = a; c <= b && n < max; c++) out[n++] = (int)c;
        s = e;
    }
    return n;
}

static inline int coli_topo_read(const char *root, const char *fmt, int a, char *buf, size_t cap)
{
    char path[512], rel[256];
    snprintf(rel, sizeof rel, fmt, a);
    snprintf(path, sizeof path, "%s/%s", root, rel);
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    size_t k = fread(buf, 1, cap - 1, f);
    fclose(f);
    buf[k] = 0;
    return 1;
}

static inline int coli_topo_load(coli_topo *t, const char *root, const unsigned char *allowed)
{
    memset(t, 0, sizeof *t);
#if defined(__linux__)
    static int cpus[COLI_TOPO_MAX_CPUS];
    char buf[8192];
    for (int node = 0; node < COLI_TOPO_MAX_NODES; node++) {
        if (!coli_topo_read(root, "devices/system/node/node%d/cpulist", node, buf, sizeof buf)) {
            char probe[256];                         /* a gap in the numbering: keep looking one more */
            if (node > 0 && !coli_topo_read(root, "devices/system/node/node%d/cpulist", node + 1, probe, sizeof probe)) break;
            continue;
        }
        int nc = coli_parse_cpulist(buf, cpus, COLI_TOPO_MAX_CPUS);
        int first_core = t->ncores;
        for (int i = 0; i < nc; i++) {
            int cpu = cpus[i];
            if (cpu < 0 || cpu >= COLI_TOPO_MAX_CPUS || (allowed && !allowed[cpu])) continue;
            int sib[COLI_TOPO_MAX_SMT * 4], ns = 0;
            if (coli_topo_read(root, "devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu, buf, sizeof buf))
                ns = coli_parse_cpulist(buf, sib, COLI_TOPO_MAX_SMT * 4);
            if (ns == 0) { sib[0] = cpu; ns = 1; }
            int lowest = sib[0];                     /* the core is named by its lowest usable sibling */
            for (int k = 0; k < ns; k++)
                if (sib[k] < lowest && (!allowed || (sib[k] >= 0 && sib[k] < COLI_TOPO_MAX_CPUS && allowed[sib[k]]))) lowest = sib[k];
            if (lowest != cpu) continue;             /* counted when its lowest sibling is visited */
            if (t->ncores >= COLI_TOPO_MAX_CPUS) break;
            int c = t->ncores++;
            t->core_node[c] = t->nnodes;
            t->core_ncpu[c] = 0;
            for (int k = 0; k < ns && t->core_ncpu[c] < COLI_TOPO_MAX_SMT; k++) {
                int sc = sib[k];
                if (sc < 0 || sc >= COLI_TOPO_MAX_CPUS || (allowed && !allowed[sc])) continue;
                t->core_cpu[c][t->core_ncpu[c]++] = sc;
            }
            /* ascending order inside the place */
            for (int a = 1; a < t->core_ncpu[c]; a++)
                for (int b = a; b > 0 && t->core_cpu[c][b - 1] > t->core_cpu[c][b]; b--) {
                    int x = t->core_cpu[c][b]; t->core_cpu[c][b] = t->core_cpu[c][b - 1]; t->core_cpu[c][b - 1] = x;
                }
        }
        if (t->ncores > first_core) {
            /* cores of one node by ascending lowest cpu (the cpulist is ascending, so they are) */
            t->node_id[t->nnodes] = node;
            t->node_cores[t->nnodes] = t->ncores - first_core;
            t->nnodes++;
        }
    }
    return t->ncores;
#else
    (void)root; (void)allowed;
    return 0;
#endif
}

/* "{0,28},{2,30},..." the places in team order: node by node, a core's SMT siblings together.
 * Returns the number of places written, 0 when the buffer is too small. */
static inline int coli_topo_places(const coli_topo *t, char *out, size_t cap)
{
    size_t n = 0;
    if (cap) out[0] = 0;
    for (int c = 0; c < t->ncores; c++) {
        char one[128]; int k = 0;
        k += snprintf(one + k, sizeof one - k, "%s{", c ? "," : "");
        for (int i = 0; i < t->core_ncpu[c]; i++) k += snprintf(one + k, sizeof one - k, "%s%d", i ? "," : "", t->core_cpu[c][i]);
        k += snprintf(one + k, sizeof one - k, "}");
        if (n + (size_t)k + 1 > cap) { if (cap) out[0] = 0; return 0; }
        memcpy(out + n, one, (size_t)k); n += (size_t)k; out[n] = 0;
    }
    return t->ncores;
}

/* The restriction of this process (sched_getaffinity) as a 0/1 array for coli_topo_load. */
#if defined(__linux__)
#include <sched.h>
static inline const unsigned char *coli_topo_allowed_mask(unsigned char *buf /* COLI_TOPO_MAX_CPUS */)
{
#ifdef CPU_SETSIZE
    cpu_set_t s; CPU_ZERO(&s);
    if (sched_getaffinity(0, sizeof s, &s) != 0) return NULL;
    for (int i = 0; i < COLI_TOPO_MAX_CPUS && i < CPU_SETSIZE; i++) buf[i] = CPU_ISSET(i, &s) ? 1 : 0;
    return buf;
#else
    (void)buf; return NULL;
#endif
}
#else
static inline const unsigned char *coli_topo_allowed_mask(unsigned char *buf){ (void)buf; return NULL; }
#endif

/* COLI_NUMA=2 and a real multi-node topology: fill `places` (OMP_PLACES) and return 1. */
static inline int coli_numa_local_places(char *places, size_t cap)
{
    const char *e = getenv("COLI_NUMA");
    if (!e || atoi(e) < 2) return 0;
    static coli_topo t; unsigned char am[COLI_TOPO_MAX_CPUS];
    if (!coli_topo_load(&t, "/sys", coli_topo_allowed_mask(am)) || t.nnodes < 2) return 0;
    return coli_topo_places(&t, places, cap) > 0;
}
#endif /* COLI_NUMA_TOPO_H */
