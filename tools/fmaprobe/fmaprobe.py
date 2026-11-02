#!/usr/bin/env python3
"""fmaprobe: which summation order did the compiler give this C loop?

A float32 reduction has no single meaning. gcc may keep a plain loop, fuse each multiply-add, split the sum over SIMD lanes and
fold the lanes at the end, or unroll differently depending on how long the loop is. Port such code to another language and the
last bit of the result depends on which. fmaprobe compiles your function for several instruction-set levels, runs it on random
inputs of many lengths, and reports, per length, which reference order it reproduces bit for bit.

  fmaprobe.py FILE.c FUNCTION [--archs v1,v2,v3,v4,native] [--lengths 1-40,64] [--trials 200] [--asm]

FUNCTION must be  float f(const float *x, const float *y, int n, float init)  (init may be ignored).
"""
import argparse, ctypes, os, random, re, struct, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
REFS = [('seq', 'plain loop, multiply then add'), ('seqfma', 'one fused chain'), ('lanes4', '4 fused lanes, folded'),
        ('lanes8', '8 fused lanes, folded (AVX2)'), ('lanes16', '16 fused lanes (AVX-512)'), ('pair', 'pairwise tree'),
        ('hyb4', 'blocks of 4 unfused, fused tail'), ('hyb8', 'blocks of 8 unfused, fused tail'), ('hyb16', 'blocks of 16 unfused, fused tail')]
ARCH = {'v1': 'x86-64', 'v2': 'x86-64-v2', 'v3': 'x86-64-v3', 'v4': 'x86-64-v4', 'native': 'native'}

def build(src, out, march, extra=()):
    subprocess.check_call(['gcc', '-O3', '-march=' + march, '-fPIC', '-shared', '-ffp-contract=fast', '-w'] + list(extra) + [src, '-o', out, '-lm'])

def load(path, name):
    lib = ctypes.CDLL(path); f = getattr(lib, name)
    f.restype = ctypes.c_float
    f.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_float]
    return f

def bits(x): return struct.unpack('<I', struct.pack('<f', x))[0]

def lengths(spec):
    out = []
    for part in spec.split(','):
        if '-' in part: a, b = part.split('-'); out += range(int(a), int(b) + 1)
        else: out.append(int(part))
    return out

def asm_summary(src, name, march):
    p = subprocess.run(['gcc', '-O3', '-march=' + march, '-ffp-contract=fast', '-w', '-S', '-o', '-', src], capture_output=True, text=True)
    body, on = [], False
    for line in p.stdout.split('\n'):
        if re.match(r'^%s:' % re.escape(name), line): on = True; continue
        if on and re.match(r'^\s*\.size\s+%s' % re.escape(name), line): break
        if on: body.append(line)
    t = '\n'.join(body)
    def c(pat): return len(re.findall(pat, t))
    return {'fused scalar': c(r'vfmadd\w*ss'), 'fused packed': c(r'vfmadd\w*ps'), 'mul scalar': c(r'v?mulss'), 'add scalar': c(r'v?addss'),
            'mul packed': c(r'v?mulps'), 'add packed': c(r'v?addps'), 'zmm': c(r'%zmm'), 'ymm': c(r'%ymm'), 'xmm only': 0 if c(r'%[yz]mm') else 1}

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('file'); ap.add_argument('func'); ap.add_argument('--archs', default='v1,v2,v3,v4')
    ap.add_argument('--lengths', default='1-24,31,32,33,48,64,100'); ap.add_argument('--trials', type=int, default=200)
    ap.add_argument('--asm', action='store_true', help='also print instruction counts of the function'); ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args(); rnd = random.Random(a.seed); lens = lengths(a.lengths); w = tempfile.mkdtemp(prefix='fmaprobe.')
    nmax = max(lens); data = [([rnd.gauss(0, 1) for _ in range(nmax)], [rnd.gauss(0, 1) for _ in range(nmax)], rnd.gauss(0, 1)) for _ in range(a.trials)]
    refpath = os.path.join(w, 'refs.so'); build(os.path.join(HERE, 'refs.c'), refpath, 'native', ['-ffp-contract=off'])
    refs = {n: load(refpath, 'ref_' + n) for n, _ in REFS}
    def call(f, x, y, n, init):
        xs = (ctypes.c_float * n)(*x[:n]); ys = (ctypes.c_float * n)(*y[:n]); return f(xs, ys, n, init)
    refval = {n: [[bits(call(refs[n], x, y, L, i)) for L in lens] for x, y, i in data] for n, _ in REFS}
    print('%s:%s   %d random inputs per length\n' % (a.file, a.func, a.trials))
    for key in a.archs.split(','):
        march = ARCH.get(key, key)
        so = os.path.join(w, key + '.so')
        try: build(a.file, so, march)
        except subprocess.CalledProcessError: print('== %s: does not build on this machine\n' % key); continue
        f = load(so, a.func); got = [[bits(call(f, x, y, L, i)) for L in lens] for x, y, i in data]
        print('== -march=%s' % march)
        if a.asm:
            s = asm_summary(a.file, a.func, march); print('   asm: ' + ', '.join('%s %s' % (k, v) for k, v in s.items() if v))
        groups, cur = [], None
        for k, L in enumerate(lens):
            m = [n for n, _ in REFS if all(got[t][k] == refval[n][t][k] for t in range(a.trials))]
            m = m or ['none']
            if cur and cur[1] == m and cur[2] == L - 1: cur[2] = L
            else: cur = [L, m, L]; groups.append(cur)
        for lo, m, hi in groups:
            span = str(lo) if lo == hi else '%d-%d' % (lo, hi)
            print('   n = %-9s %s' % (span, ' = '.join(m) if m != ['none'] else 'matches no reference (a different order)'))
        print()
    print('references: ' + '; '.join('%s: %s' % r for r in REFS))
main()
