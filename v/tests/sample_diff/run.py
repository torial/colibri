#!/usr/bin/env python3
"""Sampling differential test: c/sample.h against v/sample on generated logit rows, rng seeds, temperatures,
top-p values, bans, hostile (NaN / Inf) rows, and stop-set arming with and without a tokenizer.
usage: run.py [cases per kind, default 300] [seed, default 1]"""
import json, os, random, struct, subprocess, sys, tempfile
SAN = ['-fsanitize=address,undefined', '-fno-sanitize-recover=undefined', '-g'] if os.environ.get('SAN') else []   # SAN=1: the C original under ASan + UBSan

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
rnd = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
here = os.path.dirname(os.path.abspath(__file__))
f2u = lambda x: struct.unpack('<I', struct.pack('<f', x))[0]
u2f = lambda u: struct.unpack('<f', struct.pack('<I', u))[0]
NAN, PINF, NINF = 0x7fc00000, 0x7f800000, 0xff800000

def row(v):
    kind = rnd.choice(['norm', 'wide', 'flat', 'tied', 'peak', 'norm', 'norm'])
    if kind == 'norm': r = [f2u(rnd.gauss(0, 3)) for _ in range(v)]
    elif kind == 'wide': r = [f2u(rnd.gauss(0, 40)) for _ in range(v)]
    elif kind == 'flat': r = [f2u(rnd.gauss(0, 0.01)) for _ in range(v)]
    elif kind == 'tied': r = [f2u(float(rnd.randint(-2, 2))) for _ in range(v)]
    else:
        r = [f2u(rnd.gauss(0, 1)) for _ in range(v)]; r[rnd.randrange(v)] = f2u(rnd.uniform(20, 90))
    if rnd.random() < 0.12:                       # hostile entries
        for _ in range(rnd.randint(1, max(1, v // 3))):
            r[rnd.randrange(v)] = rnd.choice([NAN, PINF, NINF, NINF, 0x7fc00001, 0xffc00000])
    if rnd.random() < 0.03: r = [rnd.choice([NAN, PINF, NINF]) for _ in range(v)]
    return r

cases = []
for _ in range(N):
    v = rnd.choice([1, 2, 3, 5, 8, 17, 64, 200, 1000, 4096])
    temp = rnd.choice([0.0, 0.0, 1e-5, 0.3, 0.7, 1.0, 1.0, 1.5, 3.0, -1.0])
    nuc = rnd.choice([0.0, 1.0, 0.95, 0.9, 0.5, 0.1, 0.999, 0.01, 1.5])
    steps = rnd.randint(1, 6)
    c = ['S', '%08x' % f2u(temp), '%08x' % f2u(nuc), '%x' % rnd.getrandbits(64), str(v), str(steps)]
    for _ in range(steps):
        c.append(str(rnd.choice([-1, -1, rnd.randrange(v)])))
        c += ['%08x' % x for x in row(v)]
    cases.append(' '.join(c))
for v in (2, 4, 8, 16):                       # exact dyadic probabilities: the cumulative sum lands exactly on top-p
    for nuc in (0.5, 0.25, 0.75, 0.125, 0.0625, 0.375):
        cases.append(' '.join(['S', '%08x' % f2u(1.0), '%08x' % f2u(nuc), '%x' % rnd.getrandbits(64), str(v), '3'] +
                              sum([['-1'] + ['%08x' % f2u(0.0)] * v for _ in range(3)], [])))
for _ in range(N // 3):
    nc = rnd.randint(0, 70)
    cases.append('T %d %d %d %d %s' % (rnd.choice([-1, 5, 7, 300, 258]), rnd.randint(0, 1), rnd.randint(0, 1), nc,
                                       ' '.join(str(rnd.choice([3, 9, 258, 259, 260, 264, rnd.randrange(300)])) for _ in range(nc))))
for s in ['glm', 'GLM', 'chatglm', 'GlM4', 'gl', 'g', 'lm', 'model_type_GLM_moe', 'glam', 'qwen', '-', 'xxglm', 'glxm']:
    cases.append('M ' + s)

# a byte-level tokenizer with special tokens, some of them the content-control ones
bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256)); cs = bs[:]; n = 0
for b in range(256):
    if b not in bs: bs.append(b); cs.append(256 + n); n += 1
vocab = {chr(c): i for i, c in enumerate(cs)}
sp = ['<|endoftext|>', '[THINK]', '<|user|>', '[TOOL_CALLS]', '<|im_end|>', '[/THINK]', '[ARGS]', '<plain>', '[CALL_ID]']
added = [{'id': 256 + i, 'content': c, 'special': c != '<plain>'} for i, c in enumerate(sp)]
tj = {'added_tokens': added, 'pre_tokenizer': {'type': 'ByteLevel'}, 'model': {'type': 'BPE', 'vocab': vocab, 'merges': []}}

with tempfile.TemporaryDirectory() as w:
    open(f'{w}/cases.txt', 'w').write('\n'.join(cases) + '\n')
    json.dump(tj, open(f'{w}/tokenizer.json', 'w'))
    subprocess.check_call(['gcc', '-O1', '-w'] + SAN + ['-I' + here + '/../../../c', '-o', f'{w}/c', f'{here}/c_dump.c', '-lm'])
    subprocess.check_call(['v', '-o', f'{w}/v', f'{here}/v_dump.v'], stdout=subprocess.DEVNULL)
    co = subprocess.run([f'{w}/c', f'{w}/cases.txt', f'{w}/tokenizer.json'], capture_output=True, text=True)
    if co.returncode: print('C ORIGINAL FAILED:', co.stderr[-1500:]); sys.exit(1)
    co = co.stdout.strip().split('\n')
    vo = subprocess.run([f'{w}/v', f'{w}/cases.txt', f'{w}/tokenizer.json'], capture_output=True, text=True, check=True).stdout.strip().split('\n')
    if co != vo:
        for i, (a, b) in enumerate(zip(co, vo)):
            if a != b:
                print('DIFFERENT at line', i, '\n C:', a, '\n V:', b); break
        else: print('DIFFERENT: lengths', len(co), len(vo))
        sys.exit(1)
    print('sample differential: %d cases (%d output lines), C and V identical' % (len(cases), len(co)))
