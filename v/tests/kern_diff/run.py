#!/usr/bin/env python3
"""Binding test of the V `kernels` module: every kernel is run on generated inputs (odd shapes, tails, special values,
random weight bytes in every ggml block type) by the C original called directly and by the V binding, at different
thread counts; the hash of the output bits must be identical. The V checks of slice lengths are tested by the module's
own unit tests (kernels_test.v).
usage: run.py [cases per kernel, default 40] [seed]   (needs gcc and a V compiler; builds libcolikern.a first)"""
import os, random, re, struct, subprocess, sys, tempfile
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
R = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
here = os.path.dirname(os.path.abspath(__file__)); v_dir = os.path.abspath(os.path.join(here, '..', '..')); c_dir = os.path.join(v_dir, '..', 'c')
arch = os.environ.get('ARCH', 'x86-64-v3'); march = [] if os.uname().machine in ('aarch64', 'arm64') else ['-march=' + arch]
w = tempfile.mkdtemp(prefix='kern_diff.')
subprocess.check_call([v_dir + '/kernels/build.sh'], stdout=subprocess.DEVNULL, env=dict(os.environ, ARCH=arch))
subprocess.check_call(['gcc', '-O3'] + march + ['-fopenmp', '-D_FILE_OFFSET_BITS=64', '-Wno-unused-function', '-I' + c_dir, '-o', w + '/c_ref', here + '/c_ref.c', '-lm'])
subprocess.check_call(['v', '-cc', 'gcc', '-o', w + '/v_run', 'tests/kern_diff/v_run.v'], cwd=v_dir, stdout=subprocess.DEVNULL)

q = open(os.path.join(c_dir, 'ggml_quants.h')).read()
ids = {m.group(1): int(m.group(2)) for m in re.finditer(r'(GQ_[A-Z0-9_]+) = (\d+)', re.search(r'enum \{\s*GQ_F32 = 0,(.*?)\};', q, re.S).group(0))}
GT = {}                                           # type id -> (block elements, block bytes, supported)
for m in re.finditer(r'\[(GQ_[A-Z0-9_]+)\]\s*=\s*\{ "[A-Za-z0-9_]+",\s*(\d+),\s*(\d+),\s*(\d)', q):
    GT[ids[m.group(1)]] = (int(m.group(2)), int(m.group(3)), int(m.group(4)))

def fl(n, scale=1.0, special=0.0):
    out = []
    for _ in range(n):
        r = R.random()
        if r < special / 3: out.append(0.0)
        elif r < 2 * special / 3: out.append(R.choice([1e-39, -1e-39, 3e38, -3e38]))
        elif r < special: out.append(-0.0)
        else: out.append(R.gauss(0, scale))
    return struct.pack('<%df' % n, *out)
def rb(n): return bytes(R.randrange(256) for _ in range(n))
def pos(n, lo=0.001, hi=0.1): return struct.pack('<%df' % n, *[R.uniform(lo, hi) for _ in range(n)])

lines = []; cnt = [0]
def put(data):
    if data is None: return '-'
    p = '%s/f%d' % (w, cnt[0]); cnt[0] += 1; open(p, 'wb').write(data); return p
def case(name, S, I, O, extra, *arrs):
    th = R.choice([1, 2, 3, 4, 7])
    fs = [put(a) for a in arrs] + ['-'] * (4 - len(arrs))
    lines.append(' '.join(map(str, [name, S, I, O, extra, th] + fs)))
def shape():
    return R.choice([1, 1, 2, 3, 4, 5, 8, 9]), R.choice([1, 2, 3, 7, 16, 31, 32, 33, 64, 100, 129, 256, 300, 513]), R.choice([1, 2, 3, 4, 5, 8, 17, 63, 130])

for _ in range(N):
    S, I, O = shape(); sp = R.choice([0, 0, .1])
    case('f32', S, I, O, 0, fl(S * I, 1, sp), fl(O * I, 1, sp))
    case('dot', 1, I, 1, 0, fl(I, 1, sp), fl(I, 1, sp))
    S, I, O = shape(); case('q', S, I, O, 0, fl(S * I, 2, sp), rb(O * I), pos(O))
    S, I, O = shape(); case('i4', S, I, O, 0, fl(S * I, 2, sp), rb(O * ((I + 1) // 2)), pos(O))
    S, I, O = shape(); gs = R.choice([1, 2, 7, 16, 32, 64, 128]); case('i4g', S, I, O, gs, fl(S * I, 2, sp), rb(O * ((I + 1) // 2)), pos(O * ((I + gs - 1) // gs)))
    S, I, O = shape(); case('fp8', S, I, O, 0, fl(S * I, 1), bytes(R.choice([b for b in range(256) if b & 0x7f != 0x7f]) for _ in range(O * I)), pos(((O + 127) // 128) * ((I + 127) // 128), .5, 2))
    S, I, O = shape(); case('mx4', S, I, O, 0, fl(S * I, 1), rb(O * ((I + 1) // 2)), bytes(R.randrange(100, 140) for _ in range(O * ((I + 31) // 32))))
    for kind in ('qif32', 'qibf16', 'qii8'):
        S, I, O = R.choice([1, 2, 5, 7, 20, 193]), R.choice([1, 8, 33, 100, 257, 300]), R.choice([1, 3, 16, 17, 40, 70])
        wb = {'qif32': fl(O * I, 1), 'qibf16': b''.join(struct.pack('<H', R.randrange(0x3000, 0x4400) | R.choice([0, 0x8000])) for _ in range(O * I)), 'qii8': rb(O * I)}[kind]
        case(kind, S, I, O, 0, fl(S * I, 1), wb, pos(O) if kind == 'qii8' else None, fl(O, 1) if R.random() < .5 else None)
for t, (blck, by, sup) in sorted(GT.items()):
    for _ in range(max(1, N // 10)):
        blocks = R.choice([1, 2, 3, 5]); I = blck * blocks; S = R.choice([1, 2, 4, 7]); O = R.choice([1, 3, 8, 17])
        case('gqrow', 1, I, 1, t, rb(blocks * by))
        case('gq', S, I, O, t, fl(S * I, 1), rb(O * blocks * by))
# a type id with no decoder, and a width that is not whole blocks
case('gqrow', 1, 5, 1, 31, rb(100)); case('gqrow', 1, 33, 1, 12, rb(1000))

open(w + '/cases.txt', 'w').write('\n'.join(lines) + '\n')
c = subprocess.run([w + '/c_ref', w + '/cases.txt'], capture_output=True, text=True); v = subprocess.run([w + '/v_run', w + '/cases.txt'], capture_output=True, text=True)
if c.returncode or v.returncode: print('FAILED to run', c.returncode, v.returncode, c.stderr[-300:], v.stderr[-300:]); sys.exit(1)
co, vo = c.stdout.strip().split('\n'), v.stdout.strip().split('\n')
bad = [(i, a, b) for i, (a, b) in enumerate(zip(co, vo)) if a != b]
if bad or len(co) != len(vo):
    for i, a, b in bad[:6]: print('DIFFERENT case', i, lines[i][:90], '\n  C:', a, '\n  V:', b)
    print('kernel binding: DIFFERENT, kept in', w); sys.exit(1)
nan = sum(1 for l in co if ' nan=0 ' not in l)
print('kernel binding: %d kernel runs (%d ggml types, %s), C and V identical%s' % (len(co), len(GT), ' '.join(march) or 'native', '' if not nan else ' (%d outputs contain NaN, identical)' % nan))
