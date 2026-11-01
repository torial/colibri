#!/usr/bin/env python3
"""Differential test of the V OLMoE engine against c/olmoe.c: tiny converted checkpoints (several shapes, F32 and BF16
dense weights, sharded, with and without top-k renormalization, a small expert cache) are run by the C engine (greedy,
DUMP=<path> writes every step's logits) and by v/olmoe; the generated ids and every logit's bits must be identical.
usage: run.py [cases, default 12] [seed]   (needs gcc and a V compiler; builds the C engine with the kernels' flags)"""
import os, random, subprocess, sys, tempfile, importlib.util
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
R = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
here = os.path.dirname(os.path.abspath(__file__)); v_dir = os.path.abspath(os.path.join(here, '..', '..')); c_dir = os.path.join(v_dir, '..', 'c')
arch = os.environ.get('ARCH', 'x86-64-v3'); march = [] if os.uname().machine in ('aarch64', 'arm64') else ['-march=' + arch]
spec = importlib.util.spec_from_file_location('gen_model', here + '/gen_model.py'); gm = importlib.util.module_from_spec(spec); spec.loader.exec_module(gm)
w = tempfile.mkdtemp(prefix='olmoe_diff.')
subprocess.check_call([v_dir + '/kernels/build.sh'], stdout=subprocess.DEVNULL, env=dict(os.environ, ARCH=arch))
subprocess.check_call(['gcc', '-O3'] + march + ['-fopenmp', '-pthread', '-Wno-unused-function', '-w', '-D_FILE_OFFSET_BITS=64', '-I' + c_dir, '-o', w + '/olmoe_c', c_dir + '/olmoe.c', '-lm'])
subprocess.check_call(['v', '-cc', 'gcc', '-cflags', ' '.join(['-O3'] + march), '-o', w + '/olmoe_v', 'tests/olmoe_diff/v_run.v'], cwd=v_dir, stdout=subprocess.DEVNULL)
env = dict(os.environ, COLI_NO_OMP_TUNE='1', OMP_NUM_THREADS=str(R.choice([1, 2, 4])))

def c_run(d, cap, kv16):
    e = dict(env, SNAP=d, DUMP=d + '/c.dump'); e.pop('KV16', None)
    if kv16: e['KV16'] = '1'
    r = subprocess.run([w + '/olmoe_c', str(cap), '8', d + '/ref.json'], capture_output=True, text=True, env=e)
    toks = [l for l in r.stdout.split('\n') if l.startswith('C engine :')]
    return r.returncode, (toks[0].split(':', 1)[1].split() if toks else None), r.stderr

bad = 0
for k in range(N):
    H = R.choice([1, 2, 4, 8]); hd = R.choice([2, 4, 8, 16]); D = H * hd; E = R.choice([2, 4, 8, 16]); K = R.randint(1, min(E, 4))
    cfg = dict(D=D, H=H, L=R.choice([1, 2, 3]), E=E, K=K, I=R.choice([4, 8, 16, 24, 32]), V=R.choice([16, 64, 100]),
               bf=R.random() < .4, shards=R.choice([1, 1, 3]), norm_topk=R.random() < .5, tie_gate=R.random() < .4, theta=R.choice([10000.0, 500000.0, 1000.0]),
               eps=R.choice([1e-5, 1e-6]), prompt_len=R.choice([1, 2, 5, 9]), n_new=R.choice([1, 4, 7]))
    d = os.path.join(w, 'm%03d' % k); gm.make(d, k + 1, **cfg)
    cap = R.choice([1, 2, 3, 16]); kv16 = R.random() < .3
    rc, ctoks, cerr = c_run(d, cap, kv16)
    ev = dict(env, SNAP=d)
    args = [w + '/olmoe_v', d, str(cap), d + '/ref.json', d + '/v.dump'] + (['kv16'] if kv16 else [])
    v = subprocess.run(args, capture_output=True, text=True, env=env)
    if rc != 0 or v.returncode != 0:
        print('RUN FAILED', d, 'C rc', rc, cerr[-300:], '| V rc', v.returncode, v.stderr[-300:]); bad += 1; continue
    vtoks = v.stdout.split('\n')[0].split()
    cd = open(d + '/c.dump', 'rb').read(); vd = open(d + '/v.dump', 'rb').read()
    if vtoks != ctoks or cd != vd:
        bad += 1
        first = next((i for i in range(min(len(cd), len(vd))) if cd[i] != vd[i]), None)
        print('DIFFERENT', d, cfg, 'kv16' if kv16 else '', '\n  C ids', ctoks, '\n  V ids', vtoks, '\n  dump bytes', len(cd), len(vd), 'first differing byte', first, '(step %s)' % (None if first is None else first // (4 * cfg['V'])))
        if bad >= 3: break
if bad: print('olmoe differential: DIFFERENT, kept in', w); sys.exit(1)
print('olmoe differential: %d checkpoints, generated ids and every logit of every step identical (C engine vs V)' % N)
