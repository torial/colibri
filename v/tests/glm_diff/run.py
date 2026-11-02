#!/usr/bin/env python3
"""Differential test of the V GLM engine against c/colibri.c in its unquantized configuration (`colibri CAP 16 16`: 16-bit
expert and dense bits = f32 weights): tiny glm_moe_dsa checkpoints (MLA attention, sigmoid router with selection bias, shared
expert, leading dense layers; F32 and BF16; sharded) are run by the C engine (DUMP=<path> writes every forward's logits) and
by v/glm. The generated ids must be identical and the logits must agree; the report says how many are bit-identical.
usage: run.py [cases, default 10] [seed] [--exact]   (--exact: fail unless every logit is bit-identical)"""
import os, random, subprocess, sys, tempfile, importlib.util
import numpy as np
args = [a for a in sys.argv[1:] if not a.startswith('--')]; EXACT = '--exact' in sys.argv
N = int(args[0]) if args else 10; R = random.Random(int(args[1]) if len(args) > 1 else 1)
here = os.path.dirname(os.path.abspath(__file__)); v_dir = os.path.abspath(os.path.join(here, '..', '..')); c_dir = os.path.join(v_dir, '..', 'c')
arch = os.environ.get('ARCH', 'x86-64-v3'); march = [] if os.uname().machine in ('aarch64', 'arm64') else ['-march=' + arch]
spec = importlib.util.spec_from_file_location('gen_model', here + '/gen_model.py'); gm = importlib.util.module_from_spec(spec); spec.loader.exec_module(gm)
w = tempfile.mkdtemp(prefix='glm_diff.')
subprocess.check_call([v_dir + '/kernels/build.sh'], stdout=subprocess.DEVNULL, env=dict(os.environ, ARCH=arch))
subprocess.check_call(['gcc', '-O3'] + march + ['-fopenmp', '-pthread', '-Wno-unused-function', '-w', '-D_FILE_OFFSET_BITS=64', '-I' + c_dir, '-o', w + '/colibri', c_dir + '/colibri.c', '-lm'])
subprocess.check_call(['v', '-cc', 'gcc', '-cflags', ' '.join(['-O3'] + march), '-o', w + '/glm_v', 'tests/glm_diff/v_run.v'], cwd=v_dir, stdout=subprocess.DEVNULL)
env = dict(os.environ, COLI_NO_OMP_TUNE='1', OMP_NUM_THREADS='2', DRAFT='0')
exact = total = bad = n_al = ex_al = 0
for k in range(N):
    H = R.choice([1, 2, 4]); nope = R.choice([4, 8]); rope = R.choice([2, 4, 8]); vh = R.choice([4, 8]); E = R.choice([4, 8]); L = R.choice([2, 3])
    cfg = dict(D=R.choice([16, 32]), H=H, L=L, first_dense=R.choice([0, 1]), E=E, K=R.randint(1, min(E, 3)), mi=R.choice([8, 16]), di=R.choice([8, 24]),
               ql=R.choice([8, 16]), kvl=R.choice([8, 16]), nope=nope, rope=rope, vh=vh, shared=R.choice([1, 2]), V=R.choice([24, 48]), bf=R.random() < .4,
               norm_topk=R.random() < .6, routed_scale=R.choice([1.0, 1.5, 2.5]), theta=R.choice([10000.0, 1000.0]), prompt_len=R.choice([1, 3, 6]), n_new=R.choice([2, 5]),
               shards=R.choice([1, 2]))
    d = os.path.join(w, 'm%03d' % k); gm.make(d, k + 1, **cfg); cap = R.choice([1, 3, 16])
    ce = dict(env, SNAP=d, REF=d + '/ref.json', DUMP=d + '/c.dump')
    bits = int(os.environ['GLM_BITS']) if os.environ.get('GLM_BITS') else R.choice([16, 16, 8]); idot = R.random() < .6
    if not idot: ce['IDOT'] = '0'                       # int8 runs use the engine's float-activation kernels (IDOT=0)
    c = subprocess.run([w + '/colibri', str(cap), str(bits), str(bits)], capture_output=True, text=True, env=ce)
    ctoks = [l for l in c.stdout.split('\n') if l.startswith('GLM C engine')]
    ctoks = ctoks[0].split(':', 1)[1].split() if ctoks else None
    v = subprocess.run([w + '/glm_v', d, str(cap), d + '/ref.json', d + '/v.dump', str(bits)] + ([] if idot else ['noidot']), capture_output=True, text=True, env=env)
    if c.returncode or v.returncode or ctoks is None:
        print('RUN FAILED', d, c.returncode, c.stdout[-300:], c.stderr[-300:], '|', v.returncode, v.stderr[-300:]); bad += 1; continue
    vt = v.stdout.split('\n')[0].split()
    a = np.fromfile(d + '/c.dump', dtype=np.float32); b = np.fromfile(d + '/v.dump', dtype=np.float32)
    same = a.shape == b.shape and np.array_equal(a, b); total += 1; exact += same
    if os.environ.get('GLM_WHY'): print('exact' if same else 'inexact', {k: v for k, v in cfg.items() if k in ('H','L','first_dense','K','nope','rope','vh','kvl','ql','bf','shared','prompt_len','n_new','di','mi','D')})
    ok = vt == ctoks and a.shape == b.shape and np.allclose(a, b, rtol=1e-4, atol=1e-4 * max(1.0, float(np.abs(a).max())))
    if not ok or (EXACT and not same):
        bad += 1; print('DIFFERENT', d, 'bits', bits, 'idot', idot, cfg, '\n  C', ctoks, '\n  V', vt, '\n  shapes', a.shape, b.shape, 'max abs', float(np.abs(a - b).max()) if a.shape == b.shape else None)
        if bad >= 3: break
if bad: print('glm differential: DIFFERENT, kept in', w); sys.exit(1)
print('glm differential: %d checkpoints, generated ids identical, all logits within 1e-4; bit-identical in %d of %d' % (total, exact, total))
