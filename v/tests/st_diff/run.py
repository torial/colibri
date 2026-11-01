#!/usr/bin/env python3
"""Differential test of v/safetensors against c/st.h: model directories (valid ones and hostile ones: bad header
lengths, offsets and shapes out of bounds, wrong dtypes, mismatched numel, NULs, duplicates across shards) are given to both
dumpers; exit status and the dump of tensors and data hashes must be identical.
usage: run.py [N cases] [seed]   (needs gcc and a V compiler on PATH)"""
import json, os, random, shutil, struct, subprocess, sys, tempfile
SAN = ['-fsanitize=address,undefined', '-fno-sanitize-recover=undefined', '-g'] if os.environ.get('SAN') else []   # SAN=1: the C original under ASan + UBSan
N = int(sys.argv[1]) if len(sys.argv) > 1 else 400; random.seed(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.abspath(os.path.join(here, '..', '..', '..')); v_dir = os.path.join(root, 'v')
w = tempfile.mkdtemp(prefix='st_diff.')
subprocess.check_call(['gcc', '-O1'] + SAN + [ '-I' + os.path.join(root, 'c'), '-o', w + '/c_dump', here + '/c_dump.c', '-lm'])
subprocess.check_call(['v', '-o', w + '/v_dump', 'tests/st_diff/v_dump.v'], cwd=v_dir, stdout=subprocess.DEVNULL)
ESZ = {'BF16': 2, 'F16': 2, 'F32': 4, 'U8': 1, 'I8': 1, 'F8_E4M3': 1, 'F8_E8M0': 1, 'I64': 8, 'I32': 4, 'U32': 4, 'U64': 8}
def shard(tensors, meta=None, hlen_delta=0, pad=0, trunc=0, raw_header=None):
    hdr = {} if meta is None else {'__metadata__': meta}; data = b''; 
    for name, dt, shape, payload in tensors:
        a = len(data); data += payload
        hdr[name] = {'dtype': dt, 'shape': shape, 'data_offsets': [a, len(data)]}
    h = (raw_header if raw_header is not None else json.dumps(hdr)).encode() + b' ' * pad
    blob = struct.pack('<Q', max(0, min(len(h) + hlen_delta, 2**64 - 1))) + h + data
    return blob[:len(blob) - trunc] if trunc else blob
def rnd_tensor(i):
    dt = random.choice(list(ESZ)); rank = random.choice([0, 1, 1, 2, 2, 3, 4, 7, 8]); shape = [random.choice([0, 1, 2, 3, 5, 8] if rank <= 4 else [1, 1, 2]) for _ in range(rank)]
    n = 1
    for d in shape: n *= d
    return (random.choice(['w', 'model.layers.%d.x' % i, 'é', 'a.b.c']) + str(i), dt, shape, bytes(random.randrange(256) for _ in range(n * ESZ[dt])))
def case(k):
    d = os.path.join(w, 'case%05d' % k); os.makedirs(d)
    nsh = random.choice([1, 1, 2, 3]); mode = random.choice(['ok'] * 6 + ['hlen', 'trunc', 'badoff', 'badshape', 'baddt', 'mism', 'nul', 'dup', 'noobj', 'fields', 'pad', 'big', 'meta', 'f32tiny'])
    ctr = 0; first_name = None
    for s in range(nsh):
        ts = []
        for _ in range(random.choice([0, 1, 2, 4])): ctr += 1; ts.append(rnd_tensor(ctr))
        kw = {}
        if mode == 'hlen': kw['hlen_delta'] = random.choice([-3, 1, 5, 10**6, 2**63, 2**64 - 100])
        if mode == 'trunc': kw['trunc'] = random.choice([1, 5, 100])
        if mode == 'pad': kw['pad'] = random.randrange(0, 40)
        if mode == 'meta': kw['meta'] = {'format': 'pt', 'x': 'y'}
        if mode == 'dup' and s > 0 and ts and first_name: ts[0] = (first_name, ts[0][1], ts[0][2], ts[0][3])
        if mode == 'badoff' and ts:
            hdr = {n: {'dtype': dt, 'shape': sh, 'data_offsets': [0, len(p)]} for n, dt, sh, p in ts}; n0 = ts[0][0]
            hdr[n0]['data_offsets'] = random.choice([[5, 2], [0, 10**9], [-1, 3], [0.5, 2], [10**30, 10**30 + 1], [0]])
            kw['raw_header'] = json.dumps(hdr)
        if mode == 'badshape' and ts:
            hdr = {n: {'dtype': dt, 'shape': sh, 'data_offsets': [0, len(p)]} for n, dt, sh, p in ts}
            hdr[ts[0][0]]['shape'] = random.choice([[65535] * 5, [-1], [1.5], ['a'], [2**62, 4], [2**63], [1] * 9])
            kw['raw_header'] = json.dumps(hdr)
        if mode == 'baddt' and ts:
            hdr = {n: {'dtype': dt, 'shape': sh, 'data_offsets': [0, len(p)]} for n, dt, sh, p in ts}
            hdr[ts[0][0]]['dtype'] = random.choice(['FOO', 'F64', 5, None, 'bf16'])
            kw['raw_header'] = json.dumps(hdr)
        if mode == 'mism' and ts:
            hdr = {n: {'dtype': dt, 'shape': sh, 'data_offsets': [0, len(p)]} for n, dt, sh, p in ts}
            hdr[ts[0][0]]['shape'] = [random.randrange(1, 9)]
            kw['raw_header'] = json.dumps(hdr)
        if mode == 'nul': kw['raw_header'] = json.dumps({n: {'dtype': dt, 'shape': sh, 'data_offsets': [0, len(p)]} for n, dt, sh, p in ts}) + '\x00 '
        if mode == 'noobj': kw['raw_header'] = random.choice(['[]', '123', '"x"', 'null', '{"a":', '{"a":1}', '{"a":{"dtype":"F32"}}'])
        if mode == 'fields' and ts:
            hdr = {n: {'dtype': dt, 'shape': sh, 'data_offsets': [0, len(p)]} for n, dt, sh, p in ts}
            del hdr[ts[0][0]][random.choice(['dtype', 'shape', 'data_offsets'])]
            kw['raw_header'] = json.dumps(hdr)
        if mode == 'big': kw['hlen_delta'] = 0; kw['raw_header'] = json.dumps({'pad': 'x' * random.choice([10, 1000, 70000])})
        if mode == 'f32tiny': ts = [('t%d' % s, 'F32', [2, 3], struct.pack('<6f', *[random.uniform(-1e5, 1e5) for _ in range(6)]))]
        if s == 0 and ts: first_name = ts[0][0]
        name = random.choice(['model.safetensors', 'model-0000%d-of-00003.safetensors' % s, 'a%d.safetensors' % s]) if nsh > 1 else 'model.safetensors'
        if nsh > 1: name = 'model-%05d-of-%05d.safetensors' % (s, nsh)
        open(os.path.join(d, name), 'wb').write(shard(ts, **kw))
    if random.random() < .3: open(os.path.join(d, 'config.json'), 'w').write('{}')
    if random.random() < .2: open(os.path.join(d, 'notes.safetensors.txt'), 'w').write('x')
    return d
def run(binp, d):
    r = subprocess.run([binp, d], capture_output=True)
    return r.returncode != 0, r.stdout if r.returncode == 0 else b''
bad = 0; ok_cases = 0; err_cases = 0
for k in range(N):
    d = case(k); c = run(w + '/c_dump', d); v = run(w + '/v_dump', d)
    if c != v:
        bad += 1; print('DIFFERENT', d, 'C failed' if c[0] else 'C ok', '/', 'V failed' if v[0] else 'V ok')
        if bad >= 5: break
    elif c[0]: err_cases += 1
    else: ok_cases += 1
if bad: print('safetensors differential: DIFFERENT, kept in', w); sys.exit(1)
print('safetensors differential: %d directories (%d valid, %d refused alike), C and V identical' % (N, ok_cases, err_cases)); shutil.rmtree(w)
