#!/usr/bin/env python3
"""Extended safetensors differential test: c/st.h against v/safetensors on what st_diff does not reach -- a search
path of directories, the model.safetensors.index.json overlay, colibri.fmt stamps, NULs in JSON strings, read
mirrors, and every reader (capped, scale, slices, physical and replica ranges with O_DIRECT, mappings, the
missing-tensor report). Each case: the tensor table and stamps from both, then a generated list of operations
ending in at most one that is expected to be refused; exit status, stdout and stderr must be identical.
usage: run.py [N cases] [seed]   (needs gcc and a V compiler on PATH)"""
import difflib, json, os, random, shutil, struct, subprocess, sys, tempfile
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
R = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.abspath(os.path.join(here, '..', '..', '..')); v_dir = os.path.join(root, 'v')
CHUNK = 4093            # a small pread chunk on both sides: every read goes through the chunk loop
w = tempfile.mkdtemp(prefix='st_ext.', dir=os.environ.get('TMPDIR', '/tmp'))
for tag, flags in (('small', ['-DST_PREAD_CHUNK=%d' % CHUNK]), ('big', [])):
    subprocess.check_call(['gcc', '-O1', '-w'] + flags + ['-I' + os.path.join(root, 'c'), '-o', w + '/c_ext_' + tag, here + '/c_ext.c', '-lm'])
subprocess.check_call(['v', '-o', w + '/v_ext', 'tests/st_ext/v_ext.v'], cwd=v_dir, stdout=subprocess.DEVNULL)
ESZ = {'BF16': 2, 'F16': 2, 'F32': 4, 'U8': 1, 'I8': 1, 'F8_E4M3': 1, 'F8_E8M0': 1, 'I64': 8, 'I32': 4, 'U32': 4, 'U64': 8}
FLOAT = ('BF16', 'F16', 'F32')
def hexs(s): return s.encode().hex()
def payload(n): return bytes(R.randrange(256) for _ in range(n))
def tensor(name, dt=None, big=False):
    dt = dt or R.choice(list(ESZ)); rank = R.choice([1, 1, 2, 3]); shape = [R.choice([1, 2, 3, 7, 16, 33]) for _ in range(rank)]
    if big: shape = [R.choice([64, 100, 129]), R.choice([64, 70, 128])]
    n = 1
    for d in shape: n *= d
    return (name, dt, shape, payload(n * ESZ[dt]))
def shard_bytes(tensors, meta=None, raw_header=None, pad=0):
    hdr = {} if meta is None else {'__metadata__': meta}; data = b''
    for name, dt, shape, p in tensors:
        a = len(data); data += p; hdr[name] = {'dtype': dt, 'shape': shape, 'data_offsets': [a, len(data)]}
    h = (raw_header if raw_header is not None else json.dumps(hdr)).encode() + b' ' * pad
    return struct.pack('<Q', len(h)) + h + data
def write(d, name, blob):
    os.makedirs(d, exist_ok=True); open(os.path.join(d, name), 'wb').write(blob)

def build_case(k):
    """returns (primary, extra or None, mirrors [dirs], env, expect_note)"""
    base = os.path.join(w, 'c%05d' % k); kind = R.choice(['out', 'many', 'plain', 'plain', 'split', 'overlay', 'stamps', 'stamps', 'overlay', 'mirror', 'mirror', 'big', 'nul'] + ['plain'] * 3)
    prim = os.path.join(base, 'p'); extra = None; mirrors = []; env = {}
    if R.random() < .5: env['COLI_MAP_EXPERTS'] = '1'
    env['_KIND'] = kind
    small = R.random() < .5; env['_CHUNK'] = 'small' if small else 'big'
    if small: env['ST_CHUNK'] = str(CHUNK)
    if kind == 'out':
        for nm in R.choice([['out-00000', 'out-00001', 'out-00003'], ['out-00002', 'out-00003', 'out-mtp-1'], ['out-00001'], ['out-0', 'out-1', 'out-2']]):
            write(prim, nm + '.safetensors', shard_bytes([tensor('o.' + nm)]))
    elif kind == 'many':
        for i in range(R.choice([511, 512, 513])): write(prim, 'model-%05d-of-00600.safetensors' % i, shard_bytes([]))
    if kind in ('plain', 'big', 'mirror'):
        for s in range(R.choice([1, 2, 3])):
            ts = [tensor('model.layers.%d.%d.w' % (s, i), big=(kind == 'big' and i == 0)) for i in range(R.choice([1, 3, 6]))]
            write(prim, 'model-%05d-of-%05d.safetensors' % (s + 1, 3), shard_bytes(ts))
    elif kind == 'split':
        dirs = [prim] + [os.path.join(base, 'x%d' % i) for i in range(R.choice([1, 2]))]
        for d_ in dirs: os.makedirs(d_, exist_ok=True)
        for s in range(R.choice([2, 3, 4])):
            ts = [tensor('t%d.%d' % (s, i)) for i in range(R.choice([1, 2]))]
            d = R.choice(dirs); write(d, 'model-%05d-of-00004.safetensors' % (s + 1), shard_bytes(ts))
            if R.random() < .3:                                    # the same shard name on another drive: the first listed wins
                d2 = R.choice(dirs); write(d2, 'model-%05d-of-00004.safetensors' % (s + 1), shard_bytes([tensor('t%d.%d' % (s, i)) for i in range(2)]))
        os.makedirs(prim, exist_ok=True)
        seps = [';', ',', ' ; ', ', ', ';;', ',  ,']
        parts = dirs[1:] + ([os.path.join(base, 'nodir_but_empty')] if False else [])
        extra = R.choice(seps).join(parts) if R.random() < .85 else ' ' + R.choice(seps).join(parts) + ' '
        if R.random() < .1: extra = '  '
    elif kind == 'overlay':
        common = [tensor('dup.a'), tensor('dup.b'), tensor('only0')]
        write(prim, 'model-00001-of-00002.safetensors', shard_bytes(common))
        ov = [tensor('dup.a'), tensor('dup.b'), tensor('only1')]
        write(prim, 'model-00002-of-00002.safetensors', shard_bytes(ov))
        v = R.choice(['none', 'to1', 'to0', 'mixed', 'third', 'missing', 'badjson', 'nul', 'notobj', 'nonstr', 'ok_partial'])
        wm = {'only0': 'model-00001-of-00002.safetensors', 'only1': 'model-00002-of-00002.safetensors'}
        if v == 'to1': wm.update({'dup.a': 'model-00002-of-00002.safetensors', 'dup.b': 'model-00002-of-00002.safetensors'})
        elif v == 'to0': wm.update({'dup.a': 'model-00001-of-00002.safetensors', 'dup.b': 'model-00001-of-00002.safetensors'})
        elif v == 'mixed': wm.update({'dup.a': 'model-00001-of-00002.safetensors', 'dup.b': 'model-00002-of-00002.safetensors'})
        elif v == 'third': wm.update({'dup.a': 'model-00003-of-00003.safetensors', 'dup.b': 'model-00002-of-00002.safetensors'})
        elif v == 'ok_partial': wm.update({'dup.a': 'model-00002-of-00002.safetensors'})
        elif v == 'nonstr': wm.update({'dup.a': 5, 'dup.b': 'model-00001-of-00002.safetensors'})
        txt = {'none': None, 'badjson': '{"weight_map": {', 'nul': '{"weight_map": {"dup.a": "model-00001-of-00002.safetensors"}}\x00', 'notobj': '{"weight_map": [1]}'}.get(v, json.dumps({'weight_map': wm}))
        if v == 'missing': txt = json.dumps({'metadata': {}})
        if txt is not None: open(os.path.join(prim, 'model.safetensors.index.json'), 'w').write(txt)
    elif kind == 'stamps':
        names = ['a.w', 'b.w', 'c.w']
        def stamp(d): return {'colibri.fmt': json.dumps(d)}
        v = R.choice(['ok', 'ok', 'agree', 'agree', 'rawmeta', 'nulval', 'conflict', 'nonstr', 'notjson', 'notobj', 'badval', 'empty', 'many'])
        m0 = stamp({'a.w': 'int4', 'b.w': 'fp8'}); m1 = stamp({'b.w': 'fp8', 'c.w': 'int8'})
        if v == 'conflict': m1 = stamp({'b.w': 'int8'})
        if v == 'nonstr': m1 = {'colibri.fmt': 5}
        if v == 'notjson': m1 = {'colibri.fmt': '{"a":'}
        if v == 'notobj': m1 = {'colibri.fmt': '[1,2]'}
        if v == 'badval': m1 = stamp({'c.w': 7})
        if v == 'empty': m1 = {'colibri.fmt': ''}
        if v == 'many': m1 = stamp({'n%d' % i: 'f' for i in range(R.choice([4095, 4096, 4097]))})
        if v == 'rawmeta': m1 = {'colibri.fmt': '{"c.w":"x\\u0000yz","d.w":"q"}'}
        if v == 'nulval': m1 = {'colibri.fmt': '{"c.w":"ok"}\x00junk'}
        write(prim, 'model-00001-of-00002.safetensors', shard_bytes([tensor('a.w'), tensor('b.w')], meta=m0))
        write(prim, 'model-00002-of-00002.safetensors', shard_bytes([tensor('c.w')], meta=m1))
    elif kind == 'nul':
        names = R.choice([('a\\u0000b', 'ok'), ('a\\u0000b', 'a\\u0000c'), ('a', 'a\\u0000zz'), ('x', 'y')])
        dts = R.choice([('F32', 'U8'), ('F32\\u0000junk', 'U8\\u0000zz'), ('F\\u000032', 'U8')])
        raw = '{"%s": {"dtype": "%s", "shape": [2], "data_offsets": [0, 8]}, "%s": {"dtype": "%s", "shape": [3], "data_offsets": [8, 11]}}' % (names[0], dts[0], names[1], dts[1])
        if R.random() < .3: raw = raw.replace('"dtype"', '"dtype\\u0000x"', 1)
        write(prim, 'model.safetensors', shard_bytes([], raw_header=raw) + payload(11))
    if kind in ('mirror', 'plain', 'big', 'split'):
        pass
    if kind != 'many' and (kind == 'mirror' or R.random() < .25):
        for mi in range(R.choice([1, 2, 5])):
            md = os.path.join(base, 'm%d' % mi); os.makedirs(md, exist_ok=True); mirrors.append(md)
            for f in sorted(os.listdir(prim)) if os.path.isdir(prim) else []:
                if not f.endswith('.safetensors'): continue
                b = open(os.path.join(prim, f), 'rb').read(); m = R.choice(['same', 'same', 'same', 'skip', 'trunc', 'hdrflip', 'datachg'])
                if m == 'skip': continue
                if m == 'trunc': b = b[:-3]
                if m == 'hdrflip' and len(b) > 13: b = b[:12] + bytes([b[12] ^ 1]) + b[13:]
                if m == 'datachg': b = b[:-1] + bytes([b[-1] ^ 0xFF])
                open(os.path.join(md, f), 'wb').write(b)
    return base, prim, extra, mirrors, env

def run(binp, prim, extra, ops, env):
    e = dict(os.environ); e.update({k: v for k, v in env.items() if not k.startswith('_')})
    if binp.endswith('c_ext'): binp = binp + '_' + env.get('_CHUNK', 'big')
    r = subprocess.run([binp, prim, extra if extra is not None else '-', ops], capture_output=True, env=e)
    return r.returncode, r.stdout, r.stderr

def make_ops(table, sizes, mirrors, nshard, kind=''):
    ops = []
    for m in mirrors: ops.append('%s %s' % (R.choice(['mirror', 'mirror', 'mirrorinit']), m))
    if R.random() < .1: ops.append('mirrorreset')
    ops.append('nrep')
    for s in range(nshard):
        for r in range(3): ops.append('fdrep %d %d' % (s, r))
    names = [t[0] for t in table]
    ops.append('find %s' % hexs('definitely.not.there'))
    for t in R.sample(table, min(len(table), 8)):
        name, dt, off, nb, numel, sh = t
        ops.append('find %s' % name)
        if dt < 3:
            ops.append('f32 %s %d %d' % (name, R.choice([-1, numel, numel + 5, 1 << 40]), R.randrange(2)))
            if numel > 2: ops.append('slicef %s %d %d %d' % (name, R.randrange(numel // 2), R.randrange(0, numel // 2 + 1), R.randrange(2)))
        if dt in (0, 1, 2, 5): ops.append('scale %s %d %d' % (name, R.choice([numel, numel + 3]), R.randrange(2)))
        ops.append('raw %s %d %d' % (name, R.choice([-1, nb, nb + 9]), R.randrange(2)))
        if nb > 0:
            bo = R.randrange(nb); n = R.randrange(0, nb - bo + 1); ops.append('slice %s %d %d %d %d' % (name, bo, n, n + R.choice([0, 3]), R.randrange(2)))
            ops.append('slice %s %d 0 0 0' % (name, nb))
            ops.append('map %s' % name)
        ops.append('prefetch %s %d' % (name, R.choice([0, 1, 2, 7])))
        if sh < nshard:
            sz = sizes.get(sh, 0)
            if sz:
                o = R.randrange(sz); n = R.randrange(0, min(sz - o, 70000) + 1)
                ops.append('range %d %d %d %d %d' % (sh, o, n, n + R.choice([0, 5]), R.randrange(2)))
                for _ in range(2):
                    o = R.choice([R.randrange(sz), off, off + R.randrange(1, 4096), 0, 4096 * R.randrange(max(1, sz // 4096))]); o = min(o, sz)
                    n = R.randrange(0, min(sz - o, R.choice([100, 5000, 70000])) + 1)
                    ops.append('rrep %d %d %d %d %d %d %d' % (sh, R.choice([0, 0, 1, 2, 9]), o, n, n + R.choice([0, 7]), R.randrange(2), R.choice([0, 1, 1, 2])))
                ops.append('shardmap %d %d %d' % (sh, R.randrange(sz), 1))
                ops.append('shardmap %d %d %d' % (sh, 0, sz))
                ops.append('shardmap %d %d %d' % (sh, 0, sz + 1))
    for s in range(nshard): ops.append('noreadahead %d' % s)
    if kind in ('out', 'many'): ops.append('missing %s' % hexs(R.choice(['model.norm.weight', 'lm_head.weight', 'x.y'])))
    elif R.random() < .6 and table:                              # one operation that is expected to be refused
        name, dt, off, nb, numel, sh = R.choice(table); sz = sizes.get(sh, 1)
        ops.append(R.choice([
            'f32 %s %d 0' % (name, max(0, numel - 1)), 'raw %s %d 0' % (name, max(0, nb - 1)), 'scale %s %d 0' % (name, numel + 2),
            'slice %s %d %d %d 0' % (name, nb + 1, 1, 8), 'slice %s -1 1 8 0' % name, 'slice %s 0 %d %d 0' % (name, nb + 1, nb + 5), 'slice %s 0 5 2 0' % name,
            'slicef %s %d %d 0' % (name, numel, 1), 'slicef %s -1 1 0' % name, 'slicef %s 0 %d 0' % (name, numel + 1),
            'range %d %d %d %d 0' % (sh, sz, 1, 5), 'range %d -1 4 8 0' % sh, 'range 99 0 1 1 0', 'rrep %d 0 %d 8 4 0 1' % (sh, sz), 'rrep 99 0 0 1 1 0 0',
            'missing %s' % hexs('model.embed_tokens.weight'), 'missing %s' % hexs('model.norm.weight'), 'missing %s' % hexs('lm_head.weight'),
            'f32 %s -1 0' % hexs('nope')]))
    return ops

bad = 0; nok = 0; nerr = 0; nops = 0
for k in range(N):
    base, prim, extra, mirrors, env = build_case(k)
    open(base + '/empty.ops', 'w').write('')
    c = run(w + '/c_ext', prim, extra, base + '/empty.ops', env); v = run(w + '/v_ext', prim, extra, base + '/empty.ops', env)
    if c != v:
        bad += 1; print('TABLE DIFFERENT', base, c[0], v[0]); print('\n'.join(list(difflib.unified_diff(c[2].decode().split('\n'), v[2].decode().split('\n'), 'C', 'V', lineterm=''))[:20]))
        if bad >= 4: break
        continue
    if c[0] != 0:
        nerr += 1
        if os.environ.get('ST_EXT_WHY'): print('refused:', c[2].decode().strip().split('\n')[-1][:140])
        continue
    table = []; shard_names = {}
    for line in c[1].decode().split('\n'):
        if line.startswith('T '):
            f = line.split(); name = f[1]; dt = int(f[2]); off = int(f[3]); nb = int(f[4]); numel = int(f[5]); sh = int(f[-1]); shard_names[sh] = f[-2]
            table.append((name, dt, off, nb, numel, sh))
    sizes = {}
    for sh, bn in shard_names.items():
        for d in [prim] + ([] if extra is None else [p.strip() for p in extra.replace(';', ',').split(',') if p.strip()]):
            p = os.path.join(d, bn)
            if os.path.exists(p): sizes[sh] = os.path.getsize(p); break
    nshard = max(shard_names) + 1 if shard_names else 1
    ops = make_ops(table, sizes, mirrors, nshard, env.get('_KIND', ''))
    open(base + '/ops', 'w').write('\n'.join(ops) + '\n'); nops += len(ops)
    c = run(w + '/c_ext', prim, extra, base + '/ops', env); v = run(w + '/v_ext', prim, extra, base + '/ops', env)
    if c != v:
        bad += 1; print('DIFFERENT', base)
        for i, (a, b) in enumerate(zip(c[1].decode().split('\n'), v[1].decode().split('\n'))):
            if a != b: print(' stdout line', i, '\n  C:', a, '\n  V:', b); break
        else:
            print(' status', c[0], v[0], '\n  C err:', c[2].decode()[-400:], '\n  V err:', v[2].decode()[-400:])
        if bad >= 4: break
    else: nok += 1
if bad: print('st_ext differential: DIFFERENT, kept in', w); sys.exit(1)
print('st_ext differential: %d cases (%d directories refused alike at open, %d ran %d operations), C and V identical' % (N, nerr, nok, nops)); shutil.rmtree(w)
