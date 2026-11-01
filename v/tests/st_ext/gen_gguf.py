"""Writes GGUF model folders (gguf.json + .gguf files) for the st_ext differential test: every architecture map of
st_gguf.h with its tensor names (layers, fused expert tensors, aliases), split parts, a DSpark head, the colibri
packed-expert container, and the ways a header can be wrong."""
import json, os, re, struct

HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, '..', '..', '..', 'c')
_s = open(os.path.join(C, 'st_gguf.h')).read()
TABS = {m.group(1): re.findall(r'\{"([^"]*)",\s*"([^"]*)"\}', m.group(2))
        for m in re.finditer(r'static const st_gg_name (st_gg_\w+)\[\] = \{(.*?)\n\s*\{NULL, NULL\}\};', _s, re.S)}
MAPS = {a: (fmt, TABS[g], TABS[l]) for a, fmt, g, l in re.findall(r'\{"([^"]+)", "([^"]+)", (st_gg_\w+), (st_gg_\w+)\}', _s)}
# ggml type -> (block elements, block bytes); the unsupported / unknown ones have no entry
TYPES = {0: (1, 4), 1: (1, 2), 30: (1, 2), 2: (32, 18), 8: (32, 34), 12: (256, 144), 17: (256, 74), 39: (32, 17), 24: (1, 1), 26: (1, 4), 27: (1, 8), 14: (256, 210)}
T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 = range(13)

def w_str(s):
    b = s.encode() if isinstance(s, str) else s
    return struct.pack('<Q', len(b)) + b
FMT = {T_U8: '<B', T_I8: '<b', T_U16: '<H', T_I16: '<h', T_U32: '<I', T_I32: '<i', T_F32: '<f', T_BOOL: '<B', T_U64: '<Q', T_I64: '<q', T_F64: '<d'}
def w_val(typ, v):
    if typ == T_STR: return w_str(v)
    return struct.pack(FMT[typ], v)
def kv(key, typ, v, atype=None):
    out = w_str(key) + struct.pack('<I', typ)
    if typ == T_ARR:
        out += struct.pack('<IQ', atype, len(v)) + b''.join(w_val(atype, x) for x in v)
    else: out += w_val(typ, v)
    return out

def nbytes(typ, ne):
    blk, by = TYPES[typ]
    n = 1
    for d in ne[1:]: n *= d
    return ne[0] // blk * by * n

def gguf_bytes(R, kvs, tensors, align=32, version=3, nt=None, nkv=None, magic=b'GGUF', pad_extra=0, force_off=None):
    """tensors: (name, ne, type, data or None); kvs: list of kv() blobs"""
    info = b''; data = b''; offs = []
    for name, ne, typ, d in tensors:
        if typ in TYPES and d is None: d = bytes(R.randrange(256) for _ in range(nbytes(typ, ne)))
        if d is None: d = b''
        data += b'\0' * ((-len(data)) % align); offs.append(len(data)); data += d
    for i, ((name, ne, typ, d), off) in enumerate(zip(tensors, offs)):
        if force_off and i in force_off: off = force_off[i]
        info += w_str(name) + struct.pack('<I', len(ne)) + b''.join(struct.pack('<Q', x) for x in ne) + struct.pack('<IQ', typ, off)
    head = magic + struct.pack('<IQQ', version, len(tensors) if nt is None else nt, len(kvs) if nkv is None else nkv) + b''.join(kvs) + info
    head += b'\0' * ((-len(head)) % align)
    return head + data + b'\0' * pad_extra

def arch_tensors(R, arch, nlayers=3, last=0, odd=False):
    fmt, glob, layer = MAPS[arch]; ts = []
    def mk(name, expert=False, rows=None, cols=None):
        typ = R.choice([0, 1, 30, 2, 8, 12, 17, 39, 14, 24, 26, 27]); blk = TYPES[typ][0]
        cols = (cols or R.choice([1, 2, 4, 8])) * blk; rows = rows or R.choice([1, 2, 3])
        ne = [cols, rows] if not expert else [cols, rows, R.choice([1, 2, 4])]
        if not expert and R.random() < .3: ne = [cols]
        return (name, ne, typ, None)
    for g, h in R.sample(glob, min(len(glob), R.choice([1, 3, 6]))): ts.append(mk(g))
    for L in range(nlayers):
        for g, h in R.sample(layer, min(len(layer), R.choice([2, 5, 9]))):
            ts.append(mk('blk.%d.%s' % (L, g), expert='%d' in h))
    ts.append(mk('some.unmapped.tensor')); ts.append(mk('blk.%d.not_in_map.weight' % nlayers))
    if odd:
        for nm in ['blk.x.attn_norm.weight', 'blk.-1.attn_norm.weight', 'blk. 2.attn_norm.weight', 'blk.3attn_norm.weight', 'blk.4.', 'blk.5', 'blk.6x', 'blk.7.attn_norm.weight']: ts.append(mk(nm))
    if R.random() < .3: ts.append(mk(R.choice(['blk.x.attn_norm.weight', 'blk.-1.attn_norm.weight', 'blk. 2.attn_norm.weight', 'blk.3attn_norm.weight', 'blk.4.', 'blk.5', 'blk.6x'])))
    return ts

def packed_container(R, arch, nlayers=2):
    """colibri.layout expert-packed-v1: per layer a 2-d tensor blk.N.ffn_exps_colibri [stride, experts]"""
    fmt, glob, layer = MAPS[arch]
    types, bytes_, strides, dims, ts = [], [], [], [], []
    E = R.choice([1, 2, 3])
    for L in range(nlayers):
        inner = 0
        for k in range(3):
            typ = R.choice([8, 12, 17]); blk = TYPES[typ][0]; cols = blk * R.choice([1, 2]); rows = R.choice([1, 2, 4])
            b = nbytes(typ, [cols, rows]); types.append(typ); bytes_.append(b); dims += [cols, rows]; inner += b
        stride = (inner + 4095) // 4096 * 4096 if R.random() < .5 else inner + R.choice([0, 8])
        strides.append(stride)
        ts.append(('blk.%d.ffn_exps_colibri' % L, [stride, E], 24, None))   # I8 rows: stride bytes each
    return types, bytes_, strides, dims, ts, E

BAD_VARIANTS = ['bigdim_a', 'bigdim_b', 'trunc', 'magic', 'ver', 'counts', 'align', 'dim0', 'type', 'dup', 'offs', 'nd', 'kvtype', 'strarr', 'badtail', 'overflow', 'bigoff', 'misaligned']
HEAD_STAGES = [2, 3, 1, 0, 17, 16]

def write_folder(R, d, case=None, variant=None):
    """creates a GGUF folder in d; returns nothing (the dumpers read it)"""
    os.makedirs(d, exist_ok=True)
    case = case or R.choice(['plain'] * 4 + ['split', 'head', 'head', 'packed', 'packed', 'bad', 'bad', 'bad', 'bad', 'json', 'bigsplit'])
    arch = R.choice(list(MAPS)); align = R.choice([32, 32, 64, 4096]); extra = []
    kvs = [kv('general.architecture', T_STR, arch)]
    if align != 32 or R.random() < .3: kvs.append(kv('general.alignment', T_U32, align))
    for key, typ, val in R.sample([('general.name', T_STR, 'x'), ('n.u8', T_U8, 200), ('n.i8', T_I8, -5), ('n.u16', T_U16, 65535), ('n.i16', T_I16, -300),
                                    ('n.u32', T_U32, 4000000000), ('n.i32', T_I32, -7), ('n.f32', T_F32, 1.5), ('n.bool', T_BOOL, 1), ('n.u64', T_U64, 2**63 + 5),
                                    ('n.i64', T_I64, -2**40), ('n.f64', T_F64, 3.25)], R.choice([0, 3, 12])): kvs.append(kv(key, typ, val))
    if R.random() < .5: kvs.append(kv('tok.tokens', T_ARR, ['a', 'bb', 'ccc'][:R.choice([0, 3])], T_STR))
    if R.random() < .5: kvs.append(kv('tok.scores', T_ARR, [R.random() for _ in range(R.choice([0, 4]))], T_F32))
    if R.random() < .3: kvs.append(kv('tok.ids', T_ARR, [R.randrange(10**6) for _ in range(3)], T_U32))
    files = []; js = {}
    if case == 'packed':
        arch = R.choice([a for a in MAPS if any('%d' in h for g, h in MAPS[a][2] if g == 'ffn_gate_exps.weight')] or ['deepseek4'])
        kvs[0] = kv('general.architecture', T_STR, arch)
        types, by, st, dims, pts, E = packed_container(R, arch)
        kvs += [kv('colibri.layout', T_STR, 'expert-packed-v1'), kv('colibri.expert_types', T_ARR, types, T_U32), kv('colibri.expert_bytes', T_ARR, by, T_U64),
                kv('colibri.expert_stride', T_ARR, st, T_U64), kv('colibri.expert_dims', T_ARR, dims, T_U64)]
        ts = arch_tensors(R, arch, 1)
        ts += pts
        # layout corruptions
        v = R.choice(['ok', 'ok', 'ok', 'badlayout', 'nodims', 'badtypes', 'badbytes', 'badstride'])
        if v == 'badlayout': kvs[-5] = kv('colibri.layout', T_STR, 'expert-packed-v2')
        if v == 'nodims': kvs.pop()
        if v == 'badtypes': kvs[-4] = kv('colibri.expert_types', T_ARR, types[:-1], T_U32)
        if v == 'badbytes': kvs[-3] = kv('colibri.expert_bytes', T_ARR, [x + 1 for x in by], T_U64)
        if v == 'badstride': kvs[-2] = kv('colibri.expert_stride', T_ARR, [x + 64 for x in st], T_U64)
        files.append(('model.gguf', gguf_bytes(R, kvs, ts, align)))
    else:
        ts = arch_tensors(R, arch, odd=(case == 'names'))
        if case == 'bigsplit':
            n = R.choice([99, 100, 120]); parts = [[] for _ in range(n)]
            for t in ts: R.choice(parts).append(t)
            for i, p in enumerate(parts):
                files.append(('model-%05d-of-%05d.gguf' % (i + 1, n), gguf_bytes(R, kvs if i == 0 else [], p, align)))
            js['gguf'] = 'model-00001-of-%05d.gguf' % n
        elif case == 'split':
            n = R.choice([2, 3]); parts = [[] for _ in range(n)]
            for t in ts: R.choice(parts).append(t)
            for i, p in enumerate(parts):
                files.append(('model-%05d-of-%05d.gguf' % (i + 1, n), gguf_bytes(R, kvs if i == 0 else [kv('general.alignment', T_U32, align)] if align != 32 else [], p, align)))
            js['gguf'] = 'model-00001-of-%05d.gguf' % n
            if R.random() < .2: js['gguf'] = os.path.join(d, js['gguf'])
        elif case == 'bad':
            v = variant if variant in BAD_VARIANTS else R.choice(BAD_VARIANTS)
            raw = None
            if v == 'trunc': raw = gguf_bytes(R, kvs, ts, align); raw = raw[:R.choice([3, 10, 30, len(raw) // 2, len(raw) - 5])]
            if v == 'magic': raw = gguf_bytes(R, kvs, ts, align, magic=R.choice([b'GGUX', b'gguf']))
            if v == 'ver': raw = gguf_bytes(R, kvs, ts, align, version=R.choice([1, 4, 99]))
            if v == 'counts': raw = gguf_bytes(R, kvs, ts, align, nt=R.choice([1 << 25, 10**6, len(ts) + 3]), nkv=R.choice([1 << 21, len(kvs) + 2]))
            if v == 'align': raw = gguf_bytes(R, kvs[:1] + [kv('general.alignment', T_U32, R.choice([0, 3, 1 << 21, 100, 6, 96]))], ts, 32)
            if v == 'dim0': ts[0] = (ts[0][0], [0] + ts[0][1][1:], ts[0][2], b''); raw = gguf_bytes(R, kvs, ts, align)
            if v == 'type': ts[0] = (ts[0][0], [3], R.choice([4, 5, 31, 99, 500]), b'abc'); raw = gguf_bytes(R, kvs, ts, align)
            if v == 'dup': ts = ts + [ts[0]]; raw = gguf_bytes(R, kvs, ts, align)
            if v == 'offs': raw = gguf_bytes(R, kvs, ts, align); raw = raw[:-R.choice([1, 40])]
            if v.startswith('bigdim'): ts[0] = (ts[0][0], [{'bigdim_a': 2**40 + 1, 'bigdim_b': 2**41}.get(v, R.choice([2**40 + 1, 2**41, 2**40]))] + ts[0][1][1:], 24, b''); raw = gguf_bytes(R, kvs, ts, align)
            if v == 'overflow': ts[0] = (ts[0][0], [2**40, 2**40], 24, b''); raw = gguf_bytes(R, kvs, ts, align)
            if v == 'bigoff': raw = gguf_bytes(R, kvs, ts, align, force_off={0: R.choice([2**60, 2**63, align * 1000000])})
            if v == 'misaligned': raw = gguf_bytes(R, kvs, ts, align, force_off={len(ts) - 1: 1})
            if v == 'nd': raw = gguf_bytes(R, kvs, [(ts[0][0], [1] * 5, 0, b'')] + ts[1:], align)
            if v == 'kvtype': raw = gguf_bytes(R, kvs + [w_str('bad.key') + struct.pack('<I', R.choice([13, 40]))], ts, align)
            if v == 'strarr': raw = gguf_bytes(R, kvs + [w_str('big.arr') + struct.pack('<IIQ', T_ARR, T_STR, 1 << 40)], ts, align)
            if v == 'badtail': raw = gguf_bytes(R, kvs + [w_str('big.arr') + struct.pack('<IIQ', T_ARR, T_U8, 1 << 40)], ts, align)
            files.append(('model.gguf', raw))
        else:
            files.append(('model.gguf', gguf_bytes(R, kvs, ts, align)))
        if case == 'head':
            arch = 'deepseek41'; kvs[0] = kv('general.architecture', T_STR, arch)
            files[0] = ('model.gguf', gguf_bytes(R, kvs, arch_tensors(R, arch), align))
            fmt, glob, layer = MAPS['dflash']
            hts = [(g, [32, 2], 8, None) for g, h in R.sample(glob, R.choice([1, 3, 6]))]
            dl = MAPS['deepseek41'][2]
            for s_ in range(R.choice([1, 2, 3])): hts += [('blk.%d.%s' % (s_, g), [32, 1], 8, None) for g, h in R.sample(dl, 3)]
            hk = [kv('general.architecture', T_STR, 'dflash' if isinstance(variant, int) else R.choice(['dflash'] * 4 + ['llama'])), kv('dflash.block_count', T_I32, variant if isinstance(variant, int) else R.choice([2, 3, 1, 0, 17, 16, 16]))]
            files.append(('head.gguf', gguf_bytes(R, hk, hts, 32)))
            js['dspark'] = 'head.gguf'
    js.setdefault('gguf', files[0][0])
    for name, b in files: open(os.path.join(d, name), 'wb').write(b)
    if case == 'json':
        v = R.choice(['nogguf', 'nonstr', 'empty', 'long', 'nofile', 'abs_missing'])
        if v == 'nogguf': js = {'other': 1}
        if v == 'nonstr': js = {'gguf': 5}
        if v == 'empty': js = {'gguf': ''}
        if v == 'nofile': js['gguf'] = 'missing.gguf'
        if v == 'abs_missing': js['gguf'] = '/nonexistent/x.gguf'
        txt = json.dumps(js)
        if v == 'long': txt = '{"pad": "%s", "gguf": "model.gguf"}' % ('x' * 16400)
        open(os.path.join(d, 'gguf.json'), 'w').write(txt)
    else:
        open(os.path.join(d, 'gguf.json'), 'w').write(json.dumps(js))
    return case, arch
