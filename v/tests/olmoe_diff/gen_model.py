#!/usr/bin/env python3
"""Writes a tiny OLMoE checkpoint in the converted merged-int8 layout the C engine reads (c/olmoe.c): config.json,
one or more safetensors shards, and a ref.json (prompt and a reference continuation) for the C engine's harness mode.
usage: gen_model.py DIR [seed] [--bf16] [--shards N] [--norm-topk] ..."""
import json, os, random, struct, sys
import numpy as np

def st_bytes(tensors):
    hdr = {}; data = b''
    for name, (dt, shape, raw) in tensors.items():
        a = len(data); data += raw; hdr[name] = {'dtype': dt, 'shape': shape, 'data_offsets': [a, len(data)]}
    h = json.dumps(hdr).encode(); h += b' ' * ((-len(h)) % 8)
    return struct.pack('<Q', len(h)) + h + data

def bf16(a):
    u = np.asarray(a, dtype=np.float32).view(np.uint32)
    r = ((u + 0x7fff + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    return r.tobytes()

def make(d, seed=1, D=32, H=4, L=2, E=8, K=2, I=16, V=64, bf=False, shards=1, norm_topk=False, tie_gate=False, theta=10000.0, eps=1e-5, prompt_len=5, n_new=6):
    rs = np.random.RandomState(seed); os.makedirs(d, exist_ok=True)
    cfg = {'hidden_size': D, 'num_hidden_layers': L, 'num_attention_heads': H, 'num_key_value_heads': H, 'num_experts': E,
           'num_experts_per_tok': K, 'intermediate_size': I, 'vocab_size': V, 'rope_theta': theta, 'rms_norm_eps': eps, 'norm_topk_prob': norm_topk}
    json.dump(cfg, open(d + '/config.json', 'w'))
    T = {}
    def f32(name, shape, scale=0.2, center=0.0):
        a = (rs.randn(*shape) * scale + center).astype(np.float32)
        if bf:
            T[name] = ('BF16', list(shape), bf16(a))
        else:
            T[name] = ('F32', list(shape), a.tobytes())
    def f32only(name, shape, scale=0.2, center=0.0):
        a = (rs.randn(*shape) * scale + center).astype(np.float32); T[name] = ('F32', list(shape), a.tobytes())
    f32('model.embed_tokens.weight', (V, D), 1.0); f32('lm_head.weight', (V, D), 0.5); f32('model.norm.weight', (D,), 0.1, 1.0)
    for l in range(L):
        p = 'model.layers.%d.' % l
        f32(p + 'input_layernorm.weight', (D,), 0.1, 1.0); f32(p + 'post_attention_layernorm.weight', (D,), 0.1, 1.0)
        for n in ('q', 'k', 'v', 'o'): f32(p + 'self_attn.%s_proj.weight' % n, (D, D), 0.25)
        f32(p + 'self_attn.q_norm.weight', (D,), 0.1, 1.0); f32(p + 'self_attn.k_norm.weight', (D,), 0.1, 1.0)
        f32(p + 'mlp.gate.weight', (E, D), 0.8)
        if tie_gate:                                    # two identical router rows: exactly equal probabilities, the top-k tie-break decides
            nm = p + 'mlp.gate.weight'; dt, shp, raw = T[nm]; esz = 2 if dt == 'BF16' else 4; rw = D * esz
            T[nm] = (dt, shp, raw[:rw] + raw[:rw] + raw[2 * rw:])
        for e in range(E):
            w = rs.randint(-127, 128, size=(2 * I * D + D * I,)).astype(np.int8)
            T[p + 'mlp.experts.%d.merged_weight' % e] = ('I8', [int(w.size)], w.tobytes())
            qs = (rs.rand(2 * I + D) * 0.004 + 0.001).astype(np.float32)
            T[p + 'mlp.experts.%d.qs' % e] = ('F32', [int(qs.size)], qs.tobytes())
    names = sorted(T)
    for s in range(shards):
        part = {n: T[n] for i, n in enumerate(names) if i % shards == s}
        open(d + '/model-%05d-of-%05d.safetensors' % (s + 1, shards), 'wb').write(st_bytes(part))
    prompt = [int(x) for x in rs.randint(0, V, size=prompt_len)]
    full = prompt + [int(x) for x in rs.randint(0, V, size=n_new)]
    json.dump({'prompt_ids': prompt, 'full_ids': full}, open(d + '/ref.json', 'w'))
    return cfg

if __name__ == '__main__':
    a = sys.argv[1:]; kw = {}
    if '--bf16' in a: kw['bf'] = True
    if '--norm-topk' in a: kw['norm_topk'] = True
    if '--shards' in a: kw['shards'] = int(a[a.index('--shards') + 1])
    make(a[0], int(a[1]) if len(a) > 1 and a[1].isdigit() else 1, **kw)
