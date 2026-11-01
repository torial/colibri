#!/usr/bin/env python3
"""Writes a tiny OLMoE as a GGUF folder (gguf.json + model.gguf + config.json + ref.json), the form c/olmoe.c reads through
st_gguf.h: dense matrices and the routed experts in ggml block types, each tensor in a type of its own."""
import importlib.util, json, os, struct
import numpy as np
here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('gen_gguf', here + '/../st_ext/gen_gguf.py'); gg = importlib.util.module_from_spec(spec); spec.loader.exec_module(gg)

# ggml type id -> (elements per block, bytes per block, byte offsets of f16 scales that must be small and finite)
TYPES = {0: (1, 4, []), 1: (1, 2, []), 30: (1, 2, []), 8: (32, 34, [0]), 2: (32, 18, [0]), 6: (32, 22, [0]), 12: (256, 144, [0, 2]),
         13: (256, 176, [0, 2]), 14: (256, 210, [208]), 10: (256, 84, [80, 82]), 11: (256, 110, [108]), 20: (32, 18, [0]), 39: (32, 17, [])}
SMALL = [0, 1, 30, 8, 2, 6, 20, 39]          # types whose blocks are 1 or 32 elements wide
ALL = list(TYPES)

def f16(x): return np.float16(x).tobytes()

def blocks(rs, typ, n_elems, amp=1.0):
    """valid blocks of ggml type typ covering n_elems values (random codes, small finite scales)"""
    blk, by, scales = TYPES[typ]; nb = n_elems // blk
    if typ == 0: return (rs.randn(n_elems) * 0.2 * amp).astype(np.float32).tobytes()
    if typ == 1: return (rs.randn(n_elems) * 0.2 * amp).astype(np.float16).tobytes()
    if typ == 30:
        u = (rs.randn(n_elems) * 0.2 * amp).astype(np.float32).view(np.uint32); return (u >> 16).astype(np.uint16).tobytes()
    out = bytearray()
    for _ in range(nb):
        b = bytearray(rs.randint(0, 256, size=by).astype(np.uint8).tobytes())
        for off in scales: b[off:off + 2] = f16(rs.uniform(0.0005, 0.004) * amp)
        if typ == 39: b[0] = int(rs.randint(118, 126))          # e8m0 exponent: 2^-9 .. 2^-1
        out += b
    return bytes(out)

def nbytes(typ, n_elems): blk, by, _ = TYPES[typ]; return n_elems // blk * by

def make(d, seed=1, big=True, L=2, E=4, K=2, V=32, prompt_len=5, n_new=5, norm_topk=False, theta=10000.0, eps=1e-5):
    rs = np.random.RandomState(seed); os.makedirs(d, exist_ok=True)
    D = 256 if big else 64; H = 4; I = 256 if big else 64
    kinds = ALL if big else SMALL
    cfg = {'hidden_size': D, 'num_hidden_layers': L, 'num_attention_heads': H, 'num_key_value_heads': H, 'num_experts': E, 'num_experts_per_tok': K,
           'intermediate_size': I, 'vocab_size': V, 'rope_theta': theta, 'rms_norm_eps': eps, 'norm_topk_prob': norm_topk}
    json.dump(cfg, open(d + '/config.json', 'w'))
    T = []
    def mat(name, rows, cols, extra=(), amp=1.0, types=None):
        typ = int(rs.choice(types or kinds)); ne = [cols, rows] + list(extra)
        n = rows * cols * int(np.prod(extra) if extra else 1)
        T.append((name, ne, typ, blocks(rs, typ, n, amp)))
    def vec(name, n, center=1.0, scale=0.1):
        T.append((name, [n], 0, (rs.randn(n) * scale + center).astype(np.float32).tobytes()))
    mat('token_embd.weight', V, D, amp=60.0); vec('output_norm.weight', D); mat('output.weight', V, D, amp=30.0)
    for l in range(L):
        p = 'blk.%d.' % l
        vec(p + 'attn_norm.weight', D); vec(p + 'ffn_norm.weight', D)
        for n in ('q', 'k', 'v'): mat(p + 'attn_%s.weight' % n, D, D, amp=40.0)
        mat(p + 'attn_output.weight', D, D, amp=40.0)
        vec(p + 'attn_q_norm.weight', D); vec(p + 'attn_k_norm.weight', D)
        T.append((p + 'ffn_gate_inp.weight', [D, E], int(rs.choice([0, 1])), None))      # the router: float
        typ = T[-1][2]; T[-1] = (T[-1][0], T[-1][1], typ, blocks(rs, typ, D * E, 8.0))
        mat(p + 'ffn_gate_exps.weight', I, D, (E,), amp=40.0); mat(p + 'ffn_up_exps.weight', I, D, (E,), amp=40.0)
        mat(p + 'ffn_down_exps.weight', D, I, (E,), amp=40.0)
    kvs = [gg.kv('general.architecture', gg.T_STR, 'olmoe')]
    open(d + '/model.gguf', 'wb').write(gg.gguf_bytes(np.random, kvs, T, 32))
    open(d + '/gguf.json', 'w').write(json.dumps({'gguf': 'model.gguf'}))
    prompt = [int(x) for x in rs.randint(0, V, size=prompt_len)]
    json.dump({'prompt_ids': prompt, 'full_ids': prompt + [int(x) for x in rs.randint(0, V, size=n_new)]}, open(d + '/ref.json', 'w'))
    return cfg, {t[0]: t[2] for t in T}

if __name__ == '__main__':
    import sys; print(make(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1)[1])
