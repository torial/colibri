#!/usr/bin/env python3
"""Writes a tiny GLM-5.2-style checkpoint (glm_moe_dsa: MLA attention, sigmoid router with a selection bias, shared expert,
leading dense layers) as plain HF tensors, F32 or BF16, plus config.json and a ref.json for c/colibri.c's oracle mode."""
import json, os, struct
import numpy as np

def st_bytes(T):
    hdr = {}; data = b''
    for n, (dt, shape, raw) in T.items():
        a = len(data); data += raw; hdr[n] = {'dtype': dt, 'shape': shape, 'data_offsets': [a, len(data)]}
    h = json.dumps(hdr).encode(); h += b' ' * ((-len(h)) % 8)
    return struct.pack('<Q', len(h)) + h + data

def bf16(a):
    u = np.asarray(a, dtype=np.float32).view(np.uint32); return ((u + 0x7fff + ((u >> 16) & 1)) >> 16).astype(np.uint16).tobytes()

def make(d, seed=1, D=32, H=4, L=3, first_dense=1, E=8, K=2, mi=16, di=24, ql=16, kvl=16, nope=8, rope=8, vh=8, shared=1, V=48,
         bf=False, norm_topk=True, routed_scale=1.5, theta=10000.0, eps=1e-5, prompt_len=5, n_new=5, shards=1):
    rs = np.random.RandomState(seed); os.makedirs(d, exist_ok=True); qk = nope + rope
    cfg = {'model_type': 'glm_moe_dsa', 'hidden_size': D, 'num_hidden_layers': L, 'num_attention_heads': H, 'n_routed_experts': E,
           'num_experts_per_tok': K, 'moe_intermediate_size': mi, 'intermediate_size': di, 'first_k_dense_replace': first_dense,
           'q_lora_rank': ql, 'kv_lora_rank': kvl, 'qk_nope_head_dim': nope, 'qk_rope_head_dim': rope, 'v_head_dim': vh,
           'n_shared_experts': shared, 'vocab_size': V, 'n_group': 1, 'topk_group': 1, 'norm_topk_prob': norm_topk, 'rms_norm_eps': eps,
           'routed_scaling_factor': routed_scale, 'rope_parameters': {'rope_theta': theta}, 'eos_token_id': [V - 1],
           'index_topk': 0, 'index_n_heads': 0, 'index_head_dim': 0}
    json.dump(cfg, open(d + '/config.json', 'w'))
    T = {}
    def w(name, shape, scale=0.2, center=0.0, f32=False):
        a = (rs.randn(*shape) * scale + center).astype(np.float32)
        T[name] = ('F32', list(shape), a.tobytes()) if (f32 or not bf) else ('BF16', list(shape), bf16(a))
    w('model.embed_tokens.weight', (V, D), 1.0); w('lm_head.weight', (V, D), 0.5); w('model.norm.weight', (D,), 0.1, 1.0)
    for l in range(L):
        p = 'model.layers.%d.' % l
        w(p + 'input_layernorm.weight', (D,), 0.1, 1.0); w(p + 'post_attention_layernorm.weight', (D,), 0.1, 1.0)
        w(p + 'self_attn.q_a_proj.weight', (ql, D), 0.3); w(p + 'self_attn.q_a_layernorm.weight', (ql,), 0.1, 1.0)
        w(p + 'self_attn.q_b_proj.weight', (H * qk, ql), 0.3)
        w(p + 'self_attn.kv_a_proj_with_mqa.weight', (kvl + rope, D), 0.3); w(p + 'self_attn.kv_a_layernorm.weight', (kvl,), 0.1, 1.0)
        w(p + 'self_attn.kv_b_proj.weight', (H * (nope + vh), kvl), 0.3); w(p + 'self_attn.o_proj.weight', (D, H * vh), 0.3)
        if l < first_dense:
            w(p + 'mlp.gate_proj.weight', (di, D), 0.3); w(p + 'mlp.up_proj.weight', (di, D), 0.3); w(p + 'mlp.down_proj.weight', (D, di), 0.3)
        else:
            w(p + 'mlp.gate.weight', (E, D), 0.8, f32=True); w(p + 'mlp.gate.e_score_correction_bias', (E,), 0.3, f32=True)
            sI = mi * shared
            w(p + 'mlp.shared_experts.gate_proj.weight', (sI, D), 0.3); w(p + 'mlp.shared_experts.up_proj.weight', (sI, D), 0.3)
            w(p + 'mlp.shared_experts.down_proj.weight', (D, sI), 0.3)
            for e in range(E):
                q = p + 'mlp.experts.%d.' % e
                w(q + 'gate_proj.weight', (mi, D), 0.3); w(q + 'up_proj.weight', (mi, D), 0.3); w(q + 'down_proj.weight', (D, mi), 0.3)
    names = sorted(T)
    for s in range(shards):
        open(d + '/model-%05d-of-%05d.safetensors' % (s + 1, shards), 'wb').write(st_bytes({n: T[n] for i, n in enumerate(names) if i % shards == s}))
    prompt = [int(x) for x in rs.randint(0, V - 1, size=prompt_len)]
    json.dump({'prompt_ids': prompt, 'full_ids': prompt + [int(x) for x in rs.randint(0, V - 1, size=n_new)]}, open(d + '/ref.json', 'w'))
    return cfg

if __name__ == '__main__':
    import sys; make(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1, bf='--bf16' in sys.argv)
