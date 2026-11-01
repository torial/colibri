#!/usr/bin/env python3
"""Back-of-envelope cost of MiniMax-H3 (H3-Base, 768p) on a CPU, from the published config.

Facts used (MiniMaxAI/MiniMax-H3, transformer/config.json, vae/config.json, README):
  hidden 5376, 50 blocks, 56 heads x 128, ffn 14336, full attention (sparse not released),
  visual VAE f16 t4 (24 ch), patch 1x2x2 -> 32x spatial / 4x temporal per token, 24 fps,
  audio 40 Hz latent tokens, CFG-distilled (one forward per step), 33.1B parameters of which
  ~13B are AdaLN branches that inference does not need (their outputs can be cached).
NOT known from the files: the number of denoising steps (see the SGLang cookbook), whether
the FFN is gated (it changes the inference parameter count between ~13.5B and ~17.4B).

usage: h3_cost_model.py [--cores 28] [--ghz 2.3] [--eff 0.6] [--avx512-gain 1.0] [--steps 30]
"""
import argparse, math

def tokens(duration_s, short_edge=768, aspect=(16, 9), fps=24):
    h = short_edge
    w = round(short_edge * aspect[0] / aspect[1])
    frames = int(duration_s * fps)
    lat_t = (frames - 1) // 4 + 1                      # temporal compression 4, causal first frame
    per_frame = math.ceil(w / 32) * math.ceil(h / 32)  # 16x VAE, 2x2 patch
    return lat_t * per_frame + int(duration_s * 40), lat_t, per_frame

def step_flops(n, params_inf, d=5376, layers=50):
    linear = 2.0 * params_inf * n
    attn = 4.0 * n * n * d * layers                    # QK^T and PV, full attention
    return linear, attn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cores", type=int, default=28)
    ap.add_argument("--ghz", type=float, default=2.3, help="sustained AVX2 all-core clock")
    ap.add_argument("--eff", type=float, default=0.6, help="fraction of FMA peak a good kernel reaches (qi_gemm: ~0.63 on the VM)")
    ap.add_argument("--avx512-gain", type=float, default=1.0, help="multiplier if an AVX-512 kernel beats AVX2 (two FMA units: up to ~1.7)")
    ap.add_argument("--steps", type=int, default=30)
    a = ap.parse_args()
    peak = a.cores * a.ghz * 1e9 * 32          # 2 FMA ports x 8 f32 lanes x 2 flop
    rate = peak * a.eff * a.avx512_gain
    print(f"machine: {a.cores} cores x {a.ghz} GHz, AVX2 peak {peak/1e12:.2f} TFLOP/s, assumed {rate/1e12:.2f} TFLOP/s sustained")
    print(f"weights: bf16 ~2 bytes/param, int8 ~1: inference part ~14-17B params -> ~28-35 GB bf16, ~14-17 GB int8 (+ Qwen3-VL-32B encoder, first 50 of 64 layers)\n")
    print(f"{'clip':>6} {'tokens':>8} {'latent T':>9} | {'per step: linear':>17} {'attention':>11} {'minutes':>8} | {a.steps} steps")
    for dur in (4, 5, 10, 15):
        n, lt, pf = tokens(dur)
        for name, p in (("gated", 17.4e9),):
            lin, att = step_flops(n, p)
            t = (lin + att) / rate
            print(f"{dur:>5}s {n:>8} {lt:>9} | {lin/1e12:>14.0f} TF {att/1e12:>8.0f} TF {t/60:>8.1f} | {t*a.steps/3600:>6.1f} h")
    print("\n(attention is the quadratic term: it overtakes the linear layers near 20k tokens)")

if __name__ == "__main__":
    main()
