# MiniMax-H3 on colibri: scoping (nothing implemented, no weights touched)

Status: **scoping only.** Written from the public model card, `config.json` files and the LICENSE of
[MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3). No weights were downloaded and no
H3 code runs. The maintainer's licence for use in the United States is **pending**; nobody should run
H3 on a US machine until it is granted (see "Licence").

## What it is

An audio-video generator (text / first+last frame / reference images, video, audio in; up to 15 s of
24 fps video at 768p with 32 kHz stereo audio out). Three modules in MiniMax's system; only one is open:

| module | open? | notes |
|---|---|---|
| H3-Context-IR (prompt / context rewriting) | no, hosted API only | the README says it matters for quality; one can write one's own from the Prompting Guidance in `docs/` |
| **H3-Base** (768p) | **yes**, two BF16 checkpoints: `FL2VA` (t2va, first/last frame) and `Ref2VA` (reference inputs) | CFG-distilled: one forward per step |
| H3-Regenerate-2K | no | |

H3-Base components (all from the repo's configs):

| part | shape | size |
|---|---|---|
| H3-Encoder | Qwen3-VL-32B (Apache 2.0), hidden states of layer 50 of 64 | 14 shards, ~65 GB bf16; only the first 50 layers are needed |
| Omni-Transformer | 33.1B dense, single stream, 50 blocks, hidden 5376, 56 heads x 128, FFN 14336, qk-norm, 3-D MM-RoPE, modality-specific AdaLN; ~13B of the 33B are AdaLN branches whose outputs can be cached | 14 shards, 66 GB bf16 |
| Visual VAE | causal, f16 t4, 24 latent channels, patch 1x2x2 -> 32x spatial, 4x temporal per token; a 36-layer ViT decoder (dim 2048) | 3 shards, ~10 GB |
| Audio VAE | per channel, 32 kHz -> 40 Hz tokens, shared L/R | 0.6 GB |
| Scheduler | flow-style, sigma shift 12 (video) and 3 (audio) | |

Attention is **full** in the open release (the sparse-attention implementation is not released), which
is what makes long clips expensive.

## Cost on a CPU (`tools/h3_cost_model.py`)

Tokens: a 4 s, 768p, 16:9 clip is ~25k tokens (24 latent frames x 1032), 10 s is ~62k, 15 s ~93k.
One step is ~2 x (inference parameters) x tokens of linear work plus the quadratic attention
(4 N^2 d x 50 layers). On 28 Xeon Gold 6132 cores (assumed 60% of AVX2 FMA peak, 1.2 TFLOP/s):

| clip | tokens | minutes / step | 30 steps | 8 steps, with an AVX-512 kernel at 1.7x |
|---|---|---|---|---|
| 4 s | 25k | 21 | 10 h | 1.6 h |
| 5 s | 31k | 29 | 14 h | 2.3 h |
| 10 s | 62k | 86 | 43 h | 6.7 h |
| 15 s | 93k | 171 | 85 h | 13 h |

The **number of steps is not in the files** (check the SGLang / vLLM cookbooks); the FFN's gating (which moves the
inference parameter count between ~13.5B and ~17.4B) is also unconfirmed. So: hours for a 4-5 s clip
at best, days for 10-15 s. A CPU port is a way to run H3 *at all* on this machine, not a way to make a series.
The weights fit (int8: ~15-17 GB transformer, bf16 encoder ~55 GB for its 50 layers, VAE ~10 GB), 192 GB is plenty.

## With the Quadro M6000 (12 GB, Maxwell, ~5 TFLOP/s f32)

The card cannot hold the transformer, but a diffusion step reads each layer's weights once, so layers can
be streamed: ~17 GB of int8 weights over PCIe 3 is ~1.5 s a step, small against the compute. A cuBLAS-class f32 GEMM
and a flash-style attention on the card would take a 4 s clip from ~21 to ~5 min a step
(`python3 tools/h3_cost_model.py --gpu-tflops 5`): 8 steps ~40 min, 30 steps ~2.6 h; 10 s is ~21 min a step. The CPU and
the card can also split the work (the CPU on one half of the heads / token blocks). The card has no tensor cores and
no fast f16, so everything is f32 and the activations of a 25k-token clip (~0.5 GB per tensor) must be tiled. The CUDA
backend in `c/` builds for sm_52 on branch `perf/xeon-skylake-sp`. NVIDIA's driver and CUDA need glibc (not Alpine).

Done so far (branch `perf/flash-attn-f32`): `c/qi_flash.h`, a blocked f32 attention with constant memory
(25k tokens of one head in 2.5 s on a 4-core VM, checked against the naive softmax); it replaces the Tq x Tk score matrix
that makes long sequences impossible. A CUDA version of the same tiling for the M6000 is the matching GPU piece.

## Port plan, if it goes ahead

The Qwen-Image engine (`c/qwenimage.c`, `qi_gemm.h`, `qwenimage_encoder.h`, `qwenimage_vae.h`) is
the template: it already has the loader-from-diffusers-checkpoint, int8-at-load quantization, a Qwen3-VL
text encoder, a flow-matching loop and the `coli` / serve front ends. New work, in dependency order:

1. **Reference harness (first, and the thing that makes a blind port safe).** A Python script using the
   official diffusers pipeline that dumps, for one tiny prompt and a short clip: tokenizer ids, encoder layer-50
   states, packed token sequence + MM-RoPE tables, per-block transformer outputs for 1-2 steps, latents,
   decoded frames, decoded audio. Every later component is compared against these dumps (the repo's
   `test_qwenimage_*` tests work this way).
2. **Encoder at 32B.** Generalise `qwenimage_encoder.h` (hidden 5120, 64 layers, stop at layer 50). Apache-2.0 weights,
   so this part is independent of the H3 licence.
3. **Omni-Transformer block**: modality-specific AdaLN (cached modulation tables), qk-norm, 3-D MM-RoPE,
   full attention over ~25k+ tokens (needs a blocked / flash-style kernel; `qi_gemm.h`'s attention does not fit this size).
4. **Visual VAE**: causal conv encoder (only for first/last-frame and reference modes) and the 36-layer ViT decoder. **Audio VAE.**
5. **Packing** of the multimodal sequence (text, image, video, audio tokens, reference handling), the two schedulers.
6. **Output**: frames + audio to mp4 (a muxer; no video code exists in colibri today).
7. Front ends: a family in `coli`, `POST /v1/videos`-style endpoint, tests, docs.

Honest size: steps 1-2 days; 3-4 a couple of weeks each to match the reference to tolerance; 5-7 about a week.
The hard 20% is the VAE decoder and MM-RoPE details, which only the reference dumps can settle.

## Licence (read the real text; this is a summary, not legal advice)

- `LICENSE` ("MiniMax H3 Community License", Hong Kong law) grants rights only within the **Applicable
  Territory = worldwide except the EU, UK, South Korea and the United States.** Use of the works *or Outputs*
  outside the Applicable Territory is not authorised. People in an Excluded Territory may ask MiniMax for a licence
  (application form linked from `docs/QA-about-License.md`: `https://platform.minimax.io/h3-license`).
- Even where granted: commercial products over US$20M/yr need written authorisation; commercial products must display
  "MiniMax H3" in their UI; distribution needs the licence copy, a NOTICE file, and downstream terms at least as protective;
  you may not use H3 or its Outputs to improve another AI model; derivatives made by distillation are Model Derivatives.
- Acceptable Use Policy includes: no use that infringes third-party IP, no exploiting or harming minors, no impersonation,
  **no public dissemination of machine-generated content without clear disclosure that it is machine-generated**,
  no circumventing safeguards, no military use. Anyone offering generation to third parties must keep safeguards and
  a way to report violations.

Practical consequences: keep the colibri integration free of weights and of H3-derived tensors; do not publish H3 weights,
quantised copies or outputs from this repo; label any published output as AI-generated.
