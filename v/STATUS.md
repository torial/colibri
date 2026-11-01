# V port: status

Branch `port/v-leaf-libs`. Everything below is verified by `./run_all.sh` (needs `gcc`, `python3`, a V compiler).

## Done

| Piece | From | How it is checked |
|---|---|---|
| `cjson` | `c/json.h` | 20,000+ documents, trees identical |
| `safetensors` | `c/st.h` (all of it, GGUF folders included) | `tests/st_diff` (4,000+ directories) and `tests/st_ext` (search paths, overlays, stamps, mirrors, GGUF folders, ~60 operations per case; exit status, stdout and stderr identical) |
| `gguf` | `c/gguf.h` | inside `tests/st_ext` |
| `tokenizer` | `c/tok.h` | `tests/tok_diff`: 8 pre-tokenizer families, every code point's class |
| `sample` | `c/sample.h` | `tests/sample_diff`: picks, distributions, RNG state, log-prob bits |
| `kernels` | `c/quant.h`, `ggml_quants.h`, `matmul_f32.h`, `qi_gemm.h` | `tests/kern_diff`: output bits identical to the direct C call; builds for v2 / v3 / v4 / native |
| `cmd/colibri_v` | -- | `isa`, `info`, `tok`, `bench` |

Every differential test has mutation controls: a deliberate break of the V side must be caught. The few that were not
caught are equivalent mutants (code the format rules make unreachable), listed in the commit messages.

## Not in V (stays C, reached through `kernels`)

SIMD / OpenMP matmul and attention kernels, the GGUF dequantizers, CUDA / Vulkan / Metal backends.

## What an engine written in V still needs

The pieces an inference engine is made of that exist here: weights in (safetensors, GGUF, mirrors, overlays), tokens in
and out (tokenizer), the next token (sample), the matmuls (kernels). What is left is the part that differs per model
family, in `c/*_chain.h`, `colibri.c`, `glm53.c`, ...: config parsing, the layer loop (norms, rotary, attention with a KV
cache, MoE routing, expert streaming), and the CLI / server. That is a port per family, each checked the same way
against its C original on a tiny model. The least risky first family is the one with the smallest chain (OLMoE).

## Known limits

POSIX only (no Windows overlapped handles, no macOS `F_NOCACHE`). `kernels` needs `kernels/build.sh` first and
`v -cc gcc` (tcc cannot link OpenMP). Only the per-shard map is locked; concurrent readers rely on positional `pread`.
