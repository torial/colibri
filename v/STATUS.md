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
| `olmoe` | CPU path of `c/olmoe.c` | `tests/olmoe_diff`: the C engine and the V engine on generated tiny checkpoints, generated ids and every logit of every step bit-identical (v2 / v3 / v4 builds) |
| `cmd/colibri_v` | -- | `isa`, `info`, `tok`, `olmoe`, `bench` |

Every differential test has mutation controls: a deliberate break of the V side must be caught. The few that were not
caught are equivalent mutants (code the format rules make unreachable), listed in the commit messages.

## Not in V (stays C, reached through `kernels`)

SIMD / OpenMP matmul and attention kernels, the GGUF dequantizers, CUDA / Vulkan / Metal backends.

## The first engine

`olmoe` is OLMoE end to end in V on the pieces above. What it leaves to C: SIMD matmuls (`kernels`). What it does not
have yet: GGUF checkpoints, sampling beyond greedy / chat / serve (`sample` and `tokenizer` are there to build them on),
the expert prefetch pilot and hot-expert pinning, multi-threaded attention, the GPU chains. Its logits are the C engine's
bits only when V's generated C is compiled with the engine's flags (`-cflags "-O3 -march=..."`): FMA contraction in loops
like `ctx += a * v` changes the last bit otherwise (the first run, without the flags, differed by ~1e-6 on every logit and
agreed on every token).

Next engines, each checked the same way against its C original on a tiny model: the dense-attention families share most of
what `olmoe` has; GLM / DeepSeek need MLA attention and expert streaming from `c/*_chain.h`.

## Known limits

POSIX only (no Windows overlapped handles, no macOS `F_NOCACHE`). `kernels` needs `kernels/build.sh` first and
`v -cc gcc` (tcc cannot link OpenMP); for bit-exact engines add `-cflags "-O3 -march=$ARCH"` with the ARCH the kernels were built for. Only the per-shard map is locked; concurrent readers rely on positional `pread`.
