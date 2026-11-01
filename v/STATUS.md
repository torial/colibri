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
| `glm` | unquantized path of `c/colibri.c` | `tests/glm_diff`: ids identical, logits within 1e-4 of the C engine (13/16 bit-identical on real-GLM-aligned shapes) |
| `cmd/colibri_v` | -- | `isa`, `info`, `tok`, `olmoe`, `bench` |

Every differential test has mutation controls: a deliberate break of the V side must be caught. The few that were not
caught are equivalent mutants (code the format rules make unreachable), listed in the commit messages.

## Not in V (stays C, reached through `kernels`)

SIMD / OpenMP matmul and attention kernels, the GGUF dequantizers, CUDA / Vulkan / Metal backends.

## The first engine

`olmoe` is OLMoE end to end in V on the pieces above. What it leaves to C: SIMD matmuls (`kernels`). GGUF checkpoints work (bit-identical to the C engine reading the file's rows, `OLM_ILV16=0`; its default repack to 16-row groups sums in another order). What it does not
have yet: sampling beyond greedy / chat / serve (`sample` and `tokenizer` are there to build them on),
the expert prefetch pilot and hot-expert pinning, multi-threaded attention, the GPU chains. Its logits are the C engine's
bits only when V's generated C is compiled with the engine's flags (`-cflags "-O3 -march=..."`): FMA contraction in loops
like `ctx += a * v` changes the last bit otherwise (the first run, without the flags, differed by ~1e-6 on every logit and
agreed on every token).

`glm` is the GLM-5.2 engine's f32 path (MLA, sigmoid router with selection bias, shared expert). It is the first engine checked to tolerance rather than to the bit in every case: where the bits differ the logits agree to ~1e-6 and the tokens are identical; the remaining last-bit differences come from how C's compiler fuses or unrolls the scalar RoPE-key dot loop for short, odd lengths (shown by an experiment: forcing `fmaf` in V flips which shapes match); a rotary width that is a multiple of 8, like real GLM's 64, matches. Quantized GLM (int8 / int4) is the next step toward running the real model.

Next engines, each checked the same way against its C original on a tiny model: the dense-attention families share most of
what `olmoe` has; GLM / DeepSeek need MLA attention and expert streaming from `c/*_chain.h`.

## Known limits

POSIX only (no Windows overlapped handles, no macOS `F_NOCACHE`). `kernels` needs `kernels/build.sh` first and
`v -cc gcc` (tcc cannot link OpenMP); for bit-exact engines add `-cflags "-O3 -march=$ARCH"` with the ARCH the kernels were built for. Only the per-shard map is locked; concurrent readers rely on positional `pread`.
