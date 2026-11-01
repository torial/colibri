# Porting colibri to V: what the C-to-V translator gives, measured

Question: how much of colibri can `v translate` (c2v 0.4.1, a clang-AST based translator that also handles C++)
turn into V, and what would a port look like? Experiments on this repo's `c/` (dev branch), V from the
`-new-compiler` tree, clang 18. `tools/c2v_try.sh FILE` repeats them.

## Results

| input | V lines | translate | compile errors | notes |
|---|---|---|---|---|
| `numa_topo.h` + a main (170 lines of C) | 266 | 1.7 s | **0** | runs and prints the same topology as the C build |
| `idot.h` + `test_idot_extreme.c`, no -march | 615 | 2 s | **0** | runs, prints `ok` — but it translated the **scalar** branch only |
| the same with `-march=skylake-avx512` | 770 | 41 s | 101 | intrinsics become `C._mm512_*` with GCC vector types pasted as raw `C.Attribute__((__vector_size__...` types: not valid V |
| `json.h` | 696 | 3 s | 4 | a `(char[]){...}[0]` compound literal emitted with a stray newline |
| `tok.h` | 5190 | 13 s | 4 | same kind of small defect |
| `st.h` (safetensors) | 3762 | 9 s | 4 | same |

## What this means

1. **Plain C translates well**, to *translated-mode* V (`@[translated]`, `unsafe`, C-style loops, `c2v_pointer_*` helper
   functions): correct, and ugly. It is a runnable starting point, not idiomatic V. Seeing "alternative algorithmic options"
   needs the second step, a hand rewrite, with the translated version as the executable reference.
2. **c2v resolves the preprocessor once.** It sees one configuration (the `-D` / `-march` flags it is given), so the
   portability matrix (AVX2/AVX-512/VNNI/AMX/NEON, CUDA/Metal/Vulkan/HIP, Windows/Linux/BSD) collapses to one path per
   translation. A V port has to re-express those variants (`$if` blocks and C shims), not translate them.
3. **SIMD kernels do not translate.** Intrinsics and vector types stay as C: the kernels (`idot.h`, `expert_ffn.h`,
   `ggml_quants.h`, `qi_gemm.h`, ...) would remain C files behind `fn C.` bindings (V's `#include` / `#flag` make that easy) or be
   rewritten against V's own SIMD story, which is a design decision, not a translation.
4. **Scale:** 238 files / ~157k lines; one engine is 4k-16k lines (`olmoe.c` 4.1k, `qwen38.c` 4.2k, `glm53.c` 6.6k,
   `qwen36.c` 10.7k, `colibri.c` 16k) plus the headers they include. Translation time is seconds to a minute a unit; the cost
   is in the hand fixes (a handful per library so far) and in verifying numerics.

## A workable route

1. **Differential harness first.** Every translated unit is tested against the C original: same inputs, bit-exact outputs for the
   integer kernels, tolerance for floats. colibri already has ~320 C tests; the ones that read stdin/argv can run both builds.
2. **Bottom-up, leaf libraries:** `json.h`, `st.h` (safetensors), `tok.h` (tokenizers), `sample.h`, `numa_topo.h`. Translate, fix
   the few compile defects, run the harness, then refactor to idiomatic V while the tests stay green.
3. **Kernels stay C** behind a thin V binding at first; the V side owns tensors, scheduling, memory placement.
4. **One pilot engine** (OLMoE or the dense Qwen path), end to end, with a real checkpoint, compared token for token with the C
   engine. Only then the larger engines.
5. **Front end last** (`coli` launcher, serve loop): a rewrite in V is natural there, with no numeric risk.

Honest cost: leaf libraries are days; one engine plus kernel bindings is weeks; all 13 engines and the GPU/Metal paths are a
long project, and upstream moves by hundreds of commits a month, so a port should track a pinned upstream commit.
