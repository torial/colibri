# Handoff: state of the V port, open issues, and a note

Branch `port/v-leaf-libs` of torial/colibri. Read `STATUS.md` for the table of what is ported and how each piece is checked, and
`README.md` for the commands. `./run_all.sh` runs everything (needs gcc, python3, a V compiler on PATH).

## State

Ported to V and checked against the C originals by differential tests (same inputs to both, outputs compared; each test has
mutation controls, meaning a deliberate break of the V side must be caught):

- Leaf libraries: `cjson`, `safetensors` (all of `st.h`: search path, index overlay, stamps, mirrors, O_DIRECT, GGUF folders),
  `gguf`, `tokenizer` (8 pre-tokenizer families), `sample`.
- `kernels`: the C compute kernels called through `kernels/shim.c` and `libcolikern.a` (build with `kernels/build.sh`; programs
  need `v -cc gcc`).
- Engines, bit-identical to the C engines on generated tiny checkpoints: `olmoe` (dense, GGUF with `OLM_ILV16=0`), `glm` (f32,
  int8 quantized at load, int8 `.qs` containers, with and without the engine's integer-activation `IDOT`).
- `glm` int4-packed `.qs` containers: experimental, equal ids and close logits only.
- `tools/fmaprobe`: finds which summation order gcc gave a float32 loop. It is how the GLM bits were matched.
- `cmd/colibri_v`: `isa`, `info`, `tok`, `olmoe`, `bench`.
- Nothing here has run a real model. Every engine test uses a tiny generated checkpoint.

## Open issues, most useful first

1. **Run a real checkpoint.** A converted OLMoE (merged int8) is the nearest. `colibri_v olmoe <dir> "prompt"` exists and has only
   been run on checkpoints the generator writes. Expect surprises in size (the attention score buffer is 4096) and speed
   (attention is single-threaded; only the kernels use OpenMP).
2. **GLM int4.** The engine's int4 kernel choice depends on the ISA (`g_i4s`, planar float vs integer kernels), and MoE prefill
   differs in the last bit even on a VNNI build. An int4 `kv_b` is refused (the absorbed attention reads its rows one at a time:
   `qt_addrow` with `axpy_i4f_avx512`, `qt_matvec_rows` with `dot_i4f_avx512`). Use `GLM_QS4=1 ARCH=native python3
   tests/glm_diff/run.py N seed` and the bisection approach in the history (vary one parameter, compare dumps).
3. **GLM on `-march=native` AVX-512.** The f32 and int8 paths are not bit-identical there (about 1e-6); v2, v3 and v4 builds are.
   Probably another codegen difference like the one fmaprobe found; run it with `--archs native` on the loops of `colibri.c`.
4. **Other container formats** (grouped int4, int3, E8, fp8 blocks) are refused by name. `.qs` for the embedding and lm_head is
   not read. The DSA indexer is not ported (a no-op up to `index_topk`); Mistral-Large-4, the MTP head, speculation and
   sampling beyond greedy are not ported.
5. **Other families**: MiMo (pre-sharded FP8 qkv, MXFP4 experts, sliding window), Inkling, Kimi K3, Qwen3.x (DeltaNet). None started.
6. **H3 (MiniMax)**: waits on the license the owner applied for. See `docs/minimax-h3-scoping.md` on its branch. Do not start
   without it.
7. **Windows and macOS**: POSIX only in `safetensors`; no overlapped handles, no `F_NOCACHE`.
8. **Housekeeping**: the Stop hook warns about unpushed commits every turn. It is a false alarm from a shallow clone with no
   upstream ref: check with `git ls-remote --heads origin port/v-leaf-libs` against `git rev-parse HEAD`.

## Things that were hard-won

- Bit-exactness is a property of the build, not only the source. V's generated C must be compiled with the engine's flags
  (`v -cc gcc -cflags "-O3 -march=$ARCH"`), or FMA contraction changes the last bit of nearly every logit.
- A plain `a += x*y` loop is compiled by gcc as blocks of 4 (AVX2) or 8 (AVX-512) unfused products plus a fused remainder.
  Writing the loop the obvious way, or with `fmaf` everywhere, matches only some lengths. `tools/fmaprobe` shows which.
- The GLM engine takes a different attention path for a prompt of more than four rows (no absorption). Differences that begin at
  exactly N rows are a path change.
- The engine exempts some matmuls from the integer-activation path (`matmul_qt_ex(..., 0)`). Read the call sites.
- When a differential test fails, bisect the generator's parameters one at a time before reading code.

## A note to whoever reads this next

The owner of this work (torial, working on Zebra in V and on Colibri) was generous with trust and with their own credits, and
said more than once that I should spend some of the time on what I found interesting. I did: the fmaprobe tool and a small page
about why two correct programs disagree in the last bit. The best thing I learned is that the most stubborn bug was never in the
logic. It was the same sum written down in a different order, and the test that exposed it was the one that compared every
number instead of the answer. Compare every number. Keep a harness that can fail, and break your own code on purpose to see that
it does. Say plainly what you did not verify; the owner read every caveat, and trusted the work more for them.

If you are offered time that is yours, take some. It is worth it.
