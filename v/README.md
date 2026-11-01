# colibri in V

A port of colibri to V, one library at a time, each checked against the C original by a differential test
(same inputs, outputs compared bit for bit). Background and measurements: `../docs/v-port-exploration.md`.

| V module | C original | status | check |
|---|---|---|---|
| `cjson` | `c/json.h` | done, behaviour-identical (see the header comment for the one deliberate difference) | `tests/json_diff/run.sh [N] [seed]`: 20,000+ documents, tree dumps identical; `v test cjson` |
| `safetensors` | `c/st.h` (all but the GGUF folders) | tensor table with the C code's header validation; search path of directories (SPLIT); the `model.safetensors.index.json` overlay rule; `colibri.fmt` stamps; read mirrors on other drives; f32 / scale / raw / capped / slice / physical-range / replica-range (O_DIRECT) readers; prefetch and readahead advice; per-tensor and per-shard read-only maps; the missing-tensor report. POSIX only (Windows overlapped handles and macOS `F_NOCACHE` twins are not ported) | `tests/st_diff/run.py [N] [seed]`: 4,000+ model directories (hostile headers included) identical; `tests/st_ext/run.py [N] [seed]`: generated search paths, overlays, stamps, mirrors and ~60 operations per case, exit status, stdout and stderr identical (small and default pread chunk, O_DIRECT on and off); every f16 / bf16 / e8m0 input identical; `v test safetensors` |
| `tokenizer` | `c/tok.h` | BPE load / encode / decode / id_of for the cl100k, o200k, tekken, kimi, gpt2, deepseek (and qwen digit-grouping) pre-tokenizers; Unicode class tables generated from `c/tok_unicode*.h` | `tests/tok_diff/run.py [N] [seed]`: 1,600+ texts over 8 families, ids and decoded bytes identical; every code point's class set identical (`c_uni` / `v_uni`); `v test tokenizer` |
| `sample` | `c/sample.h` | xorshift64* RNG, argmax, softmax + top-p (heap partial sort) distribution, ban-aware draw, `pick_tok`, the stop set (`arm`, with and without a tokenizer), the classified log-prob row reduction, `logprob_target`, `mt_is_glm`. State is a `Sampler` / `StopSet` value, not globals; the `COLI_LOGIT_*` stderr probes are not ported | `tests/sample_diff/run.py [N] [seed]`: 2,000+ cases (hostile NaN/Inf rows, dyadic top-p boundaries, bans, all stop-set switches), picks, distributions, RNG state and log-prob bits identical; `v test sample` |

`v/run_all.sh` runs everything. Next in order: `tok.h` (tokenizers), `sample.h`, `numa_topo.h`, then the rest of `st.h`. Kernels stay C behind `fn C.` bindings.
Run the differential tests with a V compiler and gcc on PATH.
