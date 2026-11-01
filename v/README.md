# colibri in V

A port of colibri to V, one library at a time, each checked against the C original by a differential test
(same inputs, outputs compared bit for bit). Background and measurements: `../docs/v-port-exploration.md`.

| V module | C original | status | check |
|---|---|---|---|
| `cjson` | `c/json.h` | done, behaviour-identical (see the header comment for the one deliberate difference) | `tests/json_diff/run.sh [N] [seed]`: 20,000+ documents, tree dumps identical; `v test cjson` |
| `safetensors` | the core of `c/st.h` | tensor table with the C code's header validation, raw / f32 readers, bf16 / f16 / e8m0 conversions. Not yet: mirrors, multi-drive, O_DIRECT, mmap, `colibri.fmt` stamps, index overlay, GGUF | `tests/st_diff/run.py [N] [seed]`: 4,000+ model directories (hostile headers included), tensor tables and data hashes identical; every f16 / bf16 / e8m0 input identical; `v test safetensors` |
| `tokenizer` | `c/tok.h` | BPE load / encode / decode / id_of for the cl100k, o200k, tekken, kimi, gpt2, deepseek (and qwen digit-grouping) pre-tokenizers; Unicode class tables generated from `c/tok_unicode*.h` | `tests/tok_diff/run.py [N] [seed]`: 1,600+ texts over 8 families, ids and decoded bytes identical; every code point's class set identical (`c_uni` / `v_uni`); `v test tokenizer` |
| `sample` | `c/sample.h` | xorshift64* RNG, argmax, softmax + top-p (heap partial sort) distribution, ban-aware draw, `pick_tok`, the stop set (`arm`, with and without a tokenizer), the classified log-prob row reduction, `logprob_target`, `mt_is_glm`. State is a `Sampler` / `StopSet` value, not globals; the `COLI_LOGIT_*` stderr probes are not ported | `tests/sample_diff/run.py [N] [seed]`: 2,000+ cases (hostile NaN/Inf rows, dyadic top-p boundaries, bans, all stop-set switches), picks, distributions, RNG state and log-prob bits identical; `v test sample` |

`v/run_all.sh` runs everything. Next in order: `tok.h` (tokenizers), `sample.h`, `numa_topo.h`, then the rest of `st.h`. Kernels stay C behind `fn C.` bindings.
Run the differential tests with a V compiler and gcc on PATH.
