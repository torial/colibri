# colibri in V

A port of colibri to V, one library at a time, each checked against the C original by a differential test
(same inputs, outputs compared bit for bit). Background and measurements: `../docs/v-port-exploration.md`.

| V module | C original | status | check |
|---|---|---|---|
| `cjson` | `c/json.h` | done, behaviour-identical (see the header comment for the one deliberate difference) | `tests/json_diff/run.sh [N] [seed]`: 20,000+ documents, tree dumps identical; `v test cjson` |
| `safetensors` | the core of `c/st.h` | tensor table with the C code's header validation, raw / f32 readers, bf16 / f16 / e8m0 conversions. Not yet: mirrors, multi-drive, O_DIRECT, mmap, `colibri.fmt` stamps, index overlay, GGUF | `tests/st_diff/run.py [N] [seed]`: 4,000+ model directories (hostile headers included), tensor tables and data hashes identical; every f16 / bf16 / e8m0 input identical; `v test safetensors` |

`v/run_all.sh` runs everything. Next in order: `tok.h` (tokenizers), `sample.h`, `numa_topo.h`, then the rest of `st.h`. Kernels stay C behind `fn C.` bindings.
Run the differential tests with a V compiler and gcc on PATH.
