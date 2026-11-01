# colibri in V

A port of colibri to V, one library at a time, each checked against the C original by a differential test
(same inputs, outputs compared bit for bit). Background and measurements: `../docs/v-port-exploration.md`.

| V module | C original | status | check |
|---|---|---|---|
| `cjson` | `c/json.h` | done, behaviour-identical (see the header comment for the one deliberate difference) | `tests/json_diff/run.sh [N] [seed]`: 20,000+ documents, tree dumps identical; `v test cjson` |

Next in order: `st.h` (safetensors), `tok.h` (tokenizers), `sample.h`, `numa_topo.h`. Kernels stay C behind `fn C.` bindings.
Run the differential tests with a V compiler and gcc on PATH.
