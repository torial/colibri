#!/bin/sh
# Every check of the V port: unit tests, then the differential tests against the C originals.
# needs gcc, python3 and a V compiler on PATH.   usage: v/run_all.sh [cases scale, default 1]
set -eu
cd "$(dirname "$0")"; k=${1:-1}
v test cjson safetensors gguf tokenizer sample
kernels/build.sh
v -cc gcc test kernels olmoe glm
tests/json_diff/run.sh $((3000 * k)) 1
python3 tests/st_diff/run.py $((400 * k)) 1
python3 tests/st_ext/run.py $((100 * k)) 1
w=$(mktemp -d); gcc -O1 -I../c -o "$w/c_conv" tests/st_diff/c_conv.c -lm; v -o "$w/v_conv" tests/st_diff/v_conv.v >/dev/null
"$w/c_conv" > "$w/c.txt"; "$w/v_conv" > "$w/v.txt"
cmp -s "$w/c.txt" "$w/v.txt" && echo "conversions: every f16 / bf16 / e8m0 input identical" || { echo "conversions: DIFFERENT"; exit 1; }
python3 tests/tok_diff/run.py $((200 * k)) 1
gcc -O1 -I../c -o "$w/c_uni" tests/tok_diff/c_uni.c -lm; v -o "$w/v_uni" tests/tok_diff/v_uni.v >/dev/null
"$w/c_uni" > "$w/cu.txt"; "$w/v_uni" > "$w/vu.txt"
cmp -s "$w/cu.txt" "$w/vu.txt" && echo "unicode classes: every code point identical" || { echo "unicode classes: DIFFERENT"; exit 1; }
python3 tests/sample_diff/run.py $((300 * k)) 1
python3 tests/kern_diff/run.py $((40 * k)) 1
python3 tests/olmoe_diff/run.py $((12 * k)) 1
python3 tests/glm_diff/run.py $((10 * k)) 1
rm -rf "$w"
