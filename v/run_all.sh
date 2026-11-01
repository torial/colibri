#!/bin/sh
# Every check of the V port: unit tests, then the differential tests against the C originals.
# needs gcc, python3 and a V compiler on PATH.   usage: v/run_all.sh [cases scale, default 1]
set -eu
cd "$(dirname "$0")"; k=${1:-1}
v test cjson safetensors
tests/json_diff/run.sh $((3000 * k)) 1
python3 tests/st_diff/run.py $((400 * k)) 1
w=$(mktemp -d); gcc -O1 -I../c -o "$w/c_conv" tests/st_diff/c_conv.c -lm; v -o "$w/v_conv" tests/st_diff/v_conv.v >/dev/null
"$w/c_conv" > "$w/c.txt"; "$w/v_conv" > "$w/v.txt"
cmp -s "$w/c.txt" "$w/v.txt" && echo "conversions: every f16 / bf16 / e8m0 input identical" || { echo "conversions: DIFFERENT"; exit 1; }
rm -rf "$w"
