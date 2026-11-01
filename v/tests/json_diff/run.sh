#!/bin/sh
# Differential test: c/json.h against the V port on the same corpus, trees compared bit for bit.
#   tests/json_diff/run.sh [N cases] [seed]     needs gcc, python3 and a V compiler on PATH
set -eu
here=$(cd "$(dirname "$0")" && pwd); root=$here/../../..; w=${TMPDIR:-/tmp}/json_diff.$$; mkdir -p "$w"
python3 "$here/gen_corpus.py" "$w/corpus" "${1:-3000}" "${2:-1}"
gcc -O1 -I"$root/c" -o "$w/c_dump" "$here/c_dump.c" -lm
(cd "$here/../.." && v -o "$w/v_dump" tests/json_diff/v_dump.v) >/dev/null
"$w/c_dump" "$w"/corpus/*.json > "$w/c.out"
"$w/v_dump" "$w"/corpus/*.json > "$w/v.out"
if cmp -s "$w/c.out" "$w/v.out"; then echo "json differential: $(grep -c '^#' "$w/c.out") documents, C and V identical"; rm -rf "$w"
else echo "json differential: DIFFERENT (kept in $w)"; diff "$w/c.out" "$w/v.out" | head -20; exit 1; fi
