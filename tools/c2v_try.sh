#!/bin/sh
# Try the C-to-V translator (c2v, via `v translate`) on one colibri header or source file and say
# how much of the output compiles. Usage: tools/c2v_try.sh c/json.h [extra clang flags in $C2V_FLAGS]
#   needs: a V compiler on PATH, clang, network on the first run (v translate clones c2v).
# c2v translates ONE preprocessor configuration: what `#if` leaves for the flags clang is given.
set -eu
src=$1; base=$(basename "$src"); name=${base%.*}
work=${C2V_WORK:-/tmp/c2v_try}; mkdir -p "$work"; cp "$(dirname "$src")"/*.h "$work"/ 2>/dev/null || true
case $src in *.h) printf '#include "%s"\nint main(void){return 0;}\n' "$base" > "$work/t_$name.c"; unit=t_$name.c;;
             *)   cp "$src" "$work/$base"; unit=$base;; esac
[ -n "${C2V_FLAGS:-}" ] && printf '[project]\nadditional_flags = "-I. %s"\n' "$C2V_FLAGS" > "$work/c2v.toml" || rm -f "$work/c2v.toml"
cd "$work"; rm -f "${unit%.c}.v"
s=$(date +%s); v translate "$unit" >/dev/null 2>&1; e=$(date +%s)
n=$(v -o "${unit%.c}_bin" "${unit%.c}.v" 2>&1 | grep -c 'error:' || true)
echo "$base: $(wc -l < "${unit%.c}.v") V lines, $((e-s)) s to translate, $n compile errors"
