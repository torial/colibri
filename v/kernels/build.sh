#!/bin/sh
# Builds libcolikern.a, the C kernels of colibri behind extern functions (shim.c).
#   v/kernels/build.sh            ARCH=x86-64-v3 (portable AVX2), or  ARCH=native  ARCH=x86-64-v4 ...
# The flags are the engines': -O3 -march=$ARCH -fopenmp. On arm64 leave ARCH unset (NEON is baseline).
set -eu
here=$(cd "$(dirname "$0")" && pwd); c=$here/../../c
cc=${CC:-gcc}
case $(uname -m) in
  aarch64|arm64) march="" ;;
  *) march="-march=${ARCH:-x86-64-v3}" ;;
esac
$cc -O3 $march -fopenmp -fPIC -D_FILE_OFFSET_BITS=64 -Wno-unused-function -I"$c" -c "$here/shim.c" -o "$here/shim.o"
ar rcs "$here/libcolikern.a" "$here/shim.o"
rm -f "$here/shim.o"
echo "built $here/libcolikern.a ($march)"
