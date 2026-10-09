#!/bin/bash
# Clean-provenance rebuild of both fixed builds from the committed HEAD of branch zendnn-concat-fix.
# Refuses to build from a dirty tree. Run in the pod: runas sohroy bash build_concat_all.sh
set -eu
SRC=/proj/aigstaff/sohroy/llama.cpp
ZENDNN_ROOT=/proj/aigstaff/sohroy/ZenDNN-latest/build/install
CMAKE=/proj/aigstaff/sohroy/tools/cmake-3.31.6-linux-x86_64/bin/cmake
cd "$SRC"
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "DIRTY TREE"; git status --short; exit 1; }
echo "branch: $(git branch --show-current)"; git log -3 --format='%H %s'
for v in "build_zendnn_concat -DGGML_ZENDNN=ON -DZENDNN_ROOT=$ZENDNN_ROOT" "build_concat"; do
  set -- $v; d=$1; shift
  echo "=== $d"
  "$CMAKE" -B "$d" -DCMAKE_BUILD_TYPE=Release "$@" 2>&1 | grep -i -E "commit|error"
  "$CMAKE" --build "$d" -j "$(nproc)" --target llama-bench test-backend-ops 2>&1 | grep -E "error|Linking CXX executable"
  ls -l "$d/bin/llama-bench"
done
ldd build_zendnn_concat/bin/llama-bench | grep -i zendnn
