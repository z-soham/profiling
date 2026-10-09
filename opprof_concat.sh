#!/bin/bash
# Per-op profile before/after the CONCAT fix (ZenDNN builds). Run in pod: runas sohroy bash opprof_concat.sh
set -u
export HERE=/proj/aigstaff/sohroy/profiling
ROOT=${ROOT:-$HERE/opprof_concat_run}; mkdir -p "$ROOT" "$HERE/bin"
source "$HERE/diag_lib.sh" || exit 1
L=/proj/aigstaff/sohroy/llama.cpp
g++ -O2 -std=c++17 -I$L/include -I$L/ggml/include "$HERE/opprof.cpp" -L$L/build_zendnn_concat/bin -lllama -lggml -lggml-base -o "$OPPROF" || exit 1
for v in base:"" fixed:_concat; do
  export LATEST_SUFFIX=${v#*:}
  export OUT=$ROOT/${v%%:*}; mkdir -p "$OUT"
  for ub in 512 4096; do opprof latest_zendnn 4096 $ub; done
done
echo done > "$ROOT/DONE"
