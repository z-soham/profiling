#!/bin/bash
# A/B for the CONCAT fix: pp8192 at ub512,4096. baseline = existing build_zendnn / build (pre-fix), fixed = build_zendnn_concat / build_concat.
# Run in the pod: runas sohroy bash bench_concat.sh
set -u
export HERE=/proj/aigstaff/sohroy/profiling
OUT=${OUT:-$HERE/concat_$(date +%Y%m%d_%H%M)}; mkdir -p "$OUT"
source "$HERE/diag_lib.sh" || exit 1
run() {  # name cfg suffix
  LATEST_SUFFIX=$3 set_cfg "$2" || return 1
  [ -x "$BIN_DIR/llama-bench" ] || { echo "missing $BIN_DIR"; return 1; }
  echo "[$(date +%T)] $1"
  "${CLEAN[@]}" "${E[@]}" "${NUMA[@]}" "$BIN_DIR/llama-bench" -m "$GGUF" "${LLAMA_ARGS[@]}" \
    -p 8192 -n 0 -ub 512,4096 -r 3 -o json > "$OUT/$1.json" 2> "$OUT/$1.err"
}
# VARIANTS="name cfg suffix;name cfg suffix;..." (suffix _concat = fixed build, - = existing pre-fix build)
VARIANTS=${VARIANTS:-"zendnn_base latest_zendnn -;zendnn_fixed latest_zendnn _concat;stock_base latest_stock -;stock_fixed latest_stock _concat"}
IFS=';' read -ra VS <<< "$VARIANTS"
for v in "${VS[@]}"; do
  read -r n c s <<< "$v"; [ "$s" = "-" ] && s=""
  run "$n" "$c" "$s"
done
echo done > "$OUT/DONE"
