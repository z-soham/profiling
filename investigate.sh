#!/bin/bash
# investigate.sh: why ZenDNN llama.cpp (latest build, expert limits removed) is not faster than stock.
#   M  memory growth / page faults per ZENDNNL weight-cache setting (RSS, minor faults sampled during passes)
#   U  ubatch 512 vs 4096 at pp8192 for stock / ZenDNN / ZenDNN+ALGO 3 / ALGO 5
#   Z  ZenDNNL API log (which group-matmul kernels run, per-call timing) at ub512 and ub4096
#   O  per-op wall times (opprof) at ub512 and ub4096
# Run inside the pod as sohroy: runas sohroy bash investigate.sh   (PARTS="M U Z O")
set -u
export HERE=/proj/aigstaff/sohroy/profiling
export OUT=${OUT:-$HERE/investigate_$(date +%Y%m%d_%H%M)}
mkdir -p "$OUT"
source "$HERE/diag_lib.sh" || exit 1
PARTS=${PARTS:-"M U Z O"}

# run_sampled <name> <pp> <ub> <reps> <extra env...>: llama-bench with RSS / minor-fault samples every 5 s
run_sampled() {
  local name=$1 pp=$2 ub=$3 reps=$4; shift 4
  set_cfg "$CFG" || return 1
  echo "[$(date +%T)] $name: start"
  "${CLEAN[@]}" "${E[@]}" "$@" "${NUMA[@]}" "$BIN_DIR/llama-bench" -m "$GGUF" "${LLAMA_ARGS[@]}" \
    -p "$pp" -n 0 -ub "$ub" -r "$reps" --progress -o json > "$OUT/$name.json" \
    2> >(while IFS= read -r l; do printf '%s %s\n' "$EPOCHREALTIME" "$l"; done > "$OUT/$name.err") &
  local pid=$! t0=$SECONDS
  echo "t_s,rss_gb,minflt,majflt" > "$OUT/$name.mem.csv"
  while kill -0 "$pid" 2>/dev/null; do
    local rss st
    rss=$(awk '/^VmRSS/{printf "%.2f",$2/1048576}' "/proc/$pid/status" 2>/dev/null)
    st=$(awk '{print $10","$12}' "/proc/$pid/stat" 2>/dev/null)
    [ -n "$rss" ] && echo "$((SECONDS - t0)),$rss,$st" >> "$OUT/$name.mem.csv"
    sleep 5
  done
  wait "$pid" 2>/dev/null
}

if [[ " $PARTS " == *" M "* ]]; then
  CFG=latest_zendnn
  for v in "wc_default:" "wc0:ZENDNNL_MATMUL_WEIGHT_CACHE=0" "wc1:ZENDNNL_MATMUL_WEIGHT_CACHE=1" "wc2:ZENDNNL_MATMUL_WEIGHT_CACHE=2" "prepack0:ZENDNNL_GRP_MATMUL_PREPACK=0"; do
    kv=${v#*:}; [ -n "$kv" ] && x=("$kv") || x=("ZENDNNL_NONE=1")
    run_sampled "mem_${v%%:*}" 2048 512 10 "${x[@]}"
  done
  CFG=latest_stock; run_sampled mem_stock 2048 512 10 ZENDNNL_NONE=1
fi

if [[ " $PARTS " == *" U "* ]]; then
  for c in latest_stock latest_zendnn latest_zendnn_algo3 latest_zendnn_algo5; do
    CFG=$c
    run_sampled "ub_$c" 8192 512,4096 3 ZENDNNL_NONE=1
  done
fi

if [[ " $PARTS " == *" Z "* ]]; then
  for ub in 512 4096; do zendnnl_log latest_zendnn 4096 $ub; done
fi

if [[ " $PARTS " == *" O "* ]]; then
  for ub in 512 4096; do opprof latest_zendnn 4096 $ub; opprof latest_stock 4096 $ub; done
fi
echo "[$(date +%T)] done"
