#!/bin/bash
# moe_algo_sweep.sh: env-var-only sweep of ZenDNNL's grouped-matmul scheduler for the MoE MUL_MAT_ID
# (latest llama.cpp + latest ZenDNN, 32-expert limit removed, GGML_ZENDNN_ADAPTIVE_FALLBACK=0 so the
# thin-rows rule is bypassed). pp8192, ub 512 and 4096, one process (model load) per setting.
# Run inside the pod: runas sohroy bash moe_algo_sweep.sh
set -u
export HERE=/proj/aigstaff/sohroy/profiling
export OUT=${OUT:-$HERE/moe_algo_sweep_$(date +%Y%m%d_%H%M)}
mkdir -p "$OUT"
source "$HERE/diag_lib.sh" || exit 1
SETTINGS=${SETTINGS:-"default:ZENDNNL_NONE=1 algo3:ZENDNNL_GRP_MATMUL_AUTO_PROMPT_ALGO=3 algo5:ZENDNNL_GRP_MATMUL_AUTO_PROMPT_ALGO=5 algo6:ZENDNNL_GRP_MATMUL_AUTO_PROMPT_ALGO=6 algo2:ZENDNNL_GRP_MATMUL_AUTO_PROMPT_ALGO=2"}
for s in $SETTINGS; do
  name=${s%%:*}; kv=${s#*:}
  set_cfg latest_zendnn_fb0 || exit 1
  echo "[$(date +%T)] $name ($kv)"
  "${CLEAN[@]}" "${E[@]}" "$kv" "${NUMA[@]}" "$BIN_DIR/llama-bench" -m "$GGUF" "${LLAMA_ARGS[@]}" \
    -p 8192 -n 0 -ub 512,4096 -r 3 -o json > "$OUT/$name.json" 2> "$OUT/$name.err"
  echo "  rc=$?"
done
echo done
