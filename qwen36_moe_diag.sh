#!/bin/bash
# Qwen3.6-35B-A3B prefill diagnosis on one Turin pod (32 cores, one NUMA node):
#   Q1  why ZenDNN llama.cpp is barely faster than stock llama.cpp on this MoE model;
#   Q2  why ZenDNN vLLM (zentorch) is so much faster than ZenDNN llama.cpp on long prompts.
# Only the existing builds (sacsharm's stock and ZenDNN llama.cpp, the sacsharm vLLM venv) are used.
# Settings, engine commands and the reasons for every deviation from the sweeps: diag_lib.sh.
# Knobs: WEIGHTS=bf16|q8  PPS="256 8192"  UBS="512 1024 2048 4096"  REPS=3  LOAD_MODE=dio.
#
# What runs (each step is logged in steps.tsv; a failed step does not stop the others):
#   1. clean llama-bench, pp{PPS} x ub{UBS}: base_stock, base_zendnn, base_zendnn_fb0 (set_cfg in diag_lib.sh)
#   2. decode, tg128: stock and ZenDNN
#   3. vllm bench latency per prompt length; the longest also with --max-num-batched-tokens 8192
#      (one chunk); random vs real-text prompts in one engine (vllm_prompt_bench.py)
#   4. GGML_SCHED_DEBUG op placement, base_zendnn and base_zendnn_fb0 at ub512
#   5. opprof (per-ggml-op wall time and placement)
#   6. vLLM torch profiler per prompt length
#   7. ZenDNNL group-matmul log, base_zendnn_fb0 pp4096 ub4096
#   8. summarize.py -> summary.md, tarball
# cpu_mhz.tsv: effective clock of the benchmark cores every 5 s (cpuinfo_avg_freq), tagged with the step.
#
# Run inside the pod as the benchmark user, e.g. from the workstation:
#   kubectl exec turin-xcovoid0021-pod-1 -n zendnn -- runas sohroy bash -c \
#     'cd /proj/aigstaff/sohroy/profiling && setsid nohup ./qwen36_moe_diag.sh > diag.log 2>&1 < /dev/null &'
# Hardware-level profiles (perf, root only): perf_llama.sh
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
WEIGHTS=${WEIGHTS:-bf16}
OUT=${OUT:-$HERE/qwen36_${WEIGHTS}_$(date +%Y%m%d_%H%M)}
mkdir -p "$OUT"
source "$HERE/diag_lib.sh" || exit 1

CONFIGS=(base_stock base_zendnn base_zendnn_fb0)

echo "output: $OUT"
: > "$OUT/steps.tsv"
step environment environment
if ! preflight "${CONFIGS[@]}" > "$OUT/preflight.txt" 2>&1; then
  echo "preflight failed:"; cat "$OUT/preflight.txt"; exit 1
fi
step build_opprof build_opprof
step prompt_text prompt_text

clock_sampler & CLOCK_PID=$!
trap 'kill $CLOCK_PID 2>/dev/null' EXIT

# 1-2: llama.cpp clean timings
for cfg in "${CONFIGS[@]}"; do step "bench_$cfg" bench "$cfg"; done
for cfg in base_stock base_zendnn; do step "tg_$cfg" bench_tg "$cfg"; done

# 3: vLLM clean timings
source "$VLLM_VENV/bin/activate"
for pp in $PPS; do step "vllm_base_pp$pp" vllm_latency "$pp" base; done
step "vllm_mnbt8192_pp$PP_MAX" vllm_latency "$PP_MAX" mnbt8192 --max-num-batched-tokens 8192
step vllm_text vllm_text

# 4: op placement
for cfg in base_zendnn base_zendnn_fb0; do step "placement_${cfg}_ub512" placement "$cfg" 512; done

# 5: llama.cpp per-op profiles
for pp in $PPS; do
  for cfg in base_stock base_zendnn base_zendnn_fb0; do step "opprof_${cfg}_pp${pp}_ub512" opprof "$cfg" "$pp" 512; done
done
for cfg in base_stock base_zendnn_fb0; do step "opprof_${cfg}_pp${PP_MAX}_ub4096" opprof "$cfg" "$PP_MAX" 4096; done
step "opprof_base_zendnn_pp${PP_MAX}_ub512_text" opprof base_zendnn "$PP_MAX" 512 text

# 6: vLLM op profiles
for pp in $PPS; do step "vllm_profile_pp$pp" vllm_profile "$pp"; done

# 7: ZenDNNL kernel log
step zendnnl_log zendnnl_log base_zendnn_fb0 4096 4096

# 8: summary
kill $CLOCK_PID 2>/dev/null
"$VLLM_VENV/bin/python" "$HERE/summarize.py" "$OUT" > /dev/null
tar czf "$OUT.tar.gz" --exclude='*.pt.trace.json*' --exclude='.nfs*' -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo "done: $OUT/summary.md, $OUT.tar.gz"
