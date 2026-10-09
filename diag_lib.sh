# Settings and step functions shared by qwen36_moe_diag.sh (runs as the benchmark user) and
# perf_llama.sh (runs as root). Source it with OUT set; it defines no side effects beyond
# variables and functions.
#
# Engine commands: the optimised per-configuration commands for Qwen3.6-35B-A3B BF16 in
# ../Benchmarking/turin-results-2026-09-30.zip, docs/turin-optimised-commands-2026-09-30.md (boost off):
#   llama.cpp  setup B, "new-command" (both Qwen BF16 llama.cpp configurations):
#              LD_PRELOAD=tcmalloc:/proj/rdi/staff/sohroy/lib/libomp.so.5  KMP_AFFINITY=granularity=fine,compact,1,0
#              KMP_BLOCKTIME=1 KMP_TPAUSE=0 KMP_{FORKJOIN,PLAIN,REDUCTION}_BARRIER_PATTERN=dist,dist
#              OMP_NUM_THREADS=32 OMP_DYNAMIC=FALSE OMP_WAIT_POLICY=ACTIVE  (ZenDNN build: ZENDNNL_MATMUL_ALGO=1)
#              numactl --physcpubind=<32 cores> --membind=<node> ... -t 32 -tb 32 -b 4096 -ub 512 -fa on -ctk f16 -ctv f16
#   vLLM       setup C, "vLLM compile" (Qwen BF16):
#              LD_PRELOAD=tcmalloc:libiomp5(venv)  OMP_NUM_THREADS=32  VLLM_CPU_OMP_THREADS_BIND=<all 32 cores>
#              VLLM_CPU_KVCACHE_SPACE=32 (Qwen BF16; 90 otherwise)  ZENDNNL_MATMUL_WEIGHT_CACHE=1
#              ZENDNNL_MATMUL_ALGO / ZENDNNL_LRU_CACHE_CAPACITY unset; no --enforce-eager (torch.compile)
#              --dtype auto --kv-cache-dtype auto --distributed-executor-backend mp --language-model-only
#              --max-num-seqs 1 --max-num-batched-tokens 4096 --max-model-len 16384
# Differences, forced by the offline tools or by measurement hygiene:
#   - llama-bench (and opprof) instead of llama-server: no -c 32000, the context is sized to the prompt.
#   - llama.cpp loads weights with --load-mode $LOAD_MODE (default dio: O_DIRECT into this process's
#     memory, bound to MEMS by numactl) instead of mlock (which falls back to mmap under the pod's
#     8 MB memlock limit). An mmap'd GGUF runs from the host-wide page cache, which stays on whatever
#     NUMA node first read it (node 5 left by the pod-5 sweeps made this pod's runs 1.8x slower), and
#     reading it here would in turn pull the shared cache onto this pod's node under the pod-5
#     sweeps. dio touches no page cache at all.
#   - vllm bench latency: batch 1, --output-len 1, prefix caching off (it reuses one prompt).

HERE=${HERE:-/proj/aigstaff/sohroy/profiling}
WEIGHTS=${WEIGHTS:-bf16}
PPS=${PPS:-"256 8192"}
UBS=${UBS:-"512 1024 2048 4096"}
REPS=${REPS:-3}
LOAD_MODE=${LOAD_MODE:-dio}
PROMPT_TEXT=${PROMPT_TEXT:-$HERE/prompt_text.txt}

# ---------------------------------------------------------------- placement: this pod's cpuset
CPUS=${CPUS:-$(awk '/^Cpus_allowed_list/{print $2}' /proc/self/status)}
MEMS=${MEMS:-$(awk '/^Mems_allowed_list/{print $2}' /proc/self/status)}
FIRST=${CPUS%-*}; LAST=${CPUS#*-}
NT=$((LAST - FIRST + 1))
NUMA=(numactl --physcpubind="$CPUS" --membind="$MEMS" --)

# ---------------------------------------------------------------- installations and weights
BASE_DIR=/proj/aigstaff/sohroy/llama.cpp-sacsharm      # copies of sacsharm's 8fe90e1fb builds (the sweeps' binaries)
LATEST_DIR=/proj/aigstaff/sohroy/llama.cpp
LATEST_ZEN_LIB=/proj/aigstaff/sohroy/ZenDNN-latest/build/install/zendnnl/lib
ZEN_LIB=$BASE_DIR/zendnnl/lib                           # libzendnnl.so of the ZenDNN build (md5 73cb16d9)
LLAMA_SRC=/proj/rdi/staff/sacsharm/llama.cpp            # headers for opprof (8fe90e1fb)
TCMALLOC=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4
LIBOMP=/proj/rdi/staff/sohroy/lib/libomp.so.5          # LLVM libomp; takes over libgomp's GOMP_* so KMP_* apply
VLLM_VENV=/proj/rdi/staff/sacsharm/vllm/.venv
case $WEIGHTS in
  bf16) GGUF=/proj/rdi/staff/sohroy/models/Qwen3.6-35B-A3B-BF16.gguf
        HF=/proj/rdi/staff/sacsharm/models/hf/Qwen3.6-35B-A3B
        KV_SPACE=32 ;;
  q8)   GGUF=/proj/rdi/staff/sohroy/models/Qwen3.6-35B-A3B-Q8_0.gguf
        HF=/proj/rdi/staff/sohroy/models/Qwen3.6-35B-A3B-w8a8-llmcompressor
        KV_SPACE=90 ;;
  *)    echo "WEIGHTS must be bf16 or q8" >&2; return 1 2>/dev/null || exit 1 ;;
esac
OPPROF=$HERE/bin/opprof

LLAMA_ENV=(LD_PRELOAD="$TCMALLOC:$LIBOMP" KMP_AFFINITY=granularity=fine,compact,1,0
           KMP_BLOCKTIME=1 KMP_TPAUSE=0 KMP_FORKJOIN_BARRIER_PATTERN=dist,dist
           KMP_PLAIN_BARRIER_PATTERN=dist,dist KMP_REDUCTION_BARRIER_PATTERN=dist,dist
           OMP_NUM_THREADS="$NT" OMP_DYNAMIC=FALSE OMP_WAIT_POLICY=ACTIVE)
LLAMA_ARGS=(-t "$NT" -b 4096 -fa on -ctk f16 -ctv f16 -lm "$LOAD_MODE")   # llama-bench -t sets -t and -tb

VLLM_ENV=(LD_PRELOAD="$TCMALLOC:$VLLM_VENV/lib/libiomp5.so" OMP_NUM_THREADS="$NT"
          VLLM_CPU_OMP_THREADS_BIND="$CPUS" VLLM_CPU_KVCACHE_SPACE="$KV_SPACE" ZENDNNL_MATMUL_WEIGHT_CACHE=1)
VLLM_ENGINE=(--model "$HF" --dtype auto --kv-cache-dtype auto --distributed-executor-backend mp
             --language-model-only --max-model-len 16384 --max-num-seqs 1 --max-num-batched-tokens 4096
             --no-enable-prefix-caching)
VLLM_ARGS=("${VLLM_ENGINE[@]}" --batch-size 1 --output-len 1)

# no inherited OpenMP pinning (runas/login also export OMP_NUM_THREADS; ours override it)
CLEAN=(env -u OMP_PROC_BIND -u OMP_PLACES -u GOMP_CPU_AFFINITY -u KMP_AFFINITY -u LD_PRELOAD -u LD_LIBRARY_PATH)

# set_cfg <config> -> E (env), BIN_DIR. sacsharm's existing builds, as swept; no code changes.
#   base_stock       stock build
#   base_zendnn      ZenDNN build
#   base_zendnn_fb0  ZenDNN build with GGML_ZENDNN_ADAPTIVE_FALLBACK=0 (a runtime switch of that build:
#                    lifts its "> 32 experts -> CPU" rule, so the MoE experts run on ZenDNN too)
set_cfg() {
  case $1 in
    base_stock)       E=("${LLAMA_ENV[@]}" LD_LIBRARY_PATH="$BASE_DIR/build/bin"); BIN_DIR=$BASE_DIR/build/bin ;;
    base_zendnn)      E=("${LLAMA_ENV[@]}" LD_LIBRARY_PATH="$BASE_DIR/build_zendnn/bin:$ZEN_LIB" ZENDNNL_MATMUL_ALGO=1)
                      BIN_DIR=$BASE_DIR/build_zendnn/bin ;;
    base_zendnn_fb0)  set_cfg base_zendnn; E+=(GGML_ZENDNN_ADAPTIVE_FALLBACK=0) ;;
    # latest llama.cpp + latest ZenDNN, 32-expert limit removed from supports_op (build_llama_latest_zendnn.sh)
    latest_zendnn)    E=("${LLAMA_ENV[@]}" LD_LIBRARY_PATH="$LATEST_DIR/build_zendnn${LATEST_SUFFIX:-}/bin:$LATEST_ZEN_LIB" ZENDNNL_MATMUL_ALGO=1)
                      BIN_DIR=$LATEST_DIR/build_zendnn${LATEST_SUFFIX:-}/bin ;;
    latest_stock)     E=("${LLAMA_ENV[@]}" LD_LIBRARY_PATH="$LATEST_DIR/build${LATEST_SUFFIX:-}/bin"); BIN_DIR=$LATEST_DIR/build${LATEST_SUFFIX:-}/bin ;;
    latest_zendnn_algo3) set_cfg latest_zendnn; E+=(ZENDNNL_GRP_MATMUL_AUTO_PROMPT_ALGO=3) ;;
    latest_zendnn_algo5) set_cfg latest_zendnn; E+=(ZENDNNL_GRP_MATMUL_AUTO_PROMPT_ALGO=5) ;;
    latest_zendnn_fb0) set_cfg latest_zendnn; E+=(GGML_ZENDNN_ADAPTIVE_FALLBACK=0) ;;
    *) echo "unknown config $1" >&2; return 1 ;;
  esac
}

# step <name> <function> [args...] : runs one experiment, records rc and duration, never aborts
step() {
  local name=$1; shift
  echo "[$(date +%F' '%T)] >>> $name" | tee -a "$OUT/driver.log"
  echo "$name" > "$OUT/.step"
  local t0=$SECONDS
  "$@"; local rc=$?
  echo idle > "$OUT/.step"
  printf '%s\t%s\t%s\n' "$name" "$rc" "$((SECONDS - t0))" >> "$OUT/steps.tsv"
  [ $rc -eq 0 ] || echo "    FAILED rc=$rc ($name)" | tee -a "$OUT/driver.log"
  return 0
}

# numa_summary <pid>: resident memory of a process per NUMA node (numa_maps, any page size)
numa_summary() {
  awk '{ ps = 4; for (i = 1; i <= NF; i++) if ($i ~ /^kernelpagesize_kB=/) { split($i, a, "="); ps = a[2] }
         for (i = 1; i <= NF; i++) if ($i ~ /^N[0-9]+=/) { split($i, a, "="); n[a[1]] += a[2] * ps } }
       END { for (k in n) printf "%s=%.1fGB ", k, n[k] / 1048576; print "" }' "/proc/$1/numa_maps" 2>/dev/null
}

# run_probe <tag> <timeout_s> <stdout> <stderr> <cmd...>: run cmd with a timeout; once it holds
# >= 30 GB (the weights are in) write where its memory lives to numa_<tag>.txt
run_probe() {
  local tag=$1 tmo=$2 so=$3 se=$4; shift 4
  timeout "$tmo" "$@" > "$so" 2> "$se" &
  local tpid=$! pid="" rss
  ( for _ in $(seq 1 900); do
      pid=$(pgrep -P "$tpid" | head -1)
      rss=$(awk '/^VmRSS/ { print int($2 / 1048576) }' "/proc/${pid:-0}/status" 2>/dev/null)
      if [ "${rss:-0}" -ge 30 ]; then sleep 5; numa_summary "$pid" > "$OUT/numa_$tag.txt"; exit 0; fi
      kill -0 "$tpid" 2>/dev/null || exit 0
      sleep 2
    done ) &
  local probe=$!
  wait "$tpid"; local rc=$?
  wait "$probe" 2>/dev/null
  return $rc
}

PP_CSV=$(echo $PPS | tr ' ' ',')
UB_CSV=$(echo $UBS | tr ' ' ',')
PP_MIN=$(echo $PPS | tr ' ' '\n' | sort -n | head -1)
PP_MAX=$(echo $PPS | tr ' ' '\n' | sort -n | tail -1)

# ZenDNNL writes its [API ][warning] lines to stdout, into llama-bench's JSON; summarize.py drops them.
bench() {  # bench <cfg>: every prompt length x ubatch in one process (model loaded once)
  set_cfg "$1" || return 1
  run_probe "bench_$1" 7200 "$OUT/bench_$1.json" "$OUT/bench_$1.err" \
    "${CLEAN[@]}" "${E[@]}" "${NUMA[@]}" "$BIN_DIR/llama-bench" -m "$GGUF" "${LLAMA_ARGS[@]}" \
    -p "$PP_CSV" -n 0 -ub "$UB_CSV" -r "$REPS" -o json
}

bench_tg() {  # bench_tg <cfg>: decode, tg128 (ubatch is irrelevant for single-token steps)
  set_cfg "$1" || return 1
  run_probe "tg_$1" 1800 "$OUT/tg_$1.json" "$OUT/tg_$1.err" \
    "${CLEAN[@]}" "${E[@]}" "${NUMA[@]}" "$BIN_DIR/llama-bench" -m "$GGUF" "${LLAMA_ARGS[@]}" \
    -p 0 -n 128 -ub 512 -r "$REPS" -o json
}

vllm_latency() {  # vllm_latency <pp> <tag> [extra engine args...]
  local pp=$1 tag=$2; shift 2
  timeout 3600 "${CLEAN[@]}" "${VLLM_ENV[@]}" "${NUMA[@]}" "$VLLM_VENV/bin/vllm" bench latency "${VLLM_ARGS[@]}" "$@" \
    --input-len "$pp" --num-iters "$REPS" --num-iters-warmup 2 --output-json "$OUT/vllm_${tag}_pp$pp.json" \
    > "$OUT/vllm_${tag}_pp$pp.log" 2>&1
}

vllm_text() {  # random vs real-text prompts, every prompt length, one engine (vllm_prompt_bench.py)
  timeout 5400 "${CLEAN[@]}" "${VLLM_ENV[@]}" "${NUMA[@]}" "$VLLM_VENV/bin/python" "$HERE/vllm_prompt_bench.py" \
    --prompt-file "$PROMPT_TEXT" --lens "$PP_CSV" --iters "$REPS" --output-json "$OUT/vllm_text.json" \
    "${VLLM_ENGINE[@]}" > "$OUT/vllm_text.log" 2>&1
}

vllm_profile() {  # torch profiler around one generate; table -> profiler_out_0.txt, trace -> *.pt.trace.json.gz
  local dir="$OUT/vllm_profile_pp$1"; mkdir -p "$dir"
  local cfg="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$dir\",\"torch_profiler_with_stack\":false,"
  cfg+="\"torch_profiler_dump_cuda_time_total\":false,\"torch_profiler_record_shapes\":true}"
  timeout 3600 "${CLEAN[@]}" "${VLLM_ENV[@]}" "${NUMA[@]}" "$VLLM_VENV/bin/vllm" bench latency "${VLLM_ARGS[@]}" \
    --input-len "$1" --num-iters-warmup 2 --profile --profiler-config "$cfg" > "$dir/run.log" 2>&1 &&
  ls "$dir"/profiler_out_*.txt > /dev/null
}

placement() {  # placement <cfg> <ub>: backend of each op of the last pp<ub> graph, ZenDNN<->CPU split count
  set_cfg "$1" || return 1
  local tag="$1_ub$2" log="$OUT/sched_$1_ub$2.log"
  timeout 1800 "${CLEAN[@]}" "${E[@]}" GGML_SCHED_DEBUG=2 "${NUMA[@]}" "$BIN_DIR/llama-bench" -m "$GGUF" "${LLAMA_ARGS[@]}" \
    -p "$2" -n 0 -ub "$2" -r 1 --no-warmup -v > "$log" 2>&1 || return 1
  # node line: "node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] ..." -> count (op, backend) of the last graph
  awk '/^## SPLIT #0:/ { delete c; delete s }
       /^## SPLIT #/   { b = $0; sub(/^## SPLIT #[0-9]+: /, "", b); sub(/ #.*/, "", b); s[b]++ }
       /^node #/       { o = $0; sub(/^node #[ 0-9]*\(/, "", o); o = substr(o, 1, index(o, "):") - 1); gsub(/ /, "", o)
                         k = $0; sub(/^[^[]*\[/, "", k); split(k, a, " "); c[sprintf("%-12s %s", o, a[1])]++ }
       END { print "op           backend  count"; for (k in c) printf "%s  %d\n", k, c[k] | "sort"; close("sort")
             printf "\nsplits:"; for (b in s) printf "  %s=%d", b, s[b]; print "" }' "$log" > "$OUT/placement_$tag.txt"
  gzip -f "$log"
  grep -q . "$OUT/placement_$tag.txt"
}

opprof() {  # opprof <cfg> <pp> <ub> [text]: 1 full-prompt warm-up, then 1 profiled pass
  set_cfg "$1" || return 1
  local tag="$1_pp$2_ub$3" extra=()
  [ "${4:-}" = text ] && { tag+="_text"; extra=(-f "$PROMPT_TEXT"); }
  run_probe "opprof_$tag" 3600 "$OUT/opprof_$tag.log" "$OUT/opprof_$tag.err" \
    "${CLEAN[@]}" "${E[@]}" "${NUMA[@]}" "$OPPROF" -m "$GGUF" -p "$2" -ub "$3" -b 4096 -t "$NT" \
    -fa on -lm "$LOAD_MODE" -r 1 -w 1 "${extra[@]}" -o "$OUT/opprof_$tag.tsv"
}

zendnnl_log() {  # zendnnl_log <cfg> <pp> <ub>: which ZenDNNL (group) matmul kernels run, with timings
  set_cfg "$1" || return 1
  local log="$OUT/zendnnl_$1_pp$2_ub$3.log"
  timeout 1800 "${CLEAN[@]}" "${E[@]}" ZENDNNL_ENABLE_PROFILER=1 ZENDNNL_PROFILE_LOG_LEVEL=4 ZENDNNL_API_LOG_LEVEL=4 \
    "${NUMA[@]}" "$BIN_DIR/llama-bench" -m "$GGUF" "${LLAMA_ARGS[@]}" -p "$2" -n 0 -ub "$3" -r 1 \
    > "$log" 2>&1
  local rc=$?
  gzip -f "$log"
  return $rc
}

build_opprof() {
  mkdir -p "$HERE/bin"
  [ "$OPPROF" -nt "$HERE/opprof.cpp" ] && return 0
  g++ -O2 -std=c++17 -I"$LLAMA_SRC/include" -I"$LLAMA_SRC/ggml/include" "$HERE/opprof.cpp" \
    -L"$BASE_DIR/build_zendnn/bin" -lllama -lggml -lggml-base -o "$OPPROF"
}

prompt_text() {  # a fixed natural-language prompt: the GPL-3 and Apache-2.0 texts shipped in the pod (~11k tokens)
  [ -s "$PROMPT_TEXT" ] && return 0
  cat /usr/share/common-licenses/GPL-3 /usr/share/common-licenses/Apache-2.0 > "$PROMPT_TEXT"
}

preflight() {  # every path must exist; ZenDNN builds must see their device, stock builds must not
  local ok=0 f c
  for f in "$GGUF" "$HF/config.json" "$ZEN_LIB/libzendnnl.so" "$TCMALLOC" "$LIBOMP" "$VLLM_VENV/bin/vllm" \
           "$VLLM_VENV/lib/libiomp5.so" "$LLAMA_SRC/include/llama.h" "$HERE/vllm_prompt_bench.py"; do
    [ -e "$f" ] || { echo "missing: $f"; ok=1; }
  done
  for c in "$@"; do
    set_cfg "$c" || { ok=1; continue; }
    [ -x "$BIN_DIR/llama-bench" ] || { echo "missing: $BIN_DIR/llama-bench"; ok=1; continue; }
    if "${CLEAN[@]}" "${E[@]}" "$BIN_DIR/llama-bench" --list-devices 2>&1 | grep -q ZenDNN; then
      [[ $c == *zendnn* ]] || { echo "$c lists a ZenDNN device"; ok=1; }
    else
      [[ $c == *zendnn* ]] && { echo "$c does not list the ZenDNN device"; ok=1; }
    fi
  done
  return $ok
}

environment() {
  {
    date -u; hostname; echo "CPUS=$CPUS MEMS=$MEMS NT=$NT WEIGHTS=$WEIGHTS PPS=$PPS UBS=$UBS REPS=$REPS LOAD_MODE=$LOAD_MODE"
    echo "GGUF=$GGUF"; echo "HF=$HF"
    echo "LLAMA_ENV: ${LLAMA_ENV[*]}"; echo "LLAMA_ARGS: ${LLAMA_ARGS[*]}"
    echo "VLLM_ENV: ${VLLM_ENV[*]}"; echo "VLLM_ARGS: ${VLLM_ARGS[*]}"
    echo ---; /usr/local/bin/pod-info 2>/dev/null; grep -E 'allowed_list' /proc/self/status
    echo ---; lscpu; echo ---; numactl -H; echo ---; numactl --show; echo ---; free -g; ulimit -a
    echo --- builds
    md5sum "$BASE_DIR"/build*/bin/libggml-cpu.so.0.* "$BASE_DIR"/build_zendnn/bin/libggml-zendnn.so.0.* 2>/dev/null
    echo --- vLLM; "$VLLM_VENV/bin/pip" list 2>/dev/null | grep -iE '^(vllm|zentorch|torch|intel-openmp) '
    echo --- model; "$VLLM_VENV/bin/python" -c "
import json; c = json.load(open('$HF/config.json')); t = c.get('text_config', c)
print({k: t.get(k) for k in ('num_experts','num_experts_per_tok','moe_intermediate_size','shared_expert_intermediate_size',
      'hidden_size','num_hidden_layers','full_attention_interval','num_attention_heads','num_key_value_heads','head_dim',
      'linear_num_key_heads','linear_num_value_heads')})"
    ls -l "$GGUF"
  } > "$OUT/environment.txt" 2>&1
}

# mean effective clock (MHz) of the benchmark cores, from cpuinfo_avg_freq (APERF/MPERF-based on this
# kernel; /proc/cpuinfo's "cpu MHz" reads a constant 2102 in the pods). One line per 5 s: epoch, step, MHz.
clock_sampler() {
  local c f
  while :; do
    local sum=0 n=0
    for ((c = FIRST; c <= LAST; c++)); do
      f=/sys/devices/system/cpu/cpu$c/cpufreq/cpuinfo_avg_freq
      [ -r "$f" ] && { sum=$((sum + $(cat "$f"))); n=$((n + 1)); }
    done
    [ $n -gt 0 ] && printf '%s\t%s\t%d\n' "$(date +%s)" "$(cat "$OUT/.step" 2>/dev/null)" $((sum / n / 1000)) >> "$OUT/cpu_mhz.tsv"
    sleep 5
  done
}
