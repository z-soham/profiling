#!/bin/bash
# perf_llama.sh: hardware-level profile of Qwen3.6-35B-A3B prefill with the existing builds and the
# optimised commands of ../Benchmarking/turin-results-2026-09-30.zip (turin-optimised-commands-2026-09-30.md,
# boost off): llama.cpp setup B (sacsharm's ZenDNN and stock builds, no code changes) and vLLM setup C.
# Settings, and the reasons for every deviation from the server commands, are in diag_lib.sh.
#
# Per run the workload runs as the benchmark user; once it is computing (llama.cpp: its OpenMP pool
# exists; vLLM: the engine has finished warm-up) and after a settle time, perf attaches to it (for
# vLLM: to the engine's worker process) for a series of windows, one measurement each:
#   flat      perf record -F 999 (cycles)            symbols, DSOs
#   stacks    perf record -e cpu-clock -F 499 --call-graph fp   leaf function + frame-pointer chain per
#             sample, on the wall-clock timer (spinning threads are sampled too). DWARF unwinding came back
#             empty for ~96% of samples here, with cycles (IBS) and cpu-clock alike.
#   basic     task-clock, cycles, instructions, context switches, migrations, page faults
#   topdown   perf stat -M PipelineL1,PipelineL2      retiring / frontend / backend (cpu vs memory) / bad spec
#   flops     retired FP ops (all, and the multiply-accumulate part; the bf16 and packed-fp32 umasks
#             read 0 on this CPU)
#   mem       where L1D fills come from: local L2, L3 of this CCX, another CCX, DRAM (this / the other socket)
#   l2        L2 requests, L2 misses, L2 hardware prefetches that missed L3 (-> DRAM)
# The pods expose only the core PMU (no amd_l3 / amd_df uncore PMUs), so memory traffic is estimated from
# the core's fill-source and prefetch events. At most five events per window, so nothing but topdown is
# multiplexed (the FP events can only use some of the six counters).
# llama-bench runs with --progress; its stderr is timestamped, so pass times during the windows are known;
# it is stopped (by its PID) after the windows. vllm bench latency is sized to finish on its own shortly
# after the windows and reports its average latency. Then, as the benchmark user, one vLLM torch-profiler
# pass per prompt length (per-op split), and perf_analyze.py writes <out>/analysis.md.
#
# Must run as root in the pod (plain `kubectl exec`, not runas): the host has perf_event_paranoid=4.
# perf is the unpacked copy in /proj/aigstaff/sohroy/tools/perf-7.0.0-30 (nothing installed in the pod).
# Root is squashed on NFS and cannot even read the profiling dir (mode 2770), so this script and
# diag_lib.sh are staged into the pod's /tmp first; data is collected in /tmp/perf_llama.* and the text
# outputs are copied into <out> (and analysed) as the benchmark user:
#   kubectl exec turin-xcovoid0021-pod-1 -n zendnn -- runas sohroy bash -c \
#     'mkdir -p /tmp/perf_llama_bin && cp /proj/aigstaff/sohroy/profiling/{perf_llama.sh,diag_lib.sh} /tmp/perf_llama_bin/'
#   kubectl exec turin-xcovoid0021-pod-1 -n zendnn -- bash -c \
#     'setsid nohup /tmp/perf_llama_bin/perf_llama.sh > /tmp/perf_llama.log 2>&1 < /dev/null &'
# Knobs: WEIGHTS=bf16|q8, W=<seconds per window, 15>, RUNS="llama:cfg:pp:ub vllm:pp ..." (default below),
# TORCH_PROFILE_PPS="8192 256" (empty: skip).
set -u

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)            # the /tmp staging copy
HERE=${HERE:-/proj/aigstaff/sohroy/profiling}         # the NFS profiling dir (outputs, perf_analyze.py)
PERF=${PERF:-/proj/aigstaff/sohroy/tools/perf-7.0.0-30/perf}
[ "$(id -u)" = 0 ] || { echo "perf_llama.sh must run as root" >&2; exit 1; }

# the benchmark user owns $HERE; the pod's pinned cpuset and memory node are PID 1's
U=$(stat -c %u "$HERE"); G=$(stat -c %g "$HERE")
AS_USER=(setpriv --reuid "$U" --regid "$G" --clear-groups env HOME=/proj/rdi/staff/sohroy)
export CPUS=${CPUS:-$(awk '/^Cpus_allowed_list/{print $2}' /proc/1/status)}
export MEMS=${MEMS:-$(awk '/^Mems_allowed_list/{print $2}' /proc/1/status)}
export WEIGHTS=${WEIGHTS:-bf16}
W=${W:-15}
RUNS=${RUNS:-"llama:base_zendnn:8192:512 llama:base_stock:8192:512 llama:base_zendnn:256:512 llama:base_stock:256:512
               vllm:8192 vllm:256"}
TORCH_PROFILE_PPS=${TORCH_PROFILE_PPS-"8192 256"}

OUT=${OUT:-$HERE/perf_qwen36_${WEIGHTS}_$(date +%Y%m%d_%H%M)}
"${AS_USER[@]}" mkdir -p "$OUT" || exit 1
HERE=$HERE source "$SCRIPT_DIR/diag_lib.sh" || exit 1
TMP=$(mktemp -d /tmp/perf_llama.XXXXXX); chmod 755 "$TMP"
echo "perf_llama: OUT=$OUT TMP=$TMP CPUS=$CPUS MEMS=$MEMS WEIGHTS=$WEIGHTS W=$W boost=$(cat /sys/devices/system/cpu/cpufreq/boost)"

# windows <pid> <dir> [cpus]: the perf measurement series against one process (-p <pid>), or, with
# cpus, against every task on those cores (-C <cpus>). vLLM needs the latter: attached by PID, perf
# sampled its worker's main thread but almost none of the OpenMP compute threads. The pod's cores run
# nothing but the workload, so -C sees exactly it (plus the idle loop, which the analysis drops).
windows() {
  local pid=$1 d=$2 w T=(-p "$1")
  [ -n "${3:-}" ] && T=(-C "$3")
  for w in flat stacks basic topdown flops mem l2; do
    kill -0 "$pid" 2>/dev/null || { echo "  workload finished before window $w"; break; }
    printf '%s start %s\n' "$w" "$EPOCHREALTIME" >> "$d/windows.txt"
    case $w in
      flat)    "$PERF" record -F 999 "${T[@]}" -o "$d/flat.data" -- sleep "$W" ;;
      stacks)  "$PERF" record -e cpu-clock -F 499 --call-graph fp "${T[@]}" -o "$d/stacks.data" -- sleep "$W" ;;
      basic)   "$PERF" stat "${T[@]}" -o "$d/stat_basic.txt" \
                 -e task-clock,cycles,instructions,context-switches,cpu-migrations,page-faults -- sleep "$W" ;;
      topdown) "$PERF" stat "${T[@]}" -o "$d/stat_topdown.txt" -M PipelineL1,PipelineL2 -- sleep "$W" ;;
      flops)   "$PERF" stat "${T[@]}" -o "$d/stat_flops.txt" \
                 -e cycles,instructions,fp_ret_sse_avx_ops.all,fp_ret_sse_avx_ops.mac_flops -- sleep "$W" ;;
      mem)     "$PERF" stat "${T[@]}" -o "$d/stat_mem.txt" -e ls_any_fills_from_sys.local_l2,\
ls_any_fills_from_sys.local_ccx,ls_any_fills_from_sys.near_cache,\
ls_any_fills_from_sys.dram_io_near,ls_any_fills_from_sys.dram_io_far -- sleep "$W" ;;
      l2)      "$PERF" stat "${T[@]}" -o "$d/stat_l2.txt" -e cycles,l2_request_g1.all,l2_cache_req_stat.ic_dc_miss_in_l2,\
l2_pf_miss_l2_l3.l2_hwpf,l2_pf_miss_l2_l3.l1_dc_l2_hwpf,l2_pf_miss_l2_hit_l3.l2_hwpf -- sleep "$W" ;;
    esac >> "$d/perf.log" 2>&1
    printf '%s end %s\n' "$w" "$EPOCHREALTIME" >> "$d/windows.txt"
  done
}

# reports <dir>: text outputs from the recorded data, then drop the data
reports() {
  local d=$1
  "$PERF" report -i "$d/flat.data" --stdio --no-children --sort dso,sym --percent-limit 0.05 > "$d/flat_sym.txt" 2>/dev/null
  "$PERF" report -i "$d/flat.data" --stdio --no-children --sort comm,dso > "$d/flat_comm_dso.txt" 2>/dev/null
  "$PERF" report -i "$d/flat.data" --stdio --no-children --sort pid,dso > "$d/flat_tid_dso.txt" 2>/dev/null
  "$PERF" script -i "$d/stacks.data" -F pid,tid,time,ip,sym,dso 2>/dev/null | gzip > "$d/stacks_script.txt.gz"
  rm -f "$d"/*.data
}

# llama_run <cfg> <pp> <ub>
llama_run() {
  local cfg=$1 pp=$2 ub=$3
  local tag="${cfg}_pp${pp}_ub${ub}" d="$TMP/${1}_pp${2}_ub${3}"
  mkdir -p "$d"
  set_cfg "$cfg" || return 1
  # enough passes to outlast all windows: pp8192 ~30 s a pass, pp256 ~1 s
  local reps=10 settle=10
  [ "$pp" -lt 2048 ] && reps=600
  echo "[$(date +%T)] $tag: start (reps=$reps settle=${settle}s)"
  "${AS_USER[@]}" "${CLEAN[@]}" "${E[@]}" "${NUMA[@]}" "$BIN_DIR/llama-bench" -m "$GGUF" "${LLAMA_ARGS[@]}" \
    -p "$pp" -n 0 -ub "$ub" -r "$reps" --progress -o json > "$d/bench.json" \
    2> >(while IFS= read -r l; do printf '%s %s\n' "$EPOCHREALTIME" "$l"; done > "$d/bench.err") &
  local pid=$! t0=$SECONDS th
  # compute has started once the OpenMP pool (>= NT threads) exists
  while :; do
    kill -0 "$pid" 2>/dev/null || { echo "  $tag: llama-bench exited early"; tail -3 "$d/bench.err"; return 1; }
    th=$(awk '/^Threads/{print $2}' "/proc/$pid/status" 2>/dev/null)
    [ "${th:-0}" -ge "$NT" ] && break
    [ $((SECONDS - t0)) -gt 1800 ] && { echo "  $tag: no compute after 30 min"; kill "$pid"; return 1; }
    sleep 1
  done
  local load_s=$((SECONDS - t0))
  sleep "$settle"
  numa_summary "$pid" > "$d/numa.txt"
  windows "$pid" "$d"
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  printf 'engine=llama.cpp cfg=%s pp=%s ub=%s reps=%s load_s=%s settle_s=%s window_s=%s\n' \
    "$cfg" "$pp" "$ub" "$reps" "$load_s" "$settle" "$W" > "$d/meta.txt"
  echo "[$(date +%T)] $tag: reports"
  reports "$d"
}

# vllm_run <pp>: vllm bench latency (setup C), perf attached to the engine's worker process
vllm_run() {
  local pp=$1 tag="vllm_pp$1" d="$TMP/vllm_pp$1"
  mkdir -p "$d"; chown "$U:$G" "$d"      # vllm (as the user) writes its --output-json here
  # iterations to outlast settle + windows (~120 s): pp8192 ~13 s each, pp256 ~0.6 s
  local iters=16 settle=5
  [ "$pp" -lt 2048 ] && iters=300
  echo "[$(date +%T)] $tag: start (iters=$iters settle=${settle}s)"
  "${AS_USER[@]}" "${CLEAN[@]}" "${VLLM_ENV[@]}" "${NUMA[@]}" "$VLLM_VENV/bin/vllm" bench latency "${VLLM_ARGS[@]}" \
    --input-len "$pp" --num-iters "$iters" --num-iters-warmup 3 --output-json "$d/bench.json" \
    > "$d/bench.log" 2>&1 &
  local pid=$! t0=$SECONDS wpid=""
  # measure once the benchmark's timed iterations run ("Bench iterations" progress bar). Not "Warming
  # up": vLLM also prints "Warming up model for the compilation..." during engine start-up, and the
  # first attempts measured that torch.compile phase instead of the benchmark.
  until grep -q "Bench iterations" "$d/bench.log" 2>/dev/null; do
    kill -0 "$pid" 2>/dev/null || { echo "  $tag: vllm exited early"; tail -5 "$d/bench.log"; return 1; }
    [ $((SECONDS - t0)) -gt 3000 ] && { echo "  $tag: not warm after 50 min"; kill "$pid"; return 1; }
    sleep 2
  done
  local load_s=$((SECONDS - t0))
  sleep "$settle"
  # the model (weights, compute) lives in the descendant of our benchmark process with the largest
  # resident memory, the "VLLM::Worker" process; used for the numa_maps reading (the windows are -C)
  local p q best=0 rss desc=("$pid")
  for ((i = 0; i < ${#desc[@]}; i++)); do
    for q in $(ps -o pid= --ppid "${desc[$i]}" 2>/dev/null); do desc+=("$q"); done
  done
  for p in "${desc[@]}"; do
    rss=$(awk '/^VmRSS/{print $2}' "/proc/$p/status" 2>/dev/null)
    [ "${rss:-0}" -gt "$best" ] && { best=$rss; wpid=$p; }
  done
  ps -o pid=,nlwp=,args= -p "$(IFS=,; echo "${desc[*]}")" > "$d/processes.txt" 2>/dev/null
  [ -n "$wpid" ] || { echo "  $tag: worker process not found"; wait "$pid"; return 1; }
  echo "  $tag: bench pid $pid, model process $wpid ($((best / 1048576)) GB resident)"
  numa_summary "$wpid" > "$d/numa.txt"
  windows "$wpid" "$d" "$CPUS"
  wait "$pid" 2>/dev/null
  printf 'engine=vllm cfg=vllm pp=%s iters=%s load_s=%s settle_s=%s window_s=%s scope=cpus:%s\n' \
    "$pp" "$iters" "$load_s" "$settle" "$W" "$CPUS" > "$d/meta.txt"
  echo "[$(date +%T)] $tag: reports"
  reports "$d"
}

for r in $RUNS; do
  IFS=: read -r kind a b c <<< "$r"
  case $kind in
    llama) llama_run "$a" "$b" "$c" ;;
    vllm)  vllm_run "$a" ;;
  esac
done

chmod -R a+rX "$TMP"
"${AS_USER[@]}" sh -c "cp -r '$TMP'/. '$OUT'/"
# vLLM per-op split: one torch-profiler pass per prompt length, as the benchmark user (writes to $OUT)
for pp in $TORCH_PROFILE_PPS; do
  echo "[$(date +%T)] vllm torch profile pp$pp"
  "${AS_USER[@]}" env OUT="$OUT" CPUS="$CPUS" MEMS="$MEMS" WEIGHTS="$WEIGHTS" HERE="$HERE" \
    bash -c "source '$SCRIPT_DIR/diag_lib.sh' && vllm_profile $pp"
done
"${AS_USER[@]}" "$VLLM_VENV/bin/python" "$HERE/perf_analyze.py" "$OUT"
echo "[$(date +%T)] done: $OUT/analysis.md"
