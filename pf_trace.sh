#!/bin/bash
# pf_trace.sh (root in pod): where do the steady-state page faults of ZenDNN llama.cpp come from?
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
HERE=/proj/aigstaff/sohroy/profiling
PERF=/proj/aigstaff/sohroy/tools/perf-7.0.0-30/perf
U=$(stat -c %u "$HERE"); G=$(stat -c %g "$HERE")
AS_USER=(setpriv --reuid "$U" --regid "$G" --clear-groups env HOME=/proj/rdi/staff/sohroy)
export CPUS=$(awk '/^Cpus_allowed_list/{print $2}' /proc/1/status) MEMS=$(awk '/^Mems_allowed_list/{print $2}' /proc/1/status) WEIGHTS=bf16
OUT=${OUT:-$HERE/investigate_20261007/pf}; "${AS_USER[@]}" mkdir -p "$OUT"
HERE=$HERE source "$SCRIPT_DIR/diag_lib.sh" || exit 1
CFG=${CFG:-latest_zendnn}; set_cfg "$CFG" || exit 1
EXTRA=(${EXTRA_ENV:-ZENDNNL_NONE=1})
TAG=${TAG:-$CFG}
TMP=$(mktemp -d /tmp/pf.XXXXXX); chmod 755 $TMP
"${AS_USER[@]}" "${CLEAN[@]}" "${E[@]}" "${EXTRA[@]}" "${NUMA[@]}" "$BIN_DIR/llama-bench" -m "$GGUF" "${LLAMA_ARGS[@]}" \
  -p 2048 -n 0 -ub 512 -r 40 --progress -o json > $TMP/bench.json \
  2> >(while IFS= read -r l; do printf '%s %s\n' "$EPOCHREALTIME" "$l"; done > $TMP/bench.err) &
pid=$!
until [ "$(grep -c 'prompt run' $TMP/bench.err 2>/dev/null)" -ge 12 ]; do kill -0 $pid 2>/dev/null || exit 1; sleep 2; done
echo "steady, recording page faults (pid $pid)"
"$PERF" stat -p $pid -e page-faults,minor-faults,major-faults -o $TMP/pf_stat.txt -- sleep 10
"$PERF" record -e page-faults -c 50 --call-graph fp -d -p $pid -o $TMP/pf.data -- sleep 10 > $TMP/rec.log 2>&1
kill $pid; wait $pid 2>/dev/null
"$PERF" report -i $TMP/pf.data --stdio --no-children --sort dso,sym --percent-limit 0.5 > $TMP/pf_sym.txt 2>/dev/null
"$PERF" report -i $TMP/pf.data --stdio --children --sort sym --percent-limit 2 -g none > $TMP/pf_children.txt 2>/dev/null
"$PERF" report -i $TMP/pf.data --stdio --no-children --sort sym -g caller,0.5,callee --percent-limit 3 > $TMP/pf_callers.txt 2>/dev/null
"$PERF" script -i $TMP/pf.data -F tid,ip,sym,dso,addr 2>/dev/null | head -200000 | gzip > $TMP/pf_script.txt.gz
rm -f $TMP/pf.data
chmod -R a+rX $TMP; "${AS_USER[@]}" sh -c "cp -r $TMP/. $OUT/$TAG/ 2>/dev/null || (mkdir -p $OUT/$TAG && cp -r $TMP/. $OUT/$TAG/)"
echo done
