#!/bin/bash
# per-op profile at pp8192 ub4096 (matches the vLLM torch-profiler prompt length), fixed build, clean commit
set -u
export HERE=/proj/aigstaff/sohroy/profiling
ROOT=$HERE/opprof_pp8192; mkdir -p $ROOT
source "$HERE/diag_lib.sh" || exit 1
export LATEST_SUFFIX=_concat OUT=$ROOT
opprof latest_zendnn 8192 4096
echo done > $ROOT/DONE
