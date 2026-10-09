#!/bin/bash
# long RSS / minor-fault sampling: does ZenDNN's memory and fault rate saturate?
set -u
export HERE=/proj/aigstaff/sohroy/profiling OUT=/proj/aigstaff/sohroy/profiling/investigate_20261007
source "$HERE/diag_lib.sh" || exit 1
eval "$(sed -n '/^run_sampled()/,/^}/p' $HERE/investigate.sh)"
CFG=latest_zendnn
run_sampled long_default 2048 512 100 ZENDNNL_NONE=1
echo done
