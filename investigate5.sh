#!/bin/bash
# does disabling the custom-kernel pack (prefill uses AOCL DLP under ALGO_1) remove the extra memory / faults / ramp?
set -u
export HERE=/proj/aigstaff/sohroy/profiling OUT=/proj/aigstaff/sohroy/profiling/investigate_20261007
source "$HERE/diag_lib.sh" || exit 1
eval "$(sed -n '/^run_sampled()/,/^}/p' $HERE/investigate.sh)"
CFG=latest_zendnn
run_sampled mem_ck0 2048 512 14 ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL=0
run_sampled ub_zendnn_ck0 8192 512,4096 6 ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL=0
echo done
