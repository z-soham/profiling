#!/bin/bash
set -u
export HERE=/proj/aigstaff/sohroy/profiling OUT=/proj/aigstaff/sohroy/profiling/investigate_20261007
source "$HERE/diag_lib.sh" || exit 1
eval "$(sed -n '/^run_sampled()/,/^}/p' $HERE/investigate.sh)"
CFG=latest_zendnn
run_sampled ub_zendnn_cw0 8192 512,4096 6 ZENDNNL_GRP_MATMUL_CROSS_WARM=0
run_sampled ub_zendnn_default_r6 8192 512,4096 6 ZENDNNL_NONE=1
echo done
