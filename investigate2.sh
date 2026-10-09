#!/bin/bash
# CROSS_WARM / prepack variants: memory and page faults, and throughput, at pp2048 ub512 (10 passes)
set -u
export HERE=/proj/aigstaff/sohroy/profiling OUT=/proj/aigstaff/sohroy/profiling/investigate_20261007
source "$HERE/diag_lib.sh" || exit 1
PARTS=""
eval "$(sed -n '/^run_sampled()/,/^}/p' $HERE/investigate.sh)"
CFG=latest_zendnn
run_sampled mem_crosswarm0 2048 512 10 ZENDNNL_GRP_MATMUL_CROSS_WARM=0
run_sampled mem_crosswarm0_prepack0 2048 512 10 ZENDNNL_GRP_MATMUL_CROSS_WARM=0 ZENDNNL_GRP_MATMUL_PREPACK=0
echo done
