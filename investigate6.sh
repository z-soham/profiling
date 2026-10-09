#!/bin/bash
# pinning ZENDNNL_GRP_MATMUL_ALGO=1 (code: pinned algos keep one in-place AOCL layout, no CK decode pack)
set -u
export HERE=/proj/aigstaff/sohroy/profiling OUT=/proj/aigstaff/sohroy/profiling/investigate_20261007
source "$HERE/diag_lib.sh" || exit 1
eval "$(sed -n '/^run_sampled()/,/^}/p' $HERE/investigate.sh)"
CFG=latest_zendnn
run_sampled mem_pinalgo1 2048 512 14 ZENDNNL_GRP_MATMUL_ALGO=1
run_sampled ub_zendnn_pinalgo1 8192 512,4096 6 ZENDNNL_GRP_MATMUL_ALGO=1
echo done
