#!/bin/bash
# re-benchmark + re-profile with binaries built from clean commit 711391c; records host and commit
cd /proj/aigstaff/sohroy/profiling
O=concat_run2; mkdir -p $O; { hostname; git -C /proj/aigstaff/sohroy/llama.cpp log -1 --format='%H %s'; } > $O/provenance.txt
OUT=$PWD/$O bash bench_concat.sh > $O/driver.log 2>&1
ROOT=$PWD/opprof_concat_run2 bash opprof_concat.sh > opprof_concat2.log 2>&1
echo done > $O/ALL_DONE
