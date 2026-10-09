#!/bin/bash
# Builds branch zendnn-concat-fix of /proj/aigstaff/sohroy/llama.cpp with ZenDNN into its OWN dir (build_zendnn_concat);
# existing build/, build_zendnn/, build_venice/ are not touched. Run in the pod: runas sohroy bash build_concat_zendnn.sh
set -eu
SRC=/proj/aigstaff/sohroy/llama.cpp
ZENDNN_ROOT=/proj/aigstaff/sohroy/ZenDNN-latest/build/install
CMAKE=/proj/aigstaff/sohroy/tools/cmake-3.31.6-linux-x86_64/bin/cmake
cd "$SRC"; git log -1 --format='%h %s'; git status --short
"$CMAKE" -B build_zendnn_concat -DCMAKE_BUILD_TYPE=Release -DGGML_ZENDNN=ON -DZENDNN_ROOT="$ZENDNN_ROOT"
"$CMAKE" --build build_zendnn_concat -j "$(nproc)" --target llama-bench test-backend-ops
ls -l build_zendnn_concat/bin/llama-bench build_zendnn_concat/bin/test-backend-ops
ldd build_zendnn_concat/bin/llama-bench | grep -i zendnn
