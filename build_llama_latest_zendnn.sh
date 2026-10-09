#!/bin/bash
# Builds latest llama.cpp (/proj/aigstaff/sohroy/llama.cpp, expert-count limit removed from ggml-zendnn
# supports_op) against the latest ZenDNN (/proj/aigstaff/sohroy/ZenDNN-latest, built with cmake 3.31).
# Run inside the pod (znver5 -march=native): runas sohroy bash build_llama_latest_zendnn.sh
set -eu
SRC=/proj/aigstaff/sohroy/llama.cpp
ZENDNN_ROOT=/proj/aigstaff/sohroy/ZenDNN-latest/build/install
CMAKE=/proj/aigstaff/sohroy/tools/cmake-3.31.6-linux-x86_64/bin/cmake
cd "$SRC"; git log -1 --format='%h %s'; git status --short
"$CMAKE" -B build_zendnn -DCMAKE_BUILD_TYPE=Release -DGGML_ZENDNN=ON -DZENDNN_ROOT="$ZENDNN_ROOT"
"$CMAKE" --build build_zendnn -j "$(nproc)" --target llama-bench
"$CMAKE" -B build -DCMAKE_BUILD_TYPE=Release
"$CMAKE" --build build -j "$(nproc)" --target llama-bench
ls -l build/bin/llama-bench build_zendnn/bin/llama-bench; ldd build_zendnn/bin/llama-bench | grep -i zendnn
