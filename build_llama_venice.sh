#!/bin/bash
# Native build on the Venice pod (separate build dirs: build_venice = stock CPU, build_zendnn_venice = ZenDNN)
# of the latest llama.cpp (expert-limit and rows-per-expert rules removed from ggml-zendnn supports_op).
# ZENDNN_ROOT is overridable (default: the latest ZenDNN built on Turin, which dispatches at run time).
set -eu
SRC=/proj/aigstaff/sohroy/llama.cpp
ZENDNN_ROOT=${ZENDNN_ROOT:-/proj/aigstaff/sohroy/ZenDNN-latest/build/install}
CMAKE=/proj/aigstaff/sohroy/tools/cmake-3.31.6-linux-x86_64/bin/cmake
cd "$SRC"; git log -1 --format='%h %s'; git status --short
"$CMAKE" -B build_venice -DCMAKE_BUILD_TYPE=Release
"$CMAKE" --build build_venice -j "$(nproc)" --target llama-bench
"$CMAKE" -B build_zendnn_venice -DCMAKE_BUILD_TYPE=Release -DGGML_ZENDNN=ON -DZENDNN_ROOT="$ZENDNN_ROOT"
"$CMAKE" --build build_zendnn_venice -j "$(nproc)" --target llama-bench
ls -l build_venice/bin/llama-bench build_zendnn_venice/bin/llama-bench
