#!/bin/bash
set -e
S=/proj/aigstaff/sohroy/llama.cpp-matrix/src_concat; B=$S/build_check
C=/proj/aigstaff/sohroy/tools/cmake-3.31.6-linux-x86_64/bin/cmake
$C -S $S -B $B -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF >/dev/null 2>&1
$C --build $B -j 32 --target ggml-cpu ggml-base ggml 2>&1 | grep -E "error|Linking" | tail -3
cd /proj/aigstaff/sohroy/profiling
for f in concat_check concat_view_check; do
  g++ -O1 -std=c++17 $f.cpp -I$S/ggml/include -L$B/bin -lggml -lggml-base -lggml-cpu -Wl,-rpath,$B/bin -o ${f}_rebased && ./${f}_rebased
done
