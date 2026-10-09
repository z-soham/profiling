#!/bin/bash
# Builds the patched llama.cpp (branch turin-perf of /proj/aigstaff/sohroy/llama.cpp-turin, based on
# 8fe90e1fb, the commit of sacsharm's builds) twice, like sacsharm's: stock (build/) and ZenDNN
# (build_zendnn/, against sacsharm's prebuilt ZenDNNL -- the same libzendnnl.so, md5 73cb16d9, that
# the Benchmarking sweeps load -- instead of the 30-45 min download-and-build).
# Run inside the pod (znver5 -march=native): runas sohroy bash build_llama_turin.sh
set -eu
SRC=${SRC:-/proj/aigstaff/sohroy/llama.cpp-turin}
ZENDNN_ROOT=${ZENDNN_ROOT:-/proj/rdi/staff/sacsharm/ZenDNN/build/install}
TARGETS=(llama-bench llama-server llama-perplexity llama-batched-bench llama-completion)
JOBS=${JOBS:-$(nproc)}
# the pod image has no cmake; the vLLM venv ships one (pip cmake)
CMAKE=${CMAKE:-$(command -v cmake || echo /proj/aigstaff/sacsharm/vllm/.venv/bin/cmake)}

cd "$SRC"
git log -1 --format='%h %s'; git status --short

"$CMAKE" -B build -DCMAKE_BUILD_TYPE=Release
"$CMAKE" --build build -j "$JOBS" --target "${TARGETS[@]}"

"$CMAKE" -B build_zendnn -DCMAKE_BUILD_TYPE=Release -DGGML_ZENDNN=ON -DZENDNN_ROOT="$ZENDNN_ROOT"
"$CMAKE" --build build_zendnn -j "$JOBS" --target "${TARGETS[@]}"

for b in build build_zendnn; do
  echo "== $b"; ls -l "$b/bin/llama-bench"; readelf -d "$b/bin/llama-bench" | grep -E 'RUNPATH|RPATH'
done
