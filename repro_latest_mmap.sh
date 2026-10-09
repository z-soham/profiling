#!/bin/bash
# reproduce: zen_latest build, mmap (default) vs dio, pp256, 1 rep
M=/proj/aigstaff/sohroy/llama.cpp-matrix
for lm in mmap dio; do
  echo "== -lm $lm"
  env -u LD_PRELOAD LD_LIBRARY_PATH=$M/build_zen_latest/bin:/proj/aigstaff/sohroy/ZenDNN-latest/build/install/zendnnl/lib \
    ZENDNNL_MATMUL_ALGO=1 OMP_NUM_THREADS=32 \
    numactl --physcpubind=192-223 --membind=6 -- $M/build_zen_latest/bin/llama-bench \
    -m /proj/rdi/staff/sohroy/models/Qwen3.6-35B-A3B-BF16.gguf -t 32 -b 4096 -ub 4096 -fa on -ctk f16 -ctv f16 -lm $lm \
    -p 256 -n 0 -r 1 -o md 2>&1 | grep -v -E "^\s*$" | tail -6
  echo "exit=${PIPESTATUS[0]}"
done
