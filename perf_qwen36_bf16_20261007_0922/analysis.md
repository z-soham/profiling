# perf analysis: perf_qwen36_bf16_20261007_0922

Qwen3.6-35B-A3B BF16 prefill with the optimised commands of turin-optimised-commands-2026-09-30.md (boost off): llama.cpp setup B on sacsharm's existing stock and ZenDNN builds (weights loaded with --load-mode dio into local memory), vLLM setup C. 32 cores / 1 NUMA node. perf attached after load and warm-up (vLLM: to its worker process); one measurement per window. See perf_llama.sh.

## Runs

| run | prefill t/s while profiled (median pass) | passes | start-up s | resident memory per NUMA node |
|---|---|---|---|---|
| latest_stock_pp8192_ub512 | 291.0 | 3 | 115 | N6=65.1GB N0=0.0GB |
| latest_zendnn_algo3_pp8192_ub512 | 270.0 | 2 | 156 | N6=117.4GB |
| latest_zendnn_pp8192_ub512 | 287.9 | 3 | 159 | N6=117.4GB N0=0.0GB |
| vllm_pp8192 | 630.8 | 16 | 327 | N7=0.0GB N1=0.0GB N6=220.7GB N5=0.0GB N4=0.0GB |

## Counters (whole process, during the windows)

| run | cores busy | GHz | IPC | retiring % | frontend % | bad spec % | backend: cpu % | backend: memory % | FP G-ops/s (all) | FP G-ops/s (mul-add) |
|---|---|---|---|---|---|---|---|---|---|---|
| latest_stock_pp8192_ub512 | 26.4 | 2.69 | 1.74 | 20.2 | 8.2 | 0.8 | 15.6 | 55.3 | 1665 | 1639 |
| latest_zendnn_algo3_pp8192_ub512 | 23.5 | 2.71 | 1.64 | 19.1 | 13.5 | 0.5 | 6.1 | 60.7 | 1601 | 1588 |
| latest_zendnn_pp8192_ub512 | 26.2 | 2.67 | 1.64 | 19.1 | 16.3 | 0.5 | 7.0 | 56.9 | 1669 | 1656 |
| vllm_pp8192 | 24.9 | 2.72 | 2.19 | 29.3 | 7.9 | 0.3 | 28.3 | 34.2 | 3602 | 3595 |

Top-down columns are % of dispatch slots (PipelineL1/L2). FP columns: retired FP operations per second (fp_ret_sse_avx_ops.all / .mac_flops), across the whole process. Their bf16 and packed-fp32 umasks read 0 on this CPU, so the split by data type is not available.

## Where L1 data-cache fills come from (% of fills) and estimated DRAM read traffic

| run | L2 % | L3 same CCX % | other CCX % | DRAM this socket % | DRAM other socket % | L1 fills from DRAM GB/s | L2 prefetch misses to DRAM GB/s | DRAM read est. GB/s |
|---|---|---|---|---|---|---|---|---|
| latest_stock_pp8192_ub512 | 90.2 | 5.8 | 0.6 | 3.4 | 0.0 | 34.1 | 41.8 | 76 |
| latest_zendnn_algo3_pp8192_ub512 | 93.0 | 2.2 | 0.7 | 4.1 | 0.0 | 19.4 | 45.2 | 65 |
| latest_zendnn_pp8192_ub512 | 90.7 | 3.8 | 0.7 | 4.8 | 0.0 | 25.2 | 46.9 | 72 |
| vllm_pp8192 | 96.6 | 1.6 | 0.1 | 1.7 | 0.0 | 11.8 | 30.2 | 42 |

DRAM near/far = this socket / the other socket. The estimate counts 64-byte lines from L1 fills that came from DRAM plus L2 hardware prefetches that missed L3; it excludes writes. The pods expose no uncore (data-fabric) counters to measure DRAM bandwidth directly. NUMA node 1 has 3 of the socket's 12 DDR5 channels (NPS4), roughly 140 GB/s peak.

## latest_stock_pp8192_ub512

### Where the cycles go (flat profile, all threads)

| category | % of cycles |
|---|---|
| ggml: bf16 dot product (MUL_MAT_ID experts on CPU) | 42.9 |
| ggml: tinyBLAS/llamafile GEMM (dense matmul on CPU) | 16.0 |
| ggml: flash attention | 12.8 |
| OpenMP runtime (spin-wait at barriers, fork/join) | 5.6 |
| ggml: GDN recurrence + conv1d | 5.0 |
| ggml: other ops (norms, elementwise, softmax, ...) | 5.0 |
| ggml: graph loop (ggml_barrier, inlined small ops) | 4.6 |
| ggml: f32/f16 dot product | 2.4 |
| ggml: type conversion | 2.3 |
| ggml: CONCAT (GDN conv state) | 0.7 |
| kernel | 0.1 |

### Top symbols

| % | object | symbol |
|---|---|---|
| 42.87 | libggml-cpu.so.0.26.0 | ggml_vec_dot_bf16 |
| 14.49 | libggml-cpu.so.0.26.0 | void (anonymous namespace)::tinyBLAS<32, float __vector(16), std::bfloat16_t __vector(32), |
| 12.80 | libggml-cpu.so.0.26.0 | ggml_compute_forward_flash_attn_ext_tiled(ggml_compute_params const*, ggml_tensor*, int, i |
| 5.16 | libomp.so.5 | 0x000000000007df00 |
| 4.62 | libggml-cpu.so.0.26.0 | ggml_graph_compute_thread.isra.0 |
| 3.34 | libggml-cpu.so.0.26.0 | ggml_compute_forward_gated_delta_net |
| 2.31 | libggml-cpu.so.0.26.0 | ggml_vec_dot_f32 |
| 1.75 | libggml-cpu.so.0.26.0 | ggml_cpu_fp32_to_bf16 |
| 1.69 | libggml-cpu.so.0.26.0 | ggml_compute_forward_ssm_conv |
| 1.05 | libggml-cpu.so.0.26.0 | ggml_compute_forward_mul |
| 0.97 | libggml-cpu.so.0.26.0 | ggml_compute_forward_rms_norm_mul_fused |
| 0.88 | libggml-cpu.so.0.26.0 | void (anonymous namespace)::tinyBLAS<16, float __vector(16), float __vector(16), float, fl |
| 0.71 | libggml-cpu.so.0.26.0 | ggml_compute_forward_concat |
| 0.68 | libggml-cpu.so.0.26.0 | ggml_vec_silu_f32 |
| 0.59 | libggml-cpu.so.0.26.0 | ggml_vec_swiglu_f32 |
| 0.51 | libggml-cpu.so.0.26.0 | ggml_cpu_fp16_to_fp32 |
| 0.50 | libggml-cpu.so.0.26.0 | void (anonymous namespace)::tinyBLAS<32, float __vector(16), std::bfloat16_t __vector(32), |
| 0.48 | libggml-cpu.so.0.26.0 | ggml_compute_forward_add_non_quantized |
| 0.33 | libomp.so.5 | 0x00000000000e1140 |
| 0.27 | libggml-cpu.so.0.26.0 | ggml_compute_forward_rms_norm |

### Per ggml op (cpu-clock stacks: 195269 samples, 32 threads, main thread 919673)

| op (main thread's stack) | % of wall time | threads doing work (of 32) | % of all CPU samples: work | % of all CPU samples: spinning |
|---|---|---|---|---|
| [leaf] ggml: bf16 dot product (MUL_MAT_ID experts on CPU) | 37.5 | 32.0 | 45.8 | 0.0 |
| concat | 19.9 | 2.0 | 0.8 | 0.0 |
| [leaf] ggml: tinyBLAS/llamafile GEMM (dense matmul on CPU) | 13.8 | 29.2 | 17.0 | 0.0 |
| flash_attn_ext | 6.2 | 31.2 | 7.6 | 0.0 |
| (ggml barrier: waiting for the slowest thread) | 3.2 | 24.1 | 0.0 | 6.3 |
| gated_delta_net | 3.0 | 29.6 | 4.0 | 0.0 |
| [leaf] ggml: graph loop (ggml_barrier, inlined small ops) | 2.9 | 32.0 | 3.2 | 0.0 |
| [leaf] libc (memcpy / memset) | 2.4 | 30.1 | 3.0 | 0.0 |
| [leaf] ggml: f32/f16 dot product | 1.9 | 28.3 | 2.1 | 0.0 |
| [leaf] ggml: type conversion | 1.7 | 29.7 | 2.1 | 0.0 |
| ssm_conv | 1.1 | 18.9 | 1.8 | 0.0 |
| rms_norm_mul | 1.1 | 28.9 | 1.0 | 0.0 |
| mul | 0.9 | 31.0 | 1.2 | 0.0 |
| unary | 0.8 | 27.7 | 0.6 | 0.0 |
| [leaf] ggml: other ops (norms, elementwise, softmax, ...) | 0.6 | 27.6 | 0.7 | 0.0 |
| mul_mat | 0.6 | 28.8 | 0.1 | 0.7 |
| rms_norm | 0.5 | 25.4 | 0.4 | 0.0 |
| glu | 0.4 | 30.4 | 0.7 | 0.0 |
| get_rows | 0.4 | 16.8 | 0.0 | 0.0 |
| add | 0.3 | 28.7 | 0.4 | 0.0 |
| [leaf] other | 0.2 | 16.9 | 0.1 | 0.0 |
| scale | 0.1 | 25.7 | 0.2 | 0.0 |
| argsort | 0.1 | 27.3 | 0.2 | 0.0 |
| sigmoid | 0.1 | 22.0 | 0.1 | 0.0 |
| [leaf] ggml-base: other | 0.1 | 3.6 | 0.0 | 0.0 |
| mul_mat_id | 0.0 | 32.0 | 0.0 | 0.0 |
| clamp | 0.0 | 31.5 | 0.0 | 0.0 |
| soft_max | 0.0 | 28.0 | 0.0 | 0.0 |
| set_rows | 0.0 | 28.0 | 0.0 | 0.0 |
| dup_bytes | 0.0 | 31.0 | 0.0 | 0.0 |
| sum_rows | 0.0 | 25.0 | 0.0 | 0.0 |

'% of wall time' is the main thread's samples. 'threads doing work' counts, per main-thread sample in that op, the samples of all threads within 6 ms that were not spinning in libomp.

### Threads: 32, work share min/median/max = 92/93/96% (the rest is spinning in libomp)

## latest_zendnn_algo3_pp8192_ub512

### Where the cycles go (flat profile, all threads)

| category | % of cycles |
|---|---|
| ZenDNN/AOCL-DLP GEMM kernels (incl. JIT code) | 37.5 |
| OpenMP runtime (spin-wait at barriers, fork/join) | 25.4 |
| ggml: flash attention | 11.3 |
| ggml: other ops (norms, elementwise, softmax, ...) | 5.5 |
| ggml: GDN recurrence + conv1d | 5.3 |
| ggml: type conversion | 3.2 |
| ggml: f32/f16 dot product | 2.4 |
| libc (memcpy / memset) | 1.0 |
| ggml: CONCAT (GDN conv state) | 0.8 |
| kernel | 0.4 |
| ggml: tinyBLAS/llamafile GEMM (dense matmul on CPU) | 0.1 |
| ggml: bf16 dot product (MUL_MAT_ID experts on CPU) | 0.1 |

### Top symbols

| % | object | symbol |
|---|---|---|
| 21.58 | libomp.so.5 | 0x000000000007df00 |
| 11.28 | libggml-cpu.so.0.26.0 | ggml_compute_forward_flash_attn_ext_tiled(ggml_compute_params const*, ggml_tensor*, int, i |
| 3.49 | libggml-cpu.so.0.26.0 | ggml_compute_forward_gated_delta_net |
| 2.53 | libggml-base.so.0.26.0 | ggml_fp32_to_bf16_row_ref |
| 2.33 | libggml-cpu.so.0.26.0 | ggml_vec_dot_f32 |
| 1.81 | libggml-cpu.so.0.26.0 | ggml_compute_forward_ssm_conv |
| 1.35 | libggml-cpu.so.0.26.0 | ggml_compute_forward_mul |
| 1.33 | libomp.so.5 | 0x00000000000e1140 |
| 1.01 | libggml-cpu.so.0.26.0 | ggml_compute_forward_rms_norm_mul_fused |
| 0.78 | libggml-cpu.so.0.26.0 | ggml_vec_silu_f32 |
| 0.77 | libggml-cpu.so.0.26.0 | ggml_compute_forward_concat |
| 0.68 | libggml-cpu.so.0.26.0 | ggml_vec_swiglu_f32 |
| 0.57 | libggml-cpu.so.0.26.0 | ggml_compute_forward_add_non_quantized |
| 0.44 | libggml-cpu.so.0.26.0 | ggml_cpu_fp16_to_fp32 |
| 0.39 | libzendnnl.so | 0x00000000005f1b85 |
| 0.37 | [JIT] tid 921205 | 0x00007c9878f17165 |
| 0.35 | libzendnnl.so | 0x00000000005f1b69 |
| 0.33 | libomp.so.5 | 0x000000000007dfc6 |
| 0.33 | libzendnnl.so | 0x00000000005f358b |
| 0.31 | [JIT] tid 921205 | 0x00007c9878f171b9 |

### Per ggml op (cpu-clock stacks: 168601 samples, 32 threads, main thread 921205)

| op (main thread's stack) | % of wall time | threads doing work (of 32) | % of all CPU samples: work | % of all CPU samples: spinning |
|---|---|---|---|---|
| ZenDNN matmul (AOCL-DLP GEMM kernels) | 53.4 | 18.0 | 44.2 | 11.2 |
| concat | 18.8 | 2.0 | 0.8 | 0.0 |
| flash_attn_ext | 4.9 | 31.3 | 6.9 | 0.0 |
| ZenDNN mul_mat_id | 4.6 | 15.6 | 0.1 | 0.2 |
| (ggml barrier: waiting for the slowest thread) | 3.6 | 21.8 | 0.0 | 6.9 |
| gated_delta_net | 2.8 | 29.0 | 4.2 | 0.0 |
| [leaf] ggml: type conversion | 2.6 | 20.8 | 3.0 | 0.0 |
| [leaf] ggml: f32/f16 dot product | 1.5 | 29.9 | 2.1 | 0.0 |
| [leaf] libc (memcpy / memset) | 1.3 | 20.1 | 2.3 | 0.0 |
| ssm_conv | 1.1 | 18.6 | 1.9 | 0.0 |
| mul | 0.8 | 21.2 | 1.4 | 0.0 |
| rms_norm_mul | 0.7 | 27.1 | 1.0 | 0.0 |
| add | 0.6 | 23.9 | 0.6 | 0.0 |
| unary | 0.5 | 25.8 | 0.8 | 0.0 |
| rms_norm | 0.5 | 25.7 | 0.3 | 0.0 |
| glu | 0.4 | 19.2 | 0.7 | 0.0 |
| [leaf] ggml: other ops (norms, elementwise, softmax, ...) | 0.4 | 23.5 | 0.6 | 0.0 |
| [leaf] other | 0.3 | 18.8 | 0.1 | 0.0 |
| get_rows | 0.2 | 15.3 | 0.0 | 0.0 |
| argsort | 0.1 | 18.0 | 0.2 | 0.0 |
| scale | 0.1 | 18.9 | 0.2 | 0.0 |
| [leaf] ggml: tinyBLAS/llamafile GEMM (dense matmul on CPU) | 0.1 | 24.9 | 0.1 | 0.0 |
| [leaf] ggml-base: other | 0.1 | 8.0 | 0.0 | 0.0 |
| (OpenMP: idle / fork-join) | 0.1 | 24.8 | 0.0 | 9.7 |
| [leaf] ggml: bf16 dot product (MUL_MAT_ID experts on CPU) | 0.1 | 28.6 | 0.1 | 0.0 |
| mul_mat | 0.1 | 22.2 | 0.0 | 0.1 |
| [leaf] ggml: graph loop (ggml_barrier, inlined small ops) | 0.0 | 18.0 | 0.0 | 0.0 |
| sigmoid | 0.0 | 20.0 | 0.1 | 0.0 |
| clamp | 0.0 | 21.0 | 0.0 | 0.0 |
| soft_max | 0.0 | 24.0 | 0.0 | 0.0 |
| div | 0.0 | 19.0 | 0.0 | 0.0 |
| [leaf] kernel | 0.0 | 13.0 | 0.0 | 0.0 |

'% of wall time' is the main thread's samples. 'threads doing work' counts, per main-thread sample in that op, the samples of all threads within 6 ms that were not spinning in libomp.

### Threads: 32, work share min/median/max = 66/70/91% (the rest is spinning in libomp)

## latest_zendnn_pp8192_ub512

### Where the cycles go (flat profile, all threads)

| category | % of cycles |
|---|---|
| OpenMP runtime (spin-wait at barriers, fork/join) | 30.9 |
| ZenDNN/AOCL-DLP GEMM kernels (incl. JIT code) | 28.3 |
| ggml: flash attention | 10.8 |
| ggml: other ops (norms, elementwise, softmax, ...) | 5.1 |
| ggml: GDN recurrence + conv1d | 5.1 |
| ggml: type conversion | 3.2 |
| ggml: f32/f16 dot product | 2.2 |
| libc (memcpy / memset) | 0.8 |
| ggml: CONCAT (GDN conv state) | 0.7 |
| kernel | 0.6 |
| ggml: tinyBLAS/llamafile GEMM (dense matmul on CPU) | 0.1 |
| ggml: bf16 dot product (MUL_MAT_ID experts on CPU) | 0.1 |

### Top symbols

| % | object | symbol |
|---|---|---|
| 25.53 | libomp.so.5 | 0x000000000007df00 |
| 10.80 | libggml-cpu.so.0.26.0 | ggml_compute_forward_flash_attn_ext_tiled(ggml_compute_params const*, ggml_tensor*, int, i |
| 3.35 | libggml-cpu.so.0.26.0 | ggml_compute_forward_gated_delta_net |
| 2.51 | libggml-base.so.0.26.0 | ggml_fp32_to_bf16_row_ref |
| 2.19 | libggml-cpu.so.0.26.0 | ggml_vec_dot_f32 |
| 1.71 | libggml-cpu.so.0.26.0 | ggml_compute_forward_ssm_conv |
| 1.59 | libomp.so.5 | 0x00000000000e1140 |
| 1.28 | libggml-cpu.so.0.26.0 | ggml_compute_forward_mul |
| 0.95 | libggml-cpu.so.0.26.0 | ggml_compute_forward_rms_norm_mul_fused |
| 0.81 | [JIT] tid 920363 | 0x00007aade8d93165 |
| 0.80 | [JIT] tid 920363 | 0x00007aade8d93144 |
| 0.74 | libggml-cpu.so.0.26.0 | ggml_vec_silu_f32 |
| 0.72 | libggml-cpu.so.0.26.0 | ggml_compute_forward_concat |
| 0.66 | [JIT] tid 920363 | 0x00007aade8d931d8 |
| 0.65 | [JIT] tid 920363 | 0x00007aade8d93171 |
| 0.65 | [JIT] tid 920363 | 0x00007aade8d931b9 |
| 0.65 | [JIT] tid 920363 | 0x00007aade8d930f0 |
| 0.64 | [JIT] tid 920363 | 0x00007aade8d931a7 |
| 0.64 | [JIT] tid 920363 | 0x00007aade8d93177 |
| 0.64 | [JIT] tid 920363 | 0x00007aade8d9312f |

### Per ggml op (cpu-clock stacks: 188335 samples, 32 threads, main thread 920363)

| op (main thread's stack) | % of wall time | threads doing work (of 32) | % of all CPU samples: work | % of all CPU samples: spinning |
|---|---|---|---|---|
| ZenDNN matmul (AOCL-DLP GEMM kernels) | 51.0 | 18.8 | 38.3 | 0.4 |
| concat | 19.4 | 2.0 | 0.7 | 0.0 |
| ZenDNN mul_mat_id | 5.3 | 13.1 | 0.1 | 0.2 |
| flash_attn_ext | 4.8 | 31.5 | 6.2 | 0.0 |
| (ggml barrier: waiting for the slowest thread) | 3.5 | 21.4 | 0.0 | 6.5 |
| gated_delta_net | 3.1 | 29.3 | 4.0 | 0.0 |
| [leaf] ggml: type conversion | 2.7 | 19.2 | 2.8 | 0.0 |
| [leaf] ggml: f32/f16 dot product | 1.6 | 29.2 | 2.1 | 0.0 |
| [leaf] libc (memcpy / memset) | 1.5 | 19.2 | 2.5 | 0.0 |
| ssm_conv | 1.2 | 18.9 | 1.8 | 0.0 |
| mul | 1.1 | 22.1 | 1.4 | 0.0 |
| rms_norm_mul | 1.0 | 26.4 | 1.0 | 0.0 |
| unary | 0.6 | 27.8 | 0.7 | 0.0 |
| [leaf] ggml: other ops (norms, elementwise, softmax, ...) | 0.5 | 22.6 | 0.5 | 0.0 |
| glu | 0.4 | 18.1 | 0.6 | 0.0 |
| [leaf] other | 0.4 | 17.1 | 0.1 | 0.0 |
| rms_norm | 0.4 | 26.5 | 0.4 | 0.0 |
| add | 0.3 | 24.0 | 0.5 | 0.0 |
| get_rows | 0.3 | 14.2 | 0.0 | 0.0 |
| mul_mat | 0.2 | 25.3 | 0.0 | 0.1 |
| scale | 0.2 | 25.6 | 0.2 | 0.0 |
| (OpenMP: idle / fork-join) | 0.2 | 19.0 | 0.0 | 28.3 |
| [leaf] ggml-base: other | 0.1 | 14.1 | 0.0 | 0.0 |
| argsort | 0.1 | 20.4 | 0.2 | 0.0 |
| [leaf] ggml: tinyBLAS/llamafile GEMM (dense matmul on CPU) | 0.1 | 24.0 | 0.1 | 0.0 |
| [leaf] ggml: bf16 dot product (MUL_MAT_ID experts on CPU) | 0.1 | 32.0 | 0.1 | 0.0 |
| soft_max | 0.0 | 21.0 | 0.0 | 0.0 |
| sigmoid | 0.0 | 21.0 | 0.1 | 0.0 |
| [leaf] kernel | 0.0 | 1.0 | 0.0 | 0.0 |
| sum_rows | 0.0 | 20.0 | 0.0 | 0.0 |

'% of wall time' is the main thread's samples. 'threads doing work' counts, per main-thread sample in that op, the samples of all threads within 6 ms that were not spinning in libomp.

### Threads: 32, work share min/median/max = 59/64/80% (the rest is spinning in libomp)

## vllm_pp8192

### Where the cycles go (flat profile, all threads)

| category | % of cycles |
|---|---|
| ZenDNN/AOCL-DLP GEMM kernels (incl. JIT code) | 38.6 |
| vLLM C++: attention | 30.9 |
| OpenMP runtime (spin-wait at barriers, fork/join) | 13.9 |
| PyTorch runtime / aten ops | 3.8 |
| other | 2.2 |
| vLLM C++: other kernels | 1.8 |
| zentorch/ZenDNN: fused MoE (incl. its GEMMs) | 1.1 |
| libc (memcpy / memset) | 0.3 |
| vLLM C++: GDN / conv1d | 0.3 |
| zentorch/ZenDNN GEMM kernels | 0.3 |
| kernel | 0.1 |

### Top symbols

| % | object | symbol |
|---|---|---|
| 30.26 | _C.abi3.so | void cpu_attention::(anonymous namespace)::TileGemm82<c10::BFloat16>::gemm_micro<8>(float* |
| 13.80 | libiomp5.so | kmp_flag_64<false, true>::wait(kmp_info*, int, void*) |
| 3.31 | libtorch_cpu.so | void c10::function_ref<void (char**, long const*, long, long)>::callback_fn<at::native::AV |
| 1.33 | [JIT] tid 922302 | 0x00007339efb43169 |
| 1.07 | libzentorch.so | zendnnl::lowoha::matmul::(anonymous namespace)::moe_weighted_reduce_avx512_bf16(zendnnl::l |
| 1.07 | [JIT] tid 922302 | 0x00007339efb43148 |
| 1.05 | [JIT] tid 922302 | 0x00007339efb431b7 |
| 1.05 | [JIT] tid 922302 | 0x00007339efb431b1 |
| 1.05 | [JIT] tid 922302 | 0x00007339efb431ab |
| 1.03 | [JIT] tid 922302 | 0x00007339efb431bd |
| 1.01 | [JIT] tid 922302 | 0x00007339efb4317b |
| 1.01 | [JIT] tid 922302 | 0x00007339efb4316f |
| 1.01 | [JIT] tid 922302 | 0x00007339efb4319f |
| 0.99 | [JIT] tid 922302 | 0x00007339efb431a5 |
| 0.97 | [JIT] tid 922302 | 0x00007339efb43175 |
| 0.95 | [JIT] tid 922302 | 0x00007339efb4319c |
| 0.92 | [JIT] tid 922302 | 0x00007339efb430f4 |
| 0.92 | [JIT] tid 922302 | 0x00007339efb43196 |
| 0.92 | [JIT] tid 922302 | 0x00007339efb43190 |
| 0.91 | [JIT] tid 922302 | 0x00007339efb43184 |

## vLLM per-op split: vllm_profile_pp8192 (torch profiler, one generate call; self CPU total 13.000s)

| bucket | self CPU ms |
|---|---|
| MoE experts (zentorch fused MoE incl. its copies) | 6183 |
| attention | 3200 |
| linear / GEMM (zentorch_linear) | 2370 |
| GDN / linear attention (chunk rule, conv1d, gating) | 477 |
| (compiled-graph / step wrappers: self time) | 340 |
| other ops (top-50 rows) | 318 |
| concatenation (aten::cat) | 112 |

| op | self CPU % | self CPU | calls |
|---|---|---|---|
| zentorch::zentorch_fused_moe | 44.61% | 5.800s | 80 |
| _C::cpu_attention_with_kv_cache | 24.57% | 3.194s | 20 |
| zentorch_linear_unary | 18.08% | 2.350s | 540 |
| _C::chunk_gated_delta_rule_cpu | 2.76% | 358.794ms | 60 |
| ## Call CompiledFxGraph None ## | 2.49% | 324.136ms | 2 |
| zentorch::fused_moe::pass2_parallel_memcpy | 2.34% | 304.432ms | 80 |
| aten::copy_ | 1.93% | 250.938ms | 597 |
| aten::cat | 0.86% | 112.052ms | 66 |
| _C::causal_conv1d_fwd_cpu | 0.71% | 91.796ms | 60 |
| zentorch::fused_moe::pass1_active_set_build | 0.54% | 70.456ms | 80 |
| aten::topk | 0.30% | 39.143ms | 80 |
| vllm::cpu_gdn_attention_core | 0.19% | 24.196ms | 60 |
| zentorch::zentorch_linear_unary | 0.15% | 20.021ms | 542 |
| execute_context_1(4096)_generation_0(0) | 0.11% | 14.599ms | 2 |
| zentorch::fused_moe::scratchpad_allocation | 0.04% | 5.463ms | 80 |

