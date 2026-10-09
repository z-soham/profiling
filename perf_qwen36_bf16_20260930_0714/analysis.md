# perf analysis: perf_qwen36_bf16_20260930_0714

Qwen3.6-35B-A3B BF16 prefill with the optimised commands of turin-optimised-commands-2026-09-30.md (boost off): llama.cpp setup B on sacsharm's existing stock and ZenDNN builds (weights loaded with --load-mode dio into local memory), vLLM setup C. 32 cores / 1 NUMA node. perf attached after load and warm-up (vLLM: to its worker process); one measurement per window. See perf_llama.sh.

## Runs

| run | prefill t/s while profiled (median pass) | passes | start-up s | resident memory per NUMA node |
|---|---|---|---|---|
| base_stock_pp256_ub512 | 256.1 | 110 | 87 | N1=65.5GB N0=0.0GB |
| base_stock_pp8192_ub512 | 278.0 | 3 | 86 | N1=65.7GB N0=0.0GB |
| base_zendnn_pp256_ub512 | 273.3 | 117 | 87 | N1=68.1GB N0=0.0GB |
| base_zendnn_pp8192_ub512 | 284.0 | 3 | 87 | N1=68.3GB N0=0.0GB |
| vllm_pp256 | 462.5 | 300 | 206 | worker: N1=68.5GB N5=0.2GB, rest 0 (read in an earlier attempt, same command) |
| vllm_pp8192 | 631.5 | 16 | 352 | worker: N1=68.5GB N5=0.2GB, rest 0 (read in an earlier attempt, same command) |

## Counters (whole process, during the windows)

| run | cores busy | GHz | IPC | retiring % | frontend % | bad spec % | backend: cpu % | backend: memory % | FP G-ops/s (all) | FP G-ops/s (mul-add) |
|---|---|---|---|---|---|---|---|---|---|---|
| base_stock_pp256_ub512 | 26.8 | 2.72 | 1.41 | 16.6 | 8.1 | 0.8 | 13.7 | 60.8 | 1319 | 1296 |
| base_stock_pp8192_ub512 | 26.4 | 2.72 | 1.80 | 20.5 | 8.6 | 0.9 | 16.1 | 53.8 | 1633 | 1608 |
| base_zendnn_pp256_ub512 | 27.5 | 2.72 | 1.49 | 17.6 | 8.9 | 0.7 | 13.6 | 59.3 | 1404 | 1384 |
| base_zendnn_pp8192_ub512 | 25.7 | 2.72 | 1.90 | 21.8 | 8.7 | 0.8 | 15.9 | 52.7 | 1659 | 1636 |
| vllm_pp256 | 31.9 | 2.72 | 1.07 | 13.7 | 11.2 | 0.2 | 4.6 | 70.3 | 2358 | 2355 |
| vllm_pp8192 | 24.9 | 2.72 | 2.20 | 29.5 | 8.1 | 0.3 | 28.5 | 33.6 | 3628 | 3621 |

Top-down columns are % of dispatch slots (PipelineL1/L2). FP columns: retired FP operations per second (fp_ret_sse_avx_ops.all / .mac_flops), across the whole process. Their bf16 and packed-fp32 umasks read 0 on this CPU, so the split by data type is not available.

## Where L1 data-cache fills come from (% of fills) and estimated DRAM read traffic

| run | L2 % | L3 same CCX % | other CCX % | DRAM this socket % | DRAM other socket % | L1 fills from DRAM GB/s | L2 prefetch misses to DRAM GB/s | DRAM read est. GB/s |
|---|---|---|---|---|---|---|---|---|
| base_stock_pp256_ub512 | 87.0 | 6.2 | 0.9 | 6.0 | 0.0 | 43.6 | 48.7 | 92 |
| base_stock_pp8192_ub512 | 89.8 | 6.0 | 0.7 | 3.5 | 0.0 | 33.0 | 40.3 | 73 |
| base_zendnn_pp256_ub512 | 86.9 | 6.1 | 0.8 | 6.3 | 0.0 | 41.4 | 48.1 | 90 |
| base_zendnn_pp8192_ub512 | 90.4 | 5.5 | 0.6 | 3.5 | 0.0 | 29.4 | 37.5 | 67 |
| vllm_pp256 | 86.6 | 2.2 | 0.4 | 10.8 | 0.0 | 53.9 | 62.8 | 117 |
| vllm_pp8192 | 96.7 | 1.4 | 0.2 | 1.7 | 0.0 | 11.8 | 30.5 | 42 |

DRAM near/far = this socket / the other socket. The estimate counts 64-byte lines from L1 fills that came from DRAM plus L2 hardware prefetches that missed L3; it excludes writes. The pods expose no uncore (data-fabric) counters to measure DRAM bandwidth directly. NUMA node 1 has 3 of the socket's 12 DDR5 channels (NPS4), roughly 140 GB/s peak.

## base_stock_pp256_ub512

### Where the cycles go (flat profile, all threads)

| category | % of cycles |
|---|---|
| ggml: bf16 dot product (MUL_MAT_ID experts on CPU) | 52.7 |
| ggml: tinyBLAS/llamafile GEMM (dense matmul on CPU) | 16.1 |
| OpenMP runtime (spin-wait at barriers, fork/join) | 7.8 |
| ggml: graph loop (ggml_barrier, inlined small ops) | 5.5 |
| ggml: other ops (norms, elementwise, softmax, ...) | 5.3 |
| ggml: GDN recurrence + conv1d | 4.3 |
| ggml: f32/f16 dot product | 2.1 |
| ggml: type conversion | 0.8 |
| ggml: flash attention | 0.8 |
| ggml: CONCAT (GDN conv state) | 0.7 |
| ggml-base: other | 0.2 |
| kernel | 0.1 |
| libc (memcpy / memset) | 0.1 |

### Top symbols

| % | object | symbol |
|---|---|---|
| 52.68 | libggml-cpu.so.0.23.0 | ggml_vec_dot_bf16 |
| 14.56 | libggml-cpu.so.0.23.0 | void (anonymous namespace)::tinyBLAS<32, float __vector(16), short __vector(32), ggml_bf16 |
| 7.17 | libomp.so.5 | 0x000000000007df00 |
| 5.50 | libggml-cpu.so.0.23.0 | ggml_graph_compute_thread.isra.0 |
| 2.76 | libggml-cpu.so.0.23.0 | ggml_compute_forward_gated_delta_net |
| 2.07 | libggml-cpu.so.0.23.0 | ggml_vec_dot_f32 |
| 1.57 | libggml-cpu.so.0.23.0 | ggml_compute_forward_ssm_conv |
| 1.10 | libggml-cpu.so.0.23.0 | ggml_compute_forward_mul |
| 1.02 | libggml-cpu.so.0.23.0 | void (anonymous namespace)::tinyBLAS<16, float __vector(16), float __vector(16), float, fl |
| 0.95 | libggml-cpu.so.0.23.0 | ggml_compute_forward_rms_norm_mul_fused |
| 0.84 | libggml-cpu.so.0.23.0 | ggml_cpu_fp32_to_bf16 |
| 0.75 | libggml-cpu.so.0.23.0 | ggml_compute_forward_flash_attn_ext_tiled(ggml_compute_params const*, ggml_tensor*, int, i |
| 0.72 | libggml-cpu.so.0.23.0 | ggml_compute_forward_concat |
| 0.68 | libggml-cpu.so.0.23.0 | ggml_vec_silu_f32 |
| 0.67 | libggml-cpu.so.0.23.0 | ggml_vec_swiglu_f32 |
| 0.48 | libggml-cpu.so.0.23.0 | ggml_compute_forward_mul_mat |
| 0.44 | libggml-cpu.so.0.23.0 | void (anonymous namespace)::tinyBLAS<32, float __vector(16), short __vector(32), ggml_bf16 |
| 0.44 | libggml-cpu.so.0.23.0 | ggml_compute_forward_add_non_quantized |
| 0.43 | libomp.so.5 | 0x00000000000e1140 |
| 0.25 | libggml-cpu.so.0.23.0 | ggml_compute_forward_rms_norm |

### Per ggml op (cpu-clock stacks: 198428 samples, 32 threads, main thread 219780)

| op (main thread's stack) | % of wall time | threads doing work (of 32) | % of all CPU samples: work | % of all CPU samples: spinning |
|---|---|---|---|---|
| [leaf] ggml: bf16 dot product (MUL_MAT_ID experts on CPU) | 44.6 | 31.8 | 53.5 | 0.0 |
| concat | 19.7 | 2.8 | 0.7 | 0.0 |
| [leaf] ggml: tinyBLAS/llamafile GEMM (dense matmul on CPU) | 13.1 | 27.8 | 15.9 | 0.0 |
| [leaf] ggml: graph loop (ggml_barrier, inlined small ops) | 3.1 | 31.9 | 3.2 | 0.0 |
| (ggml barrier: waiting for the slowest thread) | 2.9 | 25.2 | 0.0 | 8.2 |
| gated_delta_net | 2.6 | 26.9 | 3.2 | 0.0 |
| [leaf] libc (memcpy / memset) | 2.3 | 29.6 | 3.2 | 0.0 |
| [leaf] ggml: f32/f16 dot product | 1.5 | 27.8 | 1.8 | 0.0 |
| [leaf] ggml: type conversion | 1.4 | 30.3 | 1.6 | 0.0 |
| mul_mat | 1.2 | 27.5 | 0.1 | 0.8 |
| ssm_conv | 1.1 | 16.1 | 1.6 | 0.0 |
| mul | 1.0 | 29.9 | 1.1 | 0.0 |
| rms_norm_mul | 0.9 | 28.0 | 0.9 | 0.0 |
| flash_attn_ext | 0.8 | 25.5 | 0.9 | 0.0 |
| unary | 0.7 | 25.2 | 0.7 | 0.0 |
| [leaf] ggml: other ops (norms, elementwise, softmax, ...) | 0.6 | 29.3 | 0.7 | 0.0 |
| get_rows | 0.6 | 12.7 | 0.0 | 0.0 |
| glu | 0.4 | 31.2 | 0.5 | 0.0 |
| rms_norm | 0.4 | 22.7 | 0.3 | 0.0 |
| add | 0.3 | 29.4 | 0.4 | 0.0 |
| scale | 0.2 | 18.1 | 0.2 | 0.0 |
| argsort | 0.2 | 29.8 | 0.2 | 0.0 |
| [leaf] other | 0.1 | 22.1 | 0.1 | 0.0 |
| [leaf] ggml-base: other | 0.1 | 24.6 | 0.0 | 0.0 |
| sigmoid | 0.0 | 27.5 | 0.1 | 0.0 |
| clamp | 0.0 | 42.0 | 0.0 | 0.0 |
| soft_max | 0.0 | 32.0 | 0.0 | 0.0 |
| sum_rows | 0.0 | 24.0 | 0.0 | 0.0 |

'% of wall time' is the main thread's samples. 'threads doing work' counts, per main-thread sample in that op, the samples of all threads within 6 ms that were not spinning in libomp.

### Threads: 32, work share min/median/max = 90/91/96% (the rest is spinning in libomp)

## base_stock_pp8192_ub512

### Where the cycles go (flat profile, all threads)

| category | % of cycles |
|---|---|
| ggml: bf16 dot product (MUL_MAT_ID experts on CPU) | 43.0 |
| ggml: tinyBLAS/llamafile GEMM (dense matmul on CPU) | 15.7 |
| ggml: flash attention | 13.1 |
| OpenMP runtime (spin-wait at barriers, fork/join) | 5.8 |
| ggml: other ops (norms, elementwise, softmax, ...) | 5.7 |
| ggml: graph loop (ggml_barrier, inlined small ops) | 4.8 |
| ggml: GDN recurrence + conv1d | 4.7 |
| ggml: f32/f16 dot product | 2.3 |
| ggml: type conversion | 1.3 |
| ggml: CONCAT (GDN conv state) | 0.7 |
| ggml-base: other | 0.2 |

### Top symbols

| % | object | symbol |
|---|---|---|
| 42.97 | libggml-cpu.so.0.23.0 | ggml_vec_dot_bf16 |
| 14.08 | libggml-cpu.so.0.23.0 | void (anonymous namespace)::tinyBLAS<32, float __vector(16), short __vector(32), ggml_bf16 |
| 13.13 | libggml-cpu.so.0.23.0 | ggml_compute_forward_flash_attn_ext_tiled(ggml_compute_params const*, ggml_tensor*, int, i |
| 5.35 | libomp.so.5 | 0x000000000007df00 |
| 4.79 | libggml-cpu.so.0.23.0 | ggml_graph_compute_thread.isra.0 |
| 3.00 | libggml-cpu.so.0.23.0 | ggml_compute_forward_gated_delta_net |
| 2.25 | libggml-cpu.so.0.23.0 | ggml_vec_dot_f32 |
| 1.65 | libggml-cpu.so.0.23.0 | ggml_compute_forward_ssm_conv |
| 1.08 | libggml-cpu.so.0.23.0 | ggml_compute_forward_mul |
| 1.02 | libggml-cpu.so.0.23.0 | ggml_compute_forward_rms_norm_mul_fused |
| 1.02 | libggml-cpu.so.0.23.0 | void (anonymous namespace)::tinyBLAS<16, float __vector(16), float __vector(16), float, fl |
| 0.87 | libggml-cpu.so.0.23.0 | ggml_cpu_fp32_to_bf16 |
| 0.74 | libggml-cpu.so.0.23.0 | ggml_compute_forward_concat |
| 0.72 | libggml-cpu.so.0.23.0 | ggml_vec_swiglu_f32 |
| 0.71 | libggml-cpu.so.0.23.0 | ggml_vec_silu_f32 |
| 0.48 | libggml-cpu.so.0.23.0 | ggml_compute_forward_add_non_quantized |
| 0.48 | libggml-cpu.so.0.23.0 | void (anonymous namespace)::tinyBLAS<32, float __vector(16), short __vector(32), ggml_bf16 |
| 0.46 | libggml-cpu.so.0.23.0 | ggml_compute_forward_mul_mat |
| 0.35 | libggml-cpu.so.0.23.0 | ggml_cpu_fp16_to_fp32 |
| 0.34 | libomp.so.5 | 0x00000000000e1140 |

### Per ggml op (cpu-clock stacks: 192626 samples, 32 threads, main thread 218652)

| op (main thread's stack) | % of wall time | threads doing work (of 32) | % of all CPU samples: work | % of all CPU samples: spinning |
|---|---|---|---|---|
| [leaf] ggml: bf16 dot product (MUL_MAT_ID experts on CPU) | 37.8 | 32.2 | 46.3 | 0.0 |
| concat | 21.2 | 2.1 | 0.8 | 0.0 |
| [leaf] ggml: tinyBLAS/llamafile GEMM (dense matmul on CPU) | 14.1 | 29.4 | 17.2 | 0.0 |
| flash_attn_ext | 4.7 | 31.1 | 6.0 | 0.0 |
| (ggml barrier: waiting for the slowest thread) | 3.3 | 22.7 | 0.0 | 6.6 |
| gated_delta_net | 2.9 | 28.8 | 3.8 | 0.0 |
| [leaf] ggml: graph loop (ggml_barrier, inlined small ops) | 2.7 | 32.1 | 3.3 | 0.0 |
| [leaf] libc (memcpy / memset) | 2.6 | 29.4 | 3.2 | 0.0 |
| [leaf] ggml: type conversion | 1.8 | 29.5 | 2.0 | 0.0 |
| [leaf] ggml: f32/f16 dot product | 1.8 | 29.3 | 2.2 | 0.0 |
| ssm_conv | 1.4 | 20.0 | 1.7 | 0.0 |
| rms_norm_mul | 0.9 | 28.7 | 1.1 | 0.0 |
| mul | 0.9 | 29.6 | 1.1 | 0.0 |
| unary | 0.7 | 27.3 | 0.8 | 0.0 |
| glu | 0.6 | 30.3 | 0.7 | 0.0 |
| rms_norm | 0.5 | 24.3 | 0.4 | 0.0 |
| mul_mat | 0.4 | 27.7 | 0.1 | 0.7 |
| [leaf] ggml: other ops (norms, elementwise, softmax, ...) | 0.4 | 26.5 | 0.8 | 0.0 |
| add | 0.4 | 29.4 | 0.5 | 0.0 |
| get_rows | 0.3 | 16.9 | 0.0 | 0.0 |
| [leaf] other | 0.2 | 16.7 | 0.1 | 0.0 |
| argsort | 0.1 | 28.9 | 0.2 | 0.0 |
| [leaf] ggml-base: other | 0.1 | 17.3 | 0.1 | 0.0 |
| scale | 0.1 | 30.0 | 0.2 | 0.0 |
| sigmoid | 0.0 | 20.3 | 0.0 | 0.0 |
| soft_max | 0.0 | 31.3 | 0.0 | 0.0 |
| clamp | 0.0 | 49.0 | 0.0 | 0.0 |
| sum_rows | 0.0 | 29.0 | 0.0 | 0.0 |

'% of wall time' is the main thread's samples. 'threads doing work' counts, per main-thread sample in that op, the samples of all threads within 6 ms that were not spinning in libomp.

### Threads: 32, work share min/median/max = 91/93/96% (the rest is spinning in libomp)

## base_zendnn_pp256_ub512

### Where the cycles go (flat profile, all threads)

| category | % of cycles |
|---|---|
| ggml: bf16 dot product (MUL_MAT_ID experts on CPU) | 55.0 |
| ZenDNN/AOCL-DLP GEMM kernels (incl. JIT code) | 10.8 |
| OpenMP runtime (spin-wait at barriers, fork/join) | 9.1 |
| ggml: graph loop (ggml_barrier, inlined small ops) | 5.6 |
| ggml: other ops (norms, elementwise, softmax, ...) | 5.4 |
| ggml: GDN recurrence + conv1d | 4.4 |
| ggml: f32/f16 dot product | 2.1 |
| ggml: type conversion | 1.3 |
| ggml: flash attention | 0.8 |
| ggml: CONCAT (GDN conv state) | 0.7 |
| ggml: tinyBLAS/llamafile GEMM (dense matmul on CPU) | 0.1 |
| ggml-base: other | 0.1 |
| kernel | 0.1 |

### Top symbols

| % | object | symbol |
|---|---|---|
| 55.05 | libggml-cpu.so.0.23.0 | ggml_vec_dot_bf16 |
| 8.20 | libomp.so.5 | 0x000000000007df00 |
| 5.64 | libggml-cpu.so.0.23.0 | ggml_graph_compute_thread.isra.0 |
| 2.86 | libggml-cpu.so.0.23.0 | ggml_compute_forward_gated_delta_net |
| 2.08 | libggml-cpu.so.0.23.0 | ggml_vec_dot_f32 |
| 1.56 | libggml-cpu.so.0.23.0 | ggml_compute_forward_ssm_conv |
| 1.21 | libggml-cpu.so.0.23.0 | ggml_compute_forward_mul |
| 0.99 | libggml-cpu.so.0.23.0 | ggml_compute_forward_rms_norm_mul_fused |
| 0.91 | libggml-base.so.0.23.0 | ggml_fp32_to_bf16_row_ref |
| 0.78 | libggml-cpu.so.0.23.0 | ggml_compute_forward_flash_attn_ext_tiled(ggml_compute_params const*, ggml_tensor*, int, i |
| 0.78 | libggml-cpu.so.0.23.0 | ggml_vec_silu_f32 |
| 0.75 | libggml-cpu.so.0.23.0 | ggml_vec_swiglu_f32 |
| 0.66 | libggml-cpu.so.0.23.0 | ggml_compute_forward_concat |
| 0.51 | libggml-cpu.so.0.23.0 | ggml_compute_forward_add_non_quantized |
| 0.50 | libomp.so.5 | 0x00000000000e1140 |
| 0.37 | libggml-cpu.so.0.23.0 | ggml_cpu_fp32_to_bf16 |
| 0.34 | [JIT] tid 219222 | 0x00007317e4207165 |
| 0.32 | [JIT] tid 219222 | 0x00007317e4207144 |
| 0.30 | [JIT] tid 219222 | 0x00007317e42071b3 |
| 0.29 | [JIT] tid 219222 | 0x00007317e42071a7 |

### Per ggml op (cpu-clock stacks: 203547 samples, 32 threads, main thread 219222)

| op (main thread's stack) | % of wall time | threads doing work (of 32) | % of all CPU samples: work | % of all CPU samples: spinning |
|---|---|---|---|---|
| [leaf] ggml: bf16 dot product (MUL_MAT_ID experts on CPU) | 47.8 | 31.9 | 55.6 | 0.0 |
| concat | 17.8 | 3.1 | 0.6 | 0.0 |
| ZenDNN matmul (AOCL-DLP GEMM kernels) | 11.4 | 26.0 | 12.0 | 0.0 |
| (ggml barrier: waiting for the slowest thread) | 3.3 | 24.2 | 0.0 | 8.5 |
| gated_delta_net | 3.0 | 27.5 | 3.4 | 0.0 |
| [leaf] ggml: graph loop (ggml_barrier, inlined small ops) | 2.8 | 31.7 | 3.3 | 0.0 |
| [leaf] libc (memcpy / memset) | 2.7 | 29.9 | 3.2 | 0.0 |
| [leaf] ggml: f32/f16 dot product | 1.6 | 27.8 | 1.8 | 0.0 |
| [leaf] ggml: type conversion | 1.4 | 30.0 | 1.6 | 0.0 |
| ssm_conv | 1.3 | 15.0 | 1.6 | 0.0 |
| mul | 1.0 | 29.0 | 1.1 | 0.0 |
| flash_attn_ext | 0.9 | 24.4 | 1.0 | 0.0 |
| unary | 0.8 | 23.0 | 0.8 | 0.0 |
| rms_norm_mul | 0.8 | 28.5 | 1.0 | 0.0 |
| glu | 0.6 | 30.5 | 0.7 | 0.0 |
| [leaf] ggml: other ops (norms, elementwise, softmax, ...) | 0.6 | 27.8 | 0.6 | 0.0 |
| get_rows | 0.6 | 11.6 | 0.0 | 0.0 |
| add | 0.4 | 29.8 | 0.4 | 0.0 |
| rms_norm | 0.4 | 23.3 | 0.3 | 0.0 |
| scale | 0.2 | 15.3 | 0.2 | 0.0 |
| mul_mat | 0.1 | 27.5 | 0.0 | 0.2 |
| [leaf] other | 0.1 | 26.2 | 0.1 | 0.0 |
| (OpenMP: idle / fork-join) | 0.1 | 24.6 | 0.0 | 1.5 |
| [leaf] ggml: tinyBLAS/llamafile GEMM (dense matmul on CPU) | 0.1 | 27.0 | 0.1 | 0.0 |
| argsort | 0.1 | 31.2 | 0.2 | 0.0 |
| soft_max | 0.0 | 31.0 | 0.0 | 0.0 |
| [leaf] ggml-base: other | 0.0 | 31.7 | 0.0 | 0.0 |
| sigmoid | 0.0 | 27.0 | 0.1 | 0.0 |
| clamp | 0.0 | 29.0 | 0.0 | 0.0 |

'% of wall time' is the main thread's samples. 'threads doing work' counts, per main-thread sample in that op, the samples of all threads within 6 ms that were not spinning in libomp.

### Threads: 32, work share min/median/max = 89/89/95% (the rest is spinning in libomp)

## base_zendnn_pp8192_ub512

### Where the cycles go (flat profile, all threads)

| category | % of cycles |
|---|---|
| ggml: bf16 dot product (MUL_MAT_ID experts on CPU) | 45.0 |
| ggml: flash attention | 14.0 |
| ZenDNN/AOCL-DLP GEMM kernels (incl. JIT code) | 10.1 |
| OpenMP runtime (spin-wait at barriers, fork/join) | 6.5 |
| ggml: other ops (norms, elementwise, softmax, ...) | 5.6 |
| ggml: GDN recurrence + conv1d | 4.9 |
| ggml: graph loop (ggml_barrier, inlined small ops) | 4.8 |
| ggml: f32/f16 dot product | 2.4 |
| ggml: type conversion | 1.8 |
| ggml: CONCAT (GDN conv state) | 0.8 |
| ggml: tinyBLAS/llamafile GEMM (dense matmul on CPU) | 0.1 |
| libc (memcpy / memset) | 0.1 |
| ggml-base: other | 0.1 |

### Top symbols

| % | object | symbol |
|---|---|---|
| 45.02 | libggml-cpu.so.0.23.0 | ggml_vec_dot_bf16 |
| 14.04 | libggml-cpu.so.0.23.0 | ggml_compute_forward_flash_attn_ext_tiled(ggml_compute_params const*, ggml_tensor*, int, i |
| 6.00 | libomp.so.5 | 0x000000000007df00 |
| 4.84 | libggml-cpu.so.0.23.0 | ggml_graph_compute_thread.isra.0 |
| 3.16 | libggml-cpu.so.0.23.0 | ggml_compute_forward_gated_delta_net |
| 2.30 | libggml-cpu.so.0.23.0 | ggml_vec_dot_f32 |
| 1.75 | libggml-cpu.so.0.23.0 | ggml_compute_forward_ssm_conv |
| 1.13 | libggml-cpu.so.0.23.0 | ggml_compute_forward_mul |
| 1.04 | libggml-cpu.so.0.23.0 | ggml_compute_forward_rms_norm_mul_fused |
| 0.97 | libggml-base.so.0.23.0 | ggml_fp32_to_bf16_row_ref |
| 0.88 | libggml-cpu.so.0.23.0 | ggml_vec_silu_f32 |
| 0.82 | libggml-cpu.so.0.23.0 | ggml_compute_forward_concat |
| 0.68 | libggml-cpu.so.0.23.0 | ggml_vec_swiglu_f32 |
| 0.49 | libggml-cpu.so.0.23.0 | ggml_compute_forward_add_non_quantized |
| 0.40 | libggml-cpu.so.0.23.0 | ggml_cpu_fp16_to_fp32 |
| 0.39 | libggml-cpu.so.0.23.0 | ggml_cpu_fp32_to_bf16 |
| 0.36 | libomp.so.5 | 0x00000000000e1140 |
| 0.35 | [JIT] tid 218064 | 0x00007142c4b7a165 |
| 0.29 | [JIT] tid 218064 | 0x00007142c4b7a16b |
| 0.28 | [JIT] tid 218064 | 0x00007142c4b7a1b3 |

### Per ggml op (cpu-clock stacks: 188474 samples, 32 threads, main thread 218064)

| op (main thread's stack) | % of wall time | threads doing work (of 32) | % of all CPU samples: work | % of all CPU samples: spinning |
|---|---|---|---|---|
| [leaf] ggml: bf16 dot product (MUL_MAT_ID experts on CPU) | 38.5 | 32.1 | 48.5 | 0.0 |
| concat | 23.1 | 2.0 | 0.9 | 0.0 |
| ZenDNN matmul (AOCL-DLP GEMM kernels) | 10.5 | 27.3 | 12.1 | 0.0 |
| flash_attn_ext | 5.5 | 31.3 | 7.1 | 0.0 |
| (ggml barrier: waiting for the slowest thread) | 3.3 | 23.1 | 0.0 | 7.0 |
| gated_delta_net | 3.0 | 29.2 | 4.0 | 0.0 |
| [leaf] ggml: graph loop (ggml_barrier, inlined small ops) | 3.0 | 32.0 | 3.5 | 0.0 |
| [leaf] libc (memcpy / memset) | 2.5 | 29.8 | 3.2 | 0.0 |
| [leaf] ggml: f32/f16 dot product | 1.8 | 29.8 | 2.2 | 0.0 |
| [leaf] ggml: type conversion | 1.6 | 28.1 | 1.9 | 0.0 |
| ssm_conv | 1.3 | 19.6 | 1.9 | 0.0 |
| rms_norm_mul | 1.1 | 25.7 | 1.3 | 0.0 |
| mul | 1.0 | 28.2 | 1.1 | 0.0 |
| unary | 0.8 | 26.4 | 0.8 | 0.0 |
| glu | 0.5 | 30.4 | 0.7 | 0.0 |
| [leaf] ggml: other ops (norms, elementwise, softmax, ...) | 0.5 | 28.8 | 0.6 | 0.0 |
| rms_norm | 0.4 | 25.7 | 0.3 | 0.0 |
| add | 0.4 | 27.9 | 0.6 | 0.0 |
| get_rows | 0.3 | 18.4 | 0.0 | 0.0 |
| argsort | 0.2 | 28.7 | 0.2 | 0.0 |
| scale | 0.1 | 22.1 | 0.2 | 0.0 |
| [leaf] ggml-base: other | 0.1 | 18.7 | 0.0 | 0.0 |
| [leaf] other | 0.1 | 15.9 | 0.1 | 0.0 |
| (OpenMP: idle / fork-join) | 0.1 | 27.8 | 0.0 | 1.2 |
| [leaf] ggml: tinyBLAS/llamafile GEMM (dense matmul on CPU) | 0.1 | 28.0 | 0.1 | 0.0 |
| mul_mat | 0.1 | 26.3 | 0.0 | 0.1 |
| sigmoid | 0.0 | 16.5 | 0.1 | 0.0 |
| soft_max | 0.0 | 15.0 | 0.0 | 0.0 |
| [leaf] kernel | 0.0 | 1.0 | 0.0 | 0.0 |

'% of wall time' is the main thread's samples. 'threads doing work' counts, per main-thread sample in that op, the samples of all threads within 6 ms that were not spinning in libomp.

### Threads: 32, work share min/median/max = 90/91/96% (the rest is spinning in libomp)

## vllm_pp256

### Where the cycles go (flat profile, all threads)

| category | % of cycles |
|---|---|
| ZenDNN/AOCL-DLP GEMM kernels (incl. JIT code) | 65.0 |
| OpenMP runtime (spin-wait at barriers, fork/join) | 20.7 |
| PyTorch runtime / aten ops | 1.5 |
| other | 1.3 |
| vLLM C++: attention | 1.2 |
| vLLM C++: other kernels | 0.9 |
| zentorch/ZenDNN: fused MoE (incl. its GEMMs) | 0.7 |
| kernel | 0.5 |
| zentorch/ZenDNN GEMM kernels | 0.5 |
| libc (memcpy / memset) | 0.3 |
| vLLM C++: GDN / conv1d | 0.1 |

### Top symbols

| % | object | symbol |
|---|---|---|
| 20.48 | libiomp5.so | kmp_flag_64<false, true>::wait(kmp_info*, int, void*) |
| 1.19 | [JIT] tid 242308 | 0x00007fe2544d8148 |
| 1.14 | libtorch_cpu.so | void c10::function_ref<void (char**, long const*, long, long)>::callback_fn<at::native::AV |
| 1.12 | [JIT] tid 242308 | 0x00007fe2544d8169 |
| 1.03 | [JIT] tid 242308 | 0x00007fe2544d8121 |
| 1.01 | [JIT] tid 242308 | 0x00007fe2544d8133 |
| 1.00 | [JIT] tid 242308 | 0x00007fe2544d812d |
| 0.99 | [JIT] tid 242308 | 0x00007fe2544d8139 |
| 0.98 | [JIT] tid 242308 | 0x00007fe2544d8127 |
| 0.92 | [JIT] tid 242308 | 0x00007fe2544d8142 |
| 0.90 | [JIT] tid 242308 | 0x00007fe2544d8175 |
| 0.90 | [JIT] tid 242308 | 0x00007fe2544d813c |
| 0.90 | [JIT] tid 242308 | 0x00007fe2544d819c |
| 0.89 | [JIT] tid 242308 | 0x00007fe2544d816f |
| 0.88 | [JIT] tid 242308 | 0x00007fe2544d817b |
| 0.88 | [JIT] tid 242308 | 0x00007fe2544d8196 |
| 0.88 | [JIT] tid 242308 | 0x00007fe2544d8190 |
| 0.87 | [JIT] tid 242308 | 0x00007fe2544d80f4 |
| 0.84 | [JIT] tid 242308 | 0x00007fe2544d817e |
| 0.84 | [JIT] tid 242308 | 0x00007fe2544d81b7 |

## vllm_pp8192

### Where the cycles go (flat profile, all threads)

| category | % of cycles |
|---|---|
| ZenDNN/AOCL-DLP GEMM kernels (incl. JIT code) | 37.8 |
| vLLM C++: attention | 32.1 |
| OpenMP runtime (spin-wait at barriers, fork/join) | 13.8 |
| PyTorch runtime / aten ops | 3.7 |
| other | 2.2 |
| vLLM C++: other kernels | 1.6 |
| zentorch/ZenDNN: fused MoE (incl. its GEMMs) | 1.1 |
| libc (memcpy / memset) | 0.3 |
| vLLM C++: GDN / conv1d | 0.3 |
| zentorch/ZenDNN GEMM kernels | 0.3 |
| kernel | 0.1 |

### Top symbols

| % | object | symbol |
|---|---|---|
| 31.47 | _C.abi3.so | void cpu_attention::(anonymous namespace)::TileGemm82<c10::BFloat16>::gemm_micro<8>(float* |
| 13.69 | libiomp5.so | kmp_flag_64<false, true>::wait(kmp_info*, int, void*) |
| 3.17 | libtorch_cpu.so | void c10::function_ref<void (char**, long const*, long, long)>::callback_fn<at::native::AV |
| 1.28 | [JIT] tid 240911 | 0x000071dc5195a169 |
| 1.07 | libzentorch.so | zendnnl::lowoha::matmul::(anonymous namespace)::moe_weighted_reduce_avx512_bf16(zendnnl::l |
| 1.04 | [JIT] tid 240911 | 0x000071dc5195a1b7 |
| 1.04 | [JIT] tid 240911 | 0x000071dc5195a1b1 |
| 1.04 | [JIT] tid 240911 | 0x000071dc5195a148 |
| 1.03 | [JIT] tid 240911 | 0x000071dc5195a16f |
| 1.01 | [JIT] tid 240911 | 0x000071dc5195a1bd |
| 1.00 | [JIT] tid 240911 | 0x000071dc5195a1ab |
| 0.99 | [JIT] tid 240911 | 0x000071dc5195a19f |
| 0.97 | [JIT] tid 240911 | 0x000071dc5195a17b |
| 0.97 | [JIT] tid 240911 | 0x000071dc5195a175 |
| 0.95 | [JIT] tid 240911 | 0x000071dc5195a19c |
| 0.94 | [JIT] tid 240911 | 0x000071dc5195a1a5 |
| 0.91 | [JIT] tid 240911 | 0x000071dc5195a18a |
| 0.91 | cjkl35cnfyz3knsmpysdahluwuqbz3vi44g32vo2k5eb3lpfcgvr.main.so | kernel._omp_fn.0 |
| 0.91 | [JIT] tid 240911 | 0x000071dc5195a196 |
| 0.90 | [JIT] tid 240911 | 0x000071dc5195a17e |

## vLLM per-op split: vllm_profile_pp256 (torch profiler, one generate call; self CPU total 581.212ms)

| bucket | self CPU ms |
|---|---|
| MoE experts (zentorch fused MoE incl. its copies) | 351 |
| linear / GEMM (zentorch_linear) | 107 |
| other ops (top-50 rows) | 41 |
| (compiled-graph / step wrappers: self time) | 39 |
| GDN / linear attention (chunk rule, conv1d, gating) | 31 |
| attention | 9 |
| concatenation (aten::cat) | 2 |

| op | self CPU % | self CPU | calls |
|---|---|---|---|
| zentorch::zentorch_fused_moe | 58.08% | 337.579ms | 40 |
| zentorch_linear_unary | 16.85% | 97.949ms | 270 |
| execute_context_1(256)_generation_0(0) | 3.36% | 19.543ms | 1 |
| ## Call CompiledFxGraph None ## | 3.35% | 19.446ms | 1 |
| _C::chunk_gated_delta_rule_cpu | 3.05% | 17.742ms | 30 |
| aten::topk | 2.88% | 16.723ms | 40 |
| vllm::cpu_gdn_attention_core | 1.67% | 9.683ms | 30 |
| aten::copy_ | 1.60% | 9.302ms | 314 |
| zentorch::zentorch_linear_unary | 1.54% | 8.947ms | 271 |
| zentorch::fused_moe::pass2_parallel_memcpy | 1.48% | 8.580ms | 40 |
| _C::cpu_attn_reshape_and_cache | 0.72% | 4.198ms | 10 |
| _C::cpu_attention_with_kv_cache | 0.72% | 4.196ms | 10 |
| zentorch::fused_moe::pass1_active_set_build | 0.54% | 3.136ms | 40 |
| _C::causal_conv1d_fwd_cpu | 0.51% | 2.980ms | 30 |
| aten::cat | 0.40% | 2.310ms | 33 |

## vLLM per-op split: vllm_profile_pp8192 (torch profiler, one generate call; self CPU total 12.969s)

| bucket | self CPU ms |
|---|---|
| MoE experts (zentorch fused MoE incl. its copies) | 6139 |
| attention | 3196 |
| linear / GEMM (zentorch_linear) | 2377 |
| GDN / linear attention (chunk rule, conv1d, gating) | 475 |
| (compiled-graph / step wrappers: self time) | 344 |
| other ops (top-50 rows) | 322 |
| concatenation (aten::cat) | 115 |

| op | self CPU % | self CPU | calls |
|---|---|---|---|
| zentorch::zentorch_fused_moe | 44.47% | 5.768s | 80 |
| _C::cpu_attention_with_kv_cache | 24.60% | 3.190s | 20 |
| zentorch_linear_unary | 18.18% | 2.357s | 540 |
| _C::chunk_gated_delta_rule_cpu | 2.76% | 358.059ms | 60 |
| ## Call CompiledFxGraph None ## | 2.53% | 327.927ms | 2 |
| zentorch::fused_moe::pass2_parallel_memcpy | 2.24% | 290.838ms | 80 |
| aten::copy_ | 1.92% | 248.579ms | 597 |
| aten::cat | 0.89% | 115.081ms | 66 |
| _C::causal_conv1d_fwd_cpu | 0.68% | 88.424ms | 60 |
| zentorch::fused_moe::pass1_active_set_build | 0.55% | 71.327ms | 80 |
| aten::topk | 0.30% | 38.957ms | 80 |
| vllm::cpu_gdn_attention_core | 0.20% | 26.293ms | 60 |
| zentorch::zentorch_linear_unary | 0.16% | 20.367ms | 542 |
| execute_context_1(4096)_generation_0(0) | 0.11% | 14.741ms | 2 |
| zentorch::fused_moe::scratchpad_allocation | 0.04% | 5.586ms | 80 |

