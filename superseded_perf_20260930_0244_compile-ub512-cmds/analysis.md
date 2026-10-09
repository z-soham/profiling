# perf analysis: perf_qwen36_bf16_20260930_0244

Qwen3.6-35B-A3B BF16 prefill with the optimised commands of turin-optimised-commands-2026-09-30.md (boost off): llama.cpp setup B on sacsharm's existing stock and ZenDNN builds (weights loaded with --load-mode dio into local memory), vLLM setup C. 32 cores / 1 NUMA node. perf attached after load and warm-up (vLLM: to its worker process); one measurement per window. See perf_llama.sh.

## Runs

| run | prefill t/s while profiled (median pass) | passes | start-up s | resident memory per NUMA node |
|---|---|---|---|---|
| base_stock_pp256_ub512 | 259.8 | 110 | 84 | N1=65.5GB N5=0.0GB N0=0.0GB |
| base_stock_pp8192_ub4096 | 235.6 | 2 | 85 | N1=66.3GB N5=0.0GB N0=0.0GB |
| base_stock_pp8192_ub512 | 276.3 | 3 | 86 | N1=65.7GB N5=0.0GB N0=0.0GB |
| base_zendnn_fb0_pp8192_ub512 | 287.1 | 3 | 85 | N1=181.0GB N5=0.0GB |
| base_zendnn_pp256_ub512 | 273.5 | 116 | 86 | N1=68.1GB N5=0.0GB N0=0.0GB |
| base_zendnn_pp8192_ub4096 | 243.3 | 2 | 86 | N1=67.8GB N5=0.0GB N0=0.0GB |
| base_zendnn_pp8192_ub512 | 286.8 | 3 | 84 | N1=68.3GB N5=0.0GB |

## Counters (whole process, during the windows)

| run | cores busy | GHz | IPC | retiring % | frontend % | bad spec % | backend: cpu % | backend: memory % | FP G-ops/s (all) | FP G-ops/s (mul-add) |
|---|---|---|---|---|---|---|---|---|---|---|
| base_stock_pp256_ub512 | 32.0 | 2.72 | 1.35 | 16.5 | 13.2 | 0.7 | 12.8 | 56.7 | 1337 | 1315 |
| base_stock_pp8192_ub4096 | 29.7 | 2.72 | 1.68 | 21.3 | 19.6 | 0.7 | 14.8 | 43.5 | 1345 | 1324 |
| base_stock_pp8192_ub512 | 32.0 | 2.72 | 1.66 | 19.5 | 14.8 | 0.7 | 14.5 | 50.5 | 1627 | 1602 |
| base_zendnn_fb0_pp8192_ub512 | 32.0 | 2.72 | 1.51 | 19.8 | 22.0 | 0.4 | 7.7 | 50.1 | 1693 | 1680 |
| base_zendnn_pp256_ub512 | 32.0 | 2.72 | 1.43 | 17.4 | 13.5 | 0.6 | 12.8 | 55.6 | 1403 | 1383 |
| base_zendnn_pp8192_ub4096 | 29.6 | 2.72 | 1.80 | 22.4 | 21.6 | 0.6 | 14.1 | 41.3 | 1402 | 1383 |
| base_zendnn_pp8192_ub512 | 32.0 | 2.72 | 1.74 | 20.4 | 15.4 | 0.7 | 14.2 | 49.1 | 1676 | 1653 |

Top-down columns are % of dispatch slots (PipelineL1/L2). FP columns: retired FP operations per second (fp_ret_sse_avx_ops.all / .mac_flops), across the whole process. Their bf16 and packed-fp32 umasks read 0 on this CPU, so the split by data type is not available.

## Where L1 data-cache fills come from (% of fills) and estimated DRAM read traffic

| run | L2 % | L3 same CCX % | other CCX % | DRAM this socket % | DRAM other socket % | L1 fills from DRAM GB/s | L2 prefetch misses to DRAM GB/s | DRAM read est. GB/s |
|---|---|---|---|---|---|---|---|---|
| base_stock_pp256_ub512 | 87.0 | 6.2 | 0.9 | 5.9 | 0.0 | 44.0 | 49.3 | 93 |
| base_stock_pp8192_ub4096 | 92.0 | 6.3 | 0.3 | 1.5 | 0.0 | 12.3 | 20.8 | 33 |
| base_stock_pp8192_ub512 | 89.7 | 6.0 | 0.7 | 3.6 | 0.0 | 33.2 | 39.7 | 73 |
| base_zendnn_fb0_pp8192_ub512 | 93.4 | 2.1 | 0.6 | 3.9 | 0.0 | 22.3 | 44.4 | 67 |
| base_zendnn_pp256_ub512 | 86.9 | 6.0 | 0.8 | 6.3 | 0.0 | 41.2 | 47.9 | 89 |
| base_zendnn_pp8192_ub4096 | 92.3 | 6.0 | 0.3 | 1.5 | 0.0 | 11.4 | 19.2 | 31 |
| base_zendnn_pp8192_ub512 | 90.4 | 5.5 | 0.6 | 3.5 | 0.0 | 29.6 | 37.8 | 67 |

DRAM near/far = this socket / the other socket. The estimate counts 64-byte lines from L1 fills that came from DRAM plus L2 hardware prefetches that missed L3; it excludes writes. The pods expose no uncore (data-fabric) counters to measure DRAM bandwidth directly. NUMA node 1 has 3 of the socket's 12 DDR5 channels (NPS4), roughly 140 GB/s peak.

## base_stock_pp256_ub512

### Where the cycles go (flat profile, all threads)

| category | % of cycles |
|---|---|
| ggml: bf16 dot product (MUL_MAT_ID experts on CPU) | 44.2 |
| OpenMP runtime (spin-wait at barriers, fork/join) | 22.0 |
| ggml: tinyBLAS/llamafile GEMM (dense matmul on CPU) | 13.5 |
| ggml: graph loop (ggml_barrier, inlined small ops) | 4.6 |
| ggml: other ops (norms, elementwise, softmax, ...) | 4.5 |
| ggml: GDN recurrence + conv1d | 3.6 |
| ggml: f32/f16 dot product | 1.8 |
| ggml: type conversion | 0.7 |
| ggml: flash attention | 0.6 |
| ggml: CONCAT (GDN conv state) | 0.6 |
| ggml-base: other | 0.2 |
| kernel | 0.1 |

### Top symbols

| % | object | symbol |
|---|---|---|
| 44.18 | libggml-cpu.so.0.23.0 | ggml_vec_dot_bf16 |
| 18.45 | libomp.so.5 | 0x000000000007e920 |
| 12.15 | libggml-cpu.so.0.23.0 | void (anonymous namespace)::tinyBLAS<32, float __vector(16), short __vector(32), ggml_bf16 |
| 4.63 | libggml-cpu.so.0.23.0 | ggml_graph_compute_thread.isra.0 |
| 2.39 | libggml-cpu.so.0.23.0 | ggml_compute_forward_gated_delta_net |
| 1.74 | libggml-cpu.so.0.23.0 | ggml_vec_dot_f32 |
| 1.25 | libggml-cpu.so.0.23.0 | ggml_compute_forward_ssm_conv |
| 1.17 | libomp.so.5 | 0x00000000000e1140 |
| 0.94 | libggml-cpu.so.0.23.0 | ggml_compute_forward_mul |
| 0.91 | libggml-cpu.so.0.23.0 | void (anonymous namespace)::tinyBLAS<16, float __vector(16), float __vector(16), float, fl |
| 0.82 | libggml-cpu.so.0.23.0 | ggml_compute_forward_rms_norm_mul_fused |
| 0.74 | libggml-cpu.so.0.23.0 | ggml_cpu_fp32_to_bf16 |
| 0.64 | libggml-cpu.so.0.23.0 | ggml_compute_forward_flash_attn_ext_tiled(ggml_compute_params const*, ggml_tensor*, int, i |
| 0.58 | libggml-cpu.so.0.23.0 | ggml_vec_silu_f32 |
| 0.58 | libggml-cpu.so.0.23.0 | ggml_compute_forward_concat |
| 0.55 | libggml-cpu.so.0.23.0 | ggml_vec_swiglu_f32 |
| 0.39 | libggml-cpu.so.0.23.0 | ggml_compute_forward_mul_mat |
| 0.38 | libggml-cpu.so.0.23.0 | ggml_compute_forward_add_non_quantized |
| 0.38 | libggml-cpu.so.0.23.0 | void (anonymous namespace)::tinyBLAS<32, float __vector(16), short __vector(32), ggml_bf16 |
| 0.29 | libomp.so.5 | 0x000000000007e7c7 |

### Per ggml op (cpu-clock stacks: 238355 samples, 32 threads, main thread 198982)

| op (main thread's stack) | % of wall time | threads doing work (of 32) | % of all CPU samples: work | % of all CPU samples: spinning |
|---|---|---|---|---|
| [leaf] ggml: bf16 dot product (MUL_MAT_ID experts on CPU) | 45.0 | 31.8 | 44.9 | 0.0 |
| concat | 18.6 | 3.8 | 0.6 | 0.0 |
| [leaf] ggml: tinyBLAS/llamafile GEMM (dense matmul on CPU) | 14.0 | 27.9 | 13.2 | 0.0 |
| (ggml barrier: waiting for the slowest thread) | 3.5 | 22.9 | 0.0 | 22.6 |
| [leaf] ggml: graph loop (ggml_barrier, inlined small ops) | 2.8 | 31.7 | 2.7 | 0.0 |
| gated_delta_net | 2.6 | 28.5 | 2.7 | 0.0 |
| [leaf] libc (memcpy / memset) | 2.5 | 30.1 | 3.1 | 0.0 |
| [leaf] ggml: f32/f16 dot product | 1.5 | 28.1 | 1.5 | 0.0 |
| [leaf] ggml: type conversion | 1.4 | 29.7 | 1.4 | 0.0 |
| rms_norm_mul | 0.9 | 27.0 | 0.8 | 0.0 |
| ssm_conv | 0.9 | 20.1 | 1.2 | 0.0 |
| flash_attn_ext | 0.9 | 27.6 | 0.8 | 0.0 |
| mul | 0.7 | 27.8 | 0.8 | 0.0 |
| unary | 0.7 | 21.7 | 0.6 | 0.0 |
| mul_mat | 0.6 | 25.7 | 0.1 | 0.9 |
| [leaf] ggml: other ops (norms, elementwise, softmax, ...) | 0.6 | 28.9 | 0.6 | 0.0 |
| get_rows | 0.6 | 11.8 | 0.0 | 0.0 |
| glu | 0.5 | 29.9 | 0.5 | 0.0 |
| add | 0.5 | 28.2 | 0.4 | 0.0 |
| rms_norm | 0.4 | 17.1 | 0.3 | 0.0 |
| scale | 0.3 | 9.3 | 0.2 | 0.0 |
| argsort | 0.2 | 28.4 | 0.1 | 0.0 |
| [leaf] other | 0.1 | 29.5 | 0.1 | 0.0 |
| sigmoid | 0.1 | 25.2 | 0.0 | 0.0 |
| (OpenMP: idle / fork-join) | 0.0 | 25.5 | 0.0 | 0.1 |
| [leaf] ggml-base: other | 0.0 | 32.0 | 0.0 | 0.0 |

'% of wall time' is the main thread's samples. 'threads doing work' counts, per main-thread sample in that op, the samples of all threads within 6 ms that were not spinning in libomp.

### Threads: 32, work share min/median/max = 75/76/96% (the rest is spinning in libomp)

## base_stock_pp8192_ub4096

### Where the cycles go (flat profile, all threads)

| category | % of cycles |
|---|---|
| OpenMP runtime (spin-wait at barriers, fork/join) | 38.1 |
| ggml: bf16 dot product (MUL_MAT_ID experts on CPU) | 23.2 |
| ggml: tinyBLAS/llamafile GEMM (dense matmul on CPU) | 11.8 |
| ggml: flash attention | 8.1 |
| ggml: other ops (norms, elementwise, softmax, ...) | 5.8 |
| ggml: GDN recurrence + conv1d | 3.6 |
| ggml: graph loop (ggml_barrier, inlined small ops) | 2.7 |
| ggml: f32/f16 dot product | 1.8 |
| ggml: CONCAT (GDN conv state) | 1.3 |
| ggml: type conversion | 0.8 |
| kernel | 0.7 |
| ggml-base: other | 0.2 |
| libc (memcpy / memset) | 0.1 |

### Top symbols

| % | object | symbol |
|---|---|---|
| 30.26 | libomp.so.5 | 0x000000000007e920 |
| 23.25 | libggml-cpu.so.0.23.0 | ggml_vec_dot_bf16 |
| 10.98 | libggml-cpu.so.0.23.0 | void (anonymous namespace)::tinyBLAS<32, float __vector(16), short __vector(32), ggml_bf16 |
| 8.14 | libggml-cpu.so.0.23.0 | ggml_compute_forward_flash_attn_ext_tiled(ggml_compute_params const*, ggml_tensor*, int, i |
| 2.68 | libggml-cpu.so.0.23.0 | ggml_graph_compute_thread.isra.0 |
| 2.34 | libggml-cpu.so.0.23.0 | ggml_compute_forward_gated_delta_net |
| 1.84 | libomp.so.5 | 0x00000000000e1140 |
| 1.75 | libggml-cpu.so.0.23.0 | ggml_vec_dot_f32 |
| 1.36 | libggml-cpu.so.0.23.0 | ggml_compute_forward_mul |
| 1.34 | libggml-cpu.so.0.23.0 | ggml_compute_forward_concat |
| 1.25 | libggml-cpu.so.0.23.0 | ggml_compute_forward_ssm_conv |
| 1.10 | libggml-cpu.so.0.23.0 | ggml_compute_forward_add_non_quantized |
| 0.85 | libggml-cpu.so.0.23.0 | ggml_compute_forward_rms_norm_mul_fused |
| 0.82 | libggml-cpu.so.0.23.0 | void (anonymous namespace)::tinyBLAS<16, float __vector(16), float __vector(16), float, fl |
| 0.70 | libggml-cpu.so.0.23.0 | ggml_cpu_fp32_to_bf16 |
| 0.60 | libggml-cpu.so.0.23.0 | ggml_vec_silu_f32 |
| 0.55 | libggml-cpu.so.0.23.0 | ggml_vec_swiglu_f32 |
| 0.47 | libomp.so.5 | 0x000000000007e8c8 |
| 0.45 | libomp.so.5 | 0x000000000007e7c7 |
| 0.40 | libggml-cpu.so.0.23.0 | ggml_compute_forward_mul_mat |

### Per ggml op (cpu-clock stacks: 221968 samples, 32 threads, main thread 200808)

| op (main thread's stack) | % of wall time | threads doing work (of 32) | % of all CPU samples: work | % of all CPU samples: spinning |
|---|---|---|---|---|
| concat | 42.0 | 2.0 | 1.4 | 0.0 |
| [leaf] ggml: bf16 dot product (MUL_MAT_ID experts on CPU) | 20.9 | 32.3 | 22.3 | 0.0 |
| [leaf] ggml: tinyBLAS/llamafile GEMM (dense matmul on CPU) | 11.4 | 31.8 | 11.9 | 0.0 |
| flash_attn_ext | 6.7 | 31.3 | 7.5 | 0.0 |
| (ggml barrier: waiting for the slowest thread) | 3.2 | 12.8 | 0.0 | 38.5 |
| gated_delta_net | 2.3 | 32.1 | 2.7 | 0.0 |
| [leaf] libc (memcpy / memset) | 1.8 | 29.5 | 3.1 | 0.0 |
| [leaf] ggml: graph loop (ggml_barrier, inlined small ops) | 1.7 | 32.0 | 1.9 | 0.0 |
| [leaf] ggml: f32/f16 dot product | 1.5 | 31.8 | 1.6 | 0.0 |
| [leaf] ggml: type conversion | 1.4 | 31.0 | 1.6 | 0.0 |
| mul | 1.3 | 30.9 | 1.3 | 0.0 |
| ssm_conv | 1.2 | 29.5 | 1.2 | 0.0 |
| add | 1.1 | 28.8 | 1.2 | 0.0 |
| rms_norm_mul | 1.0 | 25.9 | 0.8 | 0.0 |
| unary | 0.5 | 28.5 | 0.6 | 0.0 |
| glu | 0.5 | 31.8 | 0.5 | 0.0 |
| [leaf] other | 0.4 | 9.0 | 0.1 | 0.0 |
| [leaf] ggml: other ops (norms, elementwise, softmax, ...) | 0.4 | 29.6 | 0.4 | 0.0 |
| rms_norm | 0.3 | 21.6 | 0.2 | 0.0 |
| scale | 0.2 | 18.2 | 0.1 | 0.0 |
| argsort | 0.1 | 31.6 | 0.1 | 0.0 |
| mul_mat | 0.1 | 30.3 | 0.0 | 0.2 |
| get_rows | 0.0 | 21.0 | 0.0 | 0.0 |
| soft_max | 0.0 | 25.5 | 0.0 | 0.0 |
| sigmoid | 0.0 | 26.0 | 0.1 | 0.0 |
| sum_rows | 0.0 | 24.0 | 0.0 | 0.0 |
| clamp | 0.0 | 39.0 | 0.0 | 0.0 |

'% of wall time' is the main thread's samples. 'threads doing work' counts, per main-thread sample in that op, the samples of all threads within 6 ms that were not spinning in libomp.

### Threads: 32, work share min/median/max = 59/60/97% (the rest is spinning in libomp)

## base_stock_pp8192_ub512

### Where the cycles go (flat profile, all threads)

| category | % of cycles |
|---|---|
| ggml: bf16 dot product (MUL_MAT_ID experts on CPU) | 35.3 |
| OpenMP runtime (spin-wait at barriers, fork/join) | 22.3 |
| ggml: tinyBLAS/llamafile GEMM (dense matmul on CPU) | 12.7 |
| ggml: flash attention | 10.4 |
| ggml: other ops (norms, elementwise, softmax, ...) | 4.7 |
| ggml: graph loop (ggml_barrier, inlined small ops) | 3.9 |
| ggml: GDN recurrence + conv1d | 3.8 |
| ggml: f32/f16 dot product | 1.8 |
| ggml: type conversion | 1.0 |
| ggml: CONCAT (GDN conv state) | 0.6 |
| ggml-base: other | 0.2 |
| kernel | 0.2 |

### Top symbols

| % | object | symbol |
|---|---|---|
| 35.31 | libggml-cpu.so.0.23.0 | ggml_vec_dot_bf16 |
| 18.70 | libomp.so.5 | 0x000000000007e920 |
| 11.37 | libggml-cpu.so.0.23.0 | void (anonymous namespace)::tinyBLAS<32, float __vector(16), short __vector(32), ggml_bf16 |
| 10.38 | libggml-cpu.so.0.23.0 | ggml_compute_forward_flash_attn_ext_tiled(ggml_compute_params const*, ggml_tensor*, int, i |
| 3.87 | libggml-cpu.so.0.23.0 | ggml_graph_compute_thread.isra.0 |
| 2.46 | libggml-cpu.so.0.23.0 | ggml_compute_forward_gated_delta_net |
| 1.81 | libggml-cpu.so.0.23.0 | ggml_vec_dot_f32 |
| 1.31 | libggml-cpu.so.0.23.0 | ggml_compute_forward_ssm_conv |
| 1.16 | libomp.so.5 | 0x00000000000e1140 |
| 0.90 | libggml-cpu.so.0.23.0 | ggml_compute_forward_mul |
| 0.86 | libggml-cpu.so.0.23.0 | void (anonymous namespace)::tinyBLAS<16, float __vector(16), float __vector(16), float, fl |
| 0.84 | libggml-cpu.so.0.23.0 | ggml_compute_forward_rms_norm_mul_fused |
| 0.67 | libggml-cpu.so.0.23.0 | ggml_cpu_fp32_to_bf16 |
| 0.62 | libggml-cpu.so.0.23.0 | ggml_compute_forward_concat |
| 0.60 | libggml-cpu.so.0.23.0 | ggml_vec_silu_f32 |
| 0.59 | libggml-cpu.so.0.23.0 | ggml_vec_swiglu_f32 |
| 0.41 | libggml-cpu.so.0.23.0 | ggml_compute_forward_add_non_quantized |
| 0.39 | libggml-cpu.so.0.23.0 | void (anonymous namespace)::tinyBLAS<32, float __vector(16), short __vector(32), ggml_bf16 |
| 0.36 | libggml-cpu.so.0.23.0 | ggml_compute_forward_mul_mat |
| 0.29 | libomp.so.5 | 0x000000000007e8c8 |

### Per ggml op (cpu-clock stacks: 238430 samples, 32 threads, main thread 197869)

| op (main thread's stack) | % of wall time | threads doing work (of 32) | % of all CPU samples: work | % of all CPU samples: spinning |
|---|---|---|---|---|
| [leaf] ggml: bf16 dot product (MUL_MAT_ID experts on CPU) | 37.3 | 31.9 | 37.1 | 0.0 |
| concat | 21.4 | 3.0 | 0.7 | 0.0 |
| [leaf] ggml: tinyBLAS/llamafile GEMM (dense matmul on CPU) | 13.9 | 29.5 | 13.7 | 0.0 |
| flash_attn_ext | 5.2 | 31.1 | 5.1 | 0.0 |
| [leaf] ggml: graph loop (ggml_barrier, inlined small ops) | 3.1 | 31.5 | 2.7 | 0.0 |
| (ggml barrier: waiting for the slowest thread) | 3.1 | 22.2 | 0.0 | 24.2 |
| gated_delta_net | 2.8 | 29.3 | 3.0 | 0.0 |
| [leaf] libc (memcpy / memset) | 2.3 | 29.9 | 3.2 | 0.0 |
| [leaf] ggml: f32/f16 dot product | 1.8 | 29.3 | 1.7 | 0.0 |
| [leaf] ggml: type conversion | 1.7 | 29.5 | 1.5 | 0.0 |
| ssm_conv | 1.3 | 20.5 | 1.4 | 0.0 |
| mul | 1.2 | 29.1 | 1.0 | 0.0 |
| rms_norm_mul | 0.9 | 27.0 | 0.9 | 0.0 |
| unary | 0.8 | 27.2 | 0.6 | 0.0 |
| glu | 0.7 | 33.0 | 0.7 | 0.0 |
| add | 0.5 | 29.6 | 0.4 | 0.0 |
| mul_mat | 0.4 | 23.3 | 0.1 | 0.6 |
| [leaf] ggml: other ops (norms, elementwise, softmax, ...) | 0.4 | 29.7 | 0.5 | 0.0 |
| rms_norm | 0.3 | 25.4 | 0.2 | 0.0 |
| get_rows | 0.3 | 13.4 | 0.0 | 0.0 |
| [leaf] other | 0.2 | 15.8 | 0.1 | 0.0 |
| scale | 0.1 | 24.1 | 0.2 | 0.0 |
| [leaf] ggml-base: other | 0.1 | 9.4 | 0.0 | 0.0 |
| argsort | 0.1 | 29.3 | 0.1 | 0.0 |
| sigmoid | 0.1 | 25.6 | 0.1 | 0.0 |
| soft_max | 0.0 | 29.0 | 0.0 | 0.0 |
| dup_bytes | 0.0 | 20.0 | 0.0 | 0.0 |
| softplus | 0.0 | 25.0 | 0.0 | 0.0 |

'% of wall time' is the main thread's samples. 'threads doing work' counts, per main-thread sample in that op, the samples of all threads within 6 ms that were not spinning in libomp.

### Threads: 32, work share min/median/max = 73/74/97% (the rest is spinning in libomp)

## base_zendnn_fb0_pp8192_ub512

### Where the cycles go (flat profile, all threads)

| category | % of cycles |
|---|---|
| OpenMP runtime (spin-wait at barriers, fork/join) | 49.5 |
| ZenDNN/AOCL-DLP GEMM kernels (incl. JIT code) | 20.9 |
| ggml: flash attention | 5.0 |
| ggml: other ops (norms, elementwise, softmax, ...) | 4.9 |
| ggml: GDN recurrence + conv1d | 4.2 |
| ggml: type conversion | 2.4 |
| ggml: f32/f16 dot product | 2.1 |
| kernel | 1.0 |
| libc (memcpy / memset) | 1.0 |
| ggml: CONCAT (GDN conv state) | 0.8 |
| ggml-base: other | 0.1 |

### Top symbols

| % | object | symbol |
|---|---|---|
| 38.81 | libomp.so.5 | 0x000000000007e920 |
| 4.95 | libggml-cpu.so.0.23.0 | ggml_compute_forward_flash_attn_ext_tiled(ggml_compute_params const*, ggml_tensor*, int, i |
| 2.72 | libggml-cpu.so.0.23.0 | ggml_compute_forward_gated_delta_net |
| 2.47 | libomp.so.5 | 0x00000000000e1140 |
| 2.27 | libggml-base.so.0.23.0 | ggml_fp32_to_bf16_row_ref |
| 2.02 | libggml-cpu.so.0.23.0 | ggml_vec_dot_f32 |
| 1.52 | libggml-cpu.so.0.23.0 | ggml_compute_forward_ssm_conv |
| 1.28 | libggml-cpu.so.0.23.0 | ggml_compute_forward_mul |
| 0.92 | libomp.so.5 | 0x000000000007e560 |
| 0.91 | libggml-cpu.so.0.23.0 | ggml_compute_forward_rms_norm_mul_fused |
| 0.76 | libggml-cpu.so.0.23.0 | ggml_compute_forward_concat |
| 0.67 | libggml-cpu.so.0.23.0 | ggml_vec_silu_f32 |
| 0.64 | libggml-cpu.so.0.23.0 | ggml_vec_swiglu_f32 |
| 0.62 | [JIT] tid 199536 | 0x000076dc3b502165 |
| 0.61 | libomp.so.5 | 0x000000000007e8c8 |
| 0.59 | libomp.so.5 | 0x000000000007e7c7 |
| 0.58 | [JIT] tid 199536 | 0x000076dc3b502144 |
| 0.48 | [JIT] tid 199536 | 0x000076dc3b50211d |
| 0.48 | [JIT] tid 199536 | 0x000076dc3b502177 |
| 0.48 | [JIT] tid 199536 | 0x000076dc3b5021a7 |

### Per ggml op (cpu-clock stacks: 238149 samples, 32 threads, main thread 199536)

| op (main thread's stack) | % of wall time | threads doing work (of 32) | % of all CPU samples: work | % of all CPU samples: spinning |
|---|---|---|---|---|
| ZenDNN matmul (AOCL-DLP GEMM kernels) | 34.5 | 20.3 | 25.3 | 0.0 |
| concat | 23.4 | 3.1 | 0.7 | 0.0 |
| ZenDNN mul_mat_id | 10.9 | 15.2 | 0.2 | 0.2 |
| flash_attn_ext | 10.9 | 31.9 | 10.8 | 0.0 |
| (ggml barrier: waiting for the slowest thread) | 4.0 | 21.0 | 0.0 | 25.8 |
| gated_delta_net | 2.8 | 28.7 | 3.1 | 0.0 |
| [leaf] ggml: type conversion | 2.2 | 20.1 | 1.8 | 0.0 |
| [leaf] libc (memcpy / memset) | 2.0 | 17.0 | 3.2 | 0.0 |
| [leaf] ggml: f32/f16 dot product | 1.9 | 29.5 | 1.6 | 0.0 |
| mul | 1.2 | 22.5 | 1.1 | 0.0 |
| ssm_conv | 1.1 | 20.0 | 1.5 | 0.0 |
| rms_norm_mul | 0.8 | 25.3 | 0.8 | 0.0 |
| unary | 0.7 | 28.6 | 0.7 | 0.0 |
| glu | 0.7 | 20.4 | 0.6 | 0.0 |
| [leaf] ggml: other ops (norms, elementwise, softmax, ...) | 0.6 | 22.2 | 0.5 | 0.0 |
| rms_norm | 0.5 | 24.5 | 0.2 | 0.0 |
| [leaf] other | 0.5 | 14.1 | 0.1 | 0.0 |
| add | 0.5 | 24.0 | 0.5 | 0.0 |
| get_rows | 0.3 | 14.6 | 0.0 | 0.0 |
| (OpenMP: idle / fork-join) | 0.2 | 21.6 | 0.0 | 20.7 |
| argsort | 0.2 | 24.5 | 0.2 | 0.0 |
| scale | 0.1 | 25.6 | 0.2 | 0.0 |
| sigmoid | 0.1 | 17.0 | 0.1 | 0.0 |
| [leaf] ggml-base: other | 0.1 | 13.5 | 0.0 | 0.0 |
| soft_max | 0.0 | 28.5 | 0.0 | 0.0 |
| [leaf] ggml: graph loop (ggml_barrier, inlined small ops) | 0.0 | 18.0 | 0.0 | 0.0 |

'% of wall time' is the main thread's samples. 'threads doing work' counts, per main-thread sample in that op, the samples of all threads within 6 ms that were not spinning in libomp.

### Threads: 32, work share min/median/max = 43/52/90% (the rest is spinning in libomp)

## base_zendnn_pp256_ub512

### Where the cycles go (flat profile, all threads)

| category | % of cycles |
|---|---|
| ggml: bf16 dot product (MUL_MAT_ID experts on CPU) | 46.8 |
| OpenMP runtime (spin-wait at barriers, fork/join) | 22.1 |
| ZenDNN/AOCL-DLP GEMM kernels (incl. JIT code) | 9.2 |
| ggml: graph loop (ggml_barrier, inlined small ops) | 4.9 |
| ggml: other ops (norms, elementwise, softmax, ...) | 4.4 |
| ggml: GDN recurrence + conv1d | 3.8 |
| ggml: f32/f16 dot product | 1.8 |
| ggml: type conversion | 1.1 |
| ggml: flash attention | 0.7 |
| ggml: CONCAT (GDN conv state) | 0.6 |
| ggml: tinyBLAS/llamafile GEMM (dense matmul on CPU) | 0.1 |
| kernel | 0.1 |
| ggml-base: other | 0.1 |

### Top symbols

| % | object | symbol |
|---|---|---|
| 46.81 | libggml-cpu.so.0.23.0 | ggml_vec_dot_bf16 |
| 18.50 | libomp.so.5 | 0x000000000007e920 |
| 4.86 | libggml-cpu.so.0.23.0 | ggml_graph_compute_thread.isra.0 |
| 2.48 | libggml-cpu.so.0.23.0 | ggml_compute_forward_gated_delta_net |
| 1.79 | libggml-cpu.so.0.23.0 | ggml_vec_dot_f32 |
| 1.30 | libggml-cpu.so.0.23.0 | ggml_compute_forward_ssm_conv |
| 1.15 | libomp.so.5 | 0x00000000000e1140 |
| 1.00 | libggml-cpu.so.0.23.0 | ggml_compute_forward_mul |
| 0.81 | libggml-cpu.so.0.23.0 | ggml_compute_forward_rms_norm_mul_fused |
| 0.79 | libggml-base.so.0.23.0 | ggml_fp32_to_bf16_row_ref |
| 0.69 | libggml-cpu.so.0.23.0 | ggml_vec_silu_f32 |
| 0.67 | libggml-cpu.so.0.23.0 | ggml_compute_forward_flash_attn_ext_tiled(ggml_compute_params const*, ggml_tensor*, int, i |
| 0.61 | libggml-cpu.so.0.23.0 | ggml_vec_swiglu_f32 |
| 0.56 | libggml-cpu.so.0.23.0 | ggml_compute_forward_concat |
| 0.42 | libggml-cpu.so.0.23.0 | ggml_compute_forward_add_non_quantized |
| 0.32 | libggml-cpu.so.0.23.0 | ggml_cpu_fp32_to_bf16 |
| 0.30 | [JIT] tid 198419 | 0x00007a5dc6eda165 |
| 0.28 | libomp.so.5 | 0x000000000007e8c8 |
| 0.28 | libomp.so.5 | 0x000000000007e7c7 |
| 0.26 | [JIT] tid 198419 | 0x00007a5dc6eda144 |

### Per ggml op (cpu-clock stacks: 238098 samples, 32 threads, main thread 198419)

| op (main thread's stack) | % of wall time | threads doing work (of 32) | % of all CPU samples: work | % of all CPU samples: spinning |
|---|---|---|---|---|
| [leaf] ggml: bf16 dot product (MUL_MAT_ID experts on CPU) | 48.1 | 31.6 | 47.3 | 0.0 |
| concat | 17.8 | 3.9 | 0.5 | 0.0 |
| ZenDNN matmul (AOCL-DLP GEMM kernels) | 11.5 | 26.6 | 10.0 | 0.0 |
| (ggml barrier: waiting for the slowest thread) | 3.6 | 22.1 | 0.0 | 22.1 |
| [leaf] ggml: graph loop (ggml_barrier, inlined small ops) | 3.1 | 31.6 | 2.8 | 0.0 |
| gated_delta_net | 2.7 | 27.8 | 2.8 | 0.0 |
| [leaf] libc (memcpy / memset) | 2.2 | 29.7 | 3.1 | 0.0 |
| [leaf] ggml: f32/f16 dot product | 1.6 | 28.1 | 1.5 | 0.0 |
| [leaf] ggml: type conversion | 1.5 | 27.5 | 1.4 | 0.0 |
| unary | 1.1 | 21.2 | 0.6 | 0.0 |
| flash_attn_ext | 0.9 | 25.3 | 0.8 | 0.0 |
| ssm_conv | 0.9 | 17.9 | 1.2 | 0.0 |
| mul | 0.8 | 28.4 | 1.0 | 0.0 |
| rms_norm_mul | 0.8 | 27.9 | 0.8 | 0.0 |
| get_rows | 0.6 | 9.4 | 0.0 | 0.0 |
| [leaf] ggml: other ops (norms, elementwise, softmax, ...) | 0.5 | 24.6 | 0.5 | 0.0 |
| glu | 0.4 | 30.7 | 0.7 | 0.0 |
| add | 0.4 | 27.4 | 0.4 | 0.0 |
| rms_norm | 0.4 | 16.3 | 0.3 | 0.0 |
| scale | 0.3 | 8.6 | 0.1 | 0.0 |
| argsort | 0.2 | 26.6 | 0.2 | 0.0 |
| (OpenMP: idle / fork-join) | 0.2 | 23.9 | 0.0 | 1.4 |
| mul_mat | 0.2 | 26.7 | 0.0 | 0.2 |
| [leaf] other | 0.1 | 27.1 | 0.1 | 0.0 |
| [leaf] ggml: tinyBLAS/llamafile GEMM (dense matmul on CPU) | 0.1 | 29.9 | 0.1 | 0.0 |
| sigmoid | 0.1 | 26.6 | 0.1 | 0.0 |
| [leaf] ggml-base: other | 0.1 | 27.2 | 0.0 | 0.0 |
| soft_max | 0.0 | 24.5 | 0.0 | 0.0 |
| set_rows | 0.0 | 31.0 | 0.0 | 0.0 |
| clamp | 0.0 | 22.0 | 0.0 | 0.0 |

'% of wall time' is the main thread's samples. 'threads doing work' counts, per main-thread sample in that op, the samples of all threads within 6 ms that were not spinning in libomp.

### Threads: 32, work share min/median/max = 75/76/95% (the rest is spinning in libomp)

## base_zendnn_pp8192_ub4096

### Where the cycles go (flat profile, all threads)

| category | % of cycles |
|---|---|
| OpenMP runtime (spin-wait at barriers, fork/join) | 40.5 |
| ggml: bf16 dot product (MUL_MAT_ID experts on CPU) | 23.8 |
| ggml: flash attention | 9.2 |
| ZenDNN/AOCL-DLP GEMM kernels (incl. JIT code) | 7.0 |
| ggml: other ops (norms, elementwise, softmax, ...) | 5.4 |
| ggml: GDN recurrence + conv1d | 3.7 |
| ggml: graph loop (ggml_barrier, inlined small ops) | 2.8 |
| ggml: f32/f16 dot product | 1.7 |
| ggml: CONCAT (GDN conv state) | 1.4 |
| ggml: type conversion | 1.3 |
| kernel | 0.7 |
| ggml: tinyBLAS/llamafile GEMM (dense matmul on CPU) | 0.2 |
| libc (memcpy / memset) | 0.1 |

### Top symbols

| % | object | symbol |
|---|---|---|
| 32.10 | libomp.so.5 | 0x000000000007e920 |
| 23.80 | libggml-cpu.so.0.23.0 | ggml_vec_dot_bf16 |
| 9.18 | libggml-cpu.so.0.23.0 | ggml_compute_forward_flash_attn_ext_tiled(ggml_compute_params const*, ggml_tensor*, int, i |
| 2.79 | libggml-cpu.so.0.23.0 | ggml_graph_compute_thread.isra.0 |
| 2.36 | libggml-cpu.so.0.23.0 | ggml_compute_forward_gated_delta_net |
| 2.02 | libomp.so.5 | 0x00000000000e1140 |
| 1.74 | libggml-cpu.so.0.23.0 | ggml_vec_dot_f32 |
| 1.45 | libggml-cpu.so.0.23.0 | ggml_compute_forward_concat |
| 1.41 | libggml-cpu.so.0.23.0 | ggml_compute_forward_mul |
| 1.35 | libggml-cpu.so.0.23.0 | ggml_compute_forward_ssm_conv |
| 1.10 | libggml-cpu.so.0.23.0 | ggml_compute_forward_add_non_quantized |
| 0.85 | libggml-cpu.so.0.23.0 | ggml_compute_forward_rms_norm_mul_fused |
| 0.75 | libggml-base.so.0.23.0 | ggml_fp32_to_bf16_row_ref |
| 0.58 | libggml-cpu.so.0.23.0 | ggml_vec_silu_f32 |
| 0.56 | libggml-cpu.so.0.23.0 | ggml_vec_swiglu_f32 |
| 0.51 | libomp.so.5 | 0x000000000007e8c8 |
| 0.49 | libomp.so.5 | 0x000000000007e7c7 |
| 0.37 | libomp.so.5 | 0x000000000007e560 |
| 0.34 | libggml-cpu.so.0.23.0 | ggml_cpu_fp32_to_bf16 |
| 0.32 | libomp.so.5 | 0x000000000007e9e6 |

### Per ggml op (cpu-clock stacks: 216036 samples, 32 threads, main thread 200242)

| op (main thread's stack) | % of wall time | threads doing work (of 32) | % of all CPU samples: work | % of all CPU samples: spinning |
|---|---|---|---|---|
| concat | 45.2 | 1.9 | 1.5 | 0.0 |
| [leaf] ggml: bf16 dot product (MUL_MAT_ID experts on CPU) | 21.7 | 31.9 | 23.6 | 0.0 |
| ZenDNN matmul (AOCL-DLP GEMM kernels) | 7.5 | 30.6 | 7.7 | 0.0 |
| flash_attn_ext | 6.0 | 31.2 | 6.9 | 0.0 |
| (ggml barrier: waiting for the slowest thread) | 3.0 | 12.5 | 0.0 | 40.7 |
| gated_delta_net | 2.3 | 32.0 | 2.9 | 0.0 |
| [leaf] ggml: graph loop (ggml_barrier, inlined small ops) | 2.1 | 30.5 | 2.0 | 0.0 |
| [leaf] ggml: f32/f16 dot product | 1.8 | 31.6 | 1.6 | 0.0 |
| [leaf] libc (memcpy / memset) | 1.6 | 29.6 | 3.2 | 0.0 |
| [leaf] ggml: type conversion | 1.4 | 30.2 | 1.6 | 0.0 |
| mul | 1.3 | 31.2 | 1.5 | 0.0 |
| ssm_conv | 1.3 | 30.8 | 1.4 | 0.0 |
| add | 1.1 | 27.5 | 1.1 | 0.0 |
| rms_norm_mul | 1.0 | 25.8 | 1.0 | 0.0 |
| unary | 0.5 | 29.7 | 0.7 | 0.0 |
| rms_norm | 0.5 | 20.8 | 0.3 | 0.0 |
| glu | 0.5 | 32.7 | 0.6 | 0.0 |
| [leaf] ggml: other ops (norms, elementwise, softmax, ...) | 0.4 | 28.8 | 0.4 | 0.0 |
| [leaf] other | 0.3 | 5.1 | 0.1 | 0.0 |
| [leaf] ggml: tinyBLAS/llamafile GEMM (dense matmul on CPU) | 0.2 | 25.7 | 0.2 | 0.0 |
| argsort | 0.1 | 29.0 | 0.1 | 0.0 |
| scale | 0.1 | 15.8 | 0.2 | 0.0 |
| sigmoid | 0.1 | 22.5 | 0.0 | 0.0 |
| get_rows | 0.0 | 30.0 | 0.0 | 0.0 |
| (OpenMP: idle / fork-join) | 0.0 | 30.5 | 0.0 | 0.7 |
| [leaf] ggml-base: other | 0.0 | 16.5 | 0.0 | 0.0 |
| soft_max | 0.0 | 32.0 | 0.0 | 0.0 |
| mul_mat | 0.0 | 31.0 | 0.0 | 0.0 |
| sum_rows | 0.0 | 2.0 | 0.0 | 0.0 |

'% of wall time' is the main thread's samples. 'threads doing work' counts, per main-thread sample in that op, the samples of all threads within 6 ms that were not spinning in libomp.

### Threads: 32, work share min/median/max = 56/57/97% (the rest is spinning in libomp)

## base_zendnn_pp8192_ub512

### Where the cycles go (flat profile, all threads)

| category | % of cycles |
|---|---|
| ggml: bf16 dot product (MUL_MAT_ID experts on CPU) | 36.4 |
| OpenMP runtime (spin-wait at barriers, fork/join) | 23.6 |
| ggml: flash attention | 10.9 |
| ZenDNN/AOCL-DLP GEMM kernels (incl. JIT code) | 8.2 |
| ggml: other ops (norms, elementwise, softmax, ...) | 4.6 |
| ggml: graph loop (ggml_barrier, inlined small ops) | 4.0 |
| ggml: GDN recurrence + conv1d | 3.9 |
| ggml: f32/f16 dot product | 1.9 |
| ggml: type conversion | 1.4 |
| ggml: CONCAT (GDN conv state) | 0.6 |
| kernel | 0.2 |
| ggml: tinyBLAS/llamafile GEMM (dense matmul on CPU) | 0.1 |
| ggml-base: other | 0.1 |

### Top symbols

| % | object | symbol |
|---|---|---|
| 36.43 | libggml-cpu.so.0.23.0 | ggml_vec_dot_bf16 |
| 19.75 | libomp.so.5 | 0x000000000007e920 |
| 10.87 | libggml-cpu.so.0.23.0 | ggml_compute_forward_flash_attn_ext_tiled(ggml_compute_params const*, ggml_tensor*, int, i |
| 3.98 | libggml-cpu.so.0.23.0 | ggml_graph_compute_thread.isra.0 |
| 2.49 | libggml-cpu.so.0.23.0 | ggml_compute_forward_gated_delta_net |
| 1.87 | libggml-cpu.so.0.23.0 | ggml_vec_dot_f32 |
| 1.39 | libggml-cpu.so.0.23.0 | ggml_compute_forward_ssm_conv |
| 1.23 | libomp.so.5 | 0x00000000000e1140 |
| 0.97 | libggml-cpu.so.0.23.0 | ggml_compute_forward_mul |
| 0.80 | libggml-cpu.so.0.23.0 | ggml_compute_forward_rms_norm_mul_fused |
| 0.78 | libggml-base.so.0.23.0 | ggml_fp32_to_bf16_row_ref |
| 0.66 | libggml-cpu.so.0.23.0 | ggml_vec_silu_f32 |
| 0.64 | libggml-cpu.so.0.23.0 | ggml_compute_forward_concat |
| 0.56 | libggml-cpu.so.0.23.0 | ggml_vec_swiglu_f32 |
| 0.46 | libggml-cpu.so.0.23.0 | ggml_compute_forward_add_non_quantized |
| 0.34 | libggml-cpu.so.0.23.0 | ggml_cpu_fp32_to_bf16 |
| 0.31 | libggml-cpu.so.0.23.0 | ggml_cpu_fp16_to_fp32 |
| 0.31 | libggml-cpu.so.0.23.0 | ggml_vec_soft_max_f32 |
| 0.31 | libomp.so.5 | 0x000000000007e7c7 |
| 0.30 | libomp.so.5 | 0x000000000007e8c8 |

### Per ggml op (cpu-clock stacks: 238358 samples, 32 threads, main thread 197296)

| op (main thread's stack) | % of wall time | threads doing work (of 32) | % of all CPU samples: work | % of all CPU samples: spinning |
|---|---|---|---|---|
| [leaf] ggml: bf16 dot product (MUL_MAT_ID experts on CPU) | 38.7 | 32.0 | 39.0 | 0.0 |
| concat | 22.6 | 3.1 | 0.7 | 0.0 |
| ZenDNN matmul (AOCL-DLP GEMM kernels) | 10.3 | 27.9 | 9.7 | 0.0 |
| flash_attn_ext | 5.1 | 31.5 | 5.2 | 0.0 |
| (ggml barrier: waiting for the slowest thread) | 3.6 | 22.1 | 0.0 | 25.4 |
| gated_delta_net | 3.2 | 29.9 | 3.1 | 0.0 |
| [leaf] ggml: graph loop (ggml_barrier, inlined small ops) | 3.0 | 31.5 | 2.7 | 0.0 |
| [leaf] libc (memcpy / memset) | 2.5 | 30.3 | 3.2 | 0.0 |
| [leaf] ggml: type conversion | 1.7 | 27.5 | 1.5 | 0.0 |
| [leaf] ggml: f32/f16 dot product | 1.5 | 28.2 | 1.8 | 0.0 |
| ssm_conv | 1.3 | 19.5 | 1.5 | 0.0 |
| rms_norm_mul | 1.1 | 25.0 | 0.8 | 0.0 |
| mul | 1.0 | 28.2 | 0.9 | 0.0 |
| unary | 0.9 | 25.6 | 0.7 | 0.0 |
| glu | 0.7 | 31.1 | 0.6 | 0.0 |
| [leaf] ggml: other ops (norms, elementwise, softmax, ...) | 0.6 | 28.8 | 0.6 | 0.0 |
| rms_norm | 0.5 | 23.0 | 0.3 | 0.0 |
| add | 0.4 | 25.3 | 0.4 | 0.0 |
| get_rows | 0.2 | 11.2 | 0.0 | 0.0 |
| scale | 0.2 | 25.1 | 0.2 | 0.0 |
| [leaf] other | 0.2 | 17.8 | 0.1 | 0.0 |
| [leaf] ggml-base: other | 0.2 | 20.5 | 0.0 | 0.0 |
| argsort | 0.1 | 26.8 | 0.2 | 0.0 |
| (OpenMP: idle / fork-join) | 0.1 | 21.6 | 0.0 | 1.3 |
| [leaf] ggml: tinyBLAS/llamafile GEMM (dense matmul on CPU) | 0.1 | 25.0 | 0.1 | 0.0 |
| mul_mat | 0.1 | 26.5 | 0.0 | 0.1 |
| sigmoid | 0.0 | 25.0 | 0.0 | 0.0 |
| soft_max | 0.0 | 30.0 | 0.0 | 0.0 |
| clamp | 0.0 | 43.0 | 0.0 | 0.0 |

'% of wall time' is the main thread's samples. 'threads doing work' counts, per main-thread sample in that op, the samples of all threads within 6 ms that were not spinning in libomp.

### Threads: 32, work share min/median/max = 72/72/96% (the rest is spinning in libomp)

