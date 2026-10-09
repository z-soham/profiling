# perf analysis: perf_qwen36_bf16_20261007_0531

Qwen3.6-35B-A3B BF16 prefill with the optimised commands of turin-optimised-commands-2026-09-30.md (boost off): llama.cpp setup B on sacsharm's existing stock and ZenDNN builds (weights loaded with --load-mode dio into local memory), vLLM setup C. 32 cores / 1 NUMA node. perf attached after load and warm-up (vLLM: to its worker process); one measurement per window. See perf_llama.sh.

## Runs

| run | prefill t/s while profiled (median pass) | passes | start-up s | resident memory per NUMA node |
|---|---|---|---|---|
| latest_zendnn_fb0_pp8192_ub4096 | 269.4 | 2 | 159 | N1=118.2GB N0=0.0GB |
| latest_zendnn_fb0_pp8192_ub512 | 266.7 | 2 | 158 | N1=117.5GB N0=0.0GB |
| latest_zendnn_pp8192_ub512 | 280.3 | 3 | 116 | N1=65.2GB N0=0.0GB |

## Counters (whole process, during the windows)

| run | cores busy | GHz | IPC | retiring % | frontend % | bad spec % | backend: cpu % | backend: memory % | FP G-ops/s (all) | FP G-ops/s (mul-add) |
|---|---|---|---|---|---|---|---|---|---|---|
| latest_zendnn_fb0_pp8192_ub4096 | 15.4 | 2.71 | 2.21 | 25.8 | 7.6 | 0.8 | 7.4 | 58.4 | 1600 | 1587 |
| latest_zendnn_fb0_pp8192_ub512 | 24.7 | 2.72 | 1.60 | 19.2 | 16.3 | 0.5 | 7.0 | 57.0 | 1567 | 1555 |
| latest_zendnn_pp8192_ub512 | 24.8 | 2.72 | 1.82 | 20.5 | 13.0 | 0.8 | 13.6 | 52.0 | 1627 | 1604 |

Top-down columns are % of dispatch slots (PipelineL1/L2). FP columns: retired FP operations per second (fp_ret_sse_avx_ops.all / .mac_flops), across the whole process. Their bf16 and packed-fp32 umasks read 0 on this CPU, so the split by data type is not available.

## Where L1 data-cache fills come from (% of fills) and estimated DRAM read traffic

| run | L2 % | L3 same CCX % | other CCX % | DRAM this socket % | DRAM other socket % | L1 fills from DRAM GB/s | L2 prefetch misses to DRAM GB/s | DRAM read est. GB/s |
|---|---|---|---|---|---|---|---|---|
| latest_zendnn_fb0_pp8192_ub4096 | 93.7 | 3.7 | 0.4 | 2.2 | 0.0 | 11.3 | 26.3 | 38 |
| latest_zendnn_fb0_pp8192_ub512 | 90.9 | 3.8 | 0.8 | 4.5 | 0.0 | 23.0 | 43.5 | 66 |
| latest_zendnn_pp8192_ub512 | 89.9 | 5.8 | 0.6 | 3.7 | 0.0 | 29.3 | 34.7 | 64 |

DRAM near/far = this socket / the other socket. The estimate counts 64-byte lines from L1 fills that came from DRAM plus L2 hardware prefetches that missed L3; it excludes writes. The pods expose no uncore (data-fabric) counters to measure DRAM bandwidth directly. NUMA node 1 has 3 of the socket's 12 DDR5 channels (NPS4), roughly 140 GB/s peak.

## latest_zendnn_fb0_pp8192_ub4096

### Where the cycles go (flat profile, all threads)

| category | % of cycles |
|---|---|
| ZenDNN/AOCL-DLP GEMM kernels (incl. JIT code) | 30.3 |
| ggml: flash attention | 16.2 |
| ggml: other ops (norms, elementwise, softmax, ...) | 11.4 |
| OpenMP runtime (spin-wait at barriers, fork/join) | 10.9 |
| ggml: GDN recurrence + conv1d | 8.8 |
| ggml: type conversion | 5.1 |
| ggml: f32/f16 dot product | 3.8 |
| ggml: CONCAT (GDN conv state) | 3.0 |
| libc (memcpy / memset) | 2.1 |
| ggml-base: other | 0.1 |
| kernel | 0.1 |

### Top symbols

| % | object | symbol |
|---|---|---|
| 16.22 | libggml-cpu.so.0.26.0 | ggml_compute_forward_flash_attn_ext_tiled(ggml_compute_params const*, ggml_tensor*, int, i |
| 9.63 | libomp.so.5 | 0x000000000007df00 |
| 5.75 | libggml-cpu.so.0.26.0 | ggml_compute_forward_gated_delta_net |
| 4.76 | libggml-base.so.0.26.0 | ggml_fp32_to_bf16_row_ref |
| 3.74 | libggml-cpu.so.0.26.0 | ggml_vec_dot_f32 |
| 3.07 | libggml-cpu.so.0.26.0 | ggml_compute_forward_ssm_conv |
| 3.02 | libggml-cpu.so.0.26.0 | ggml_compute_forward_concat |
| 2.88 | libggml-cpu.so.0.26.0 | ggml_compute_forward_mul |
| 2.32 | libggml-cpu.so.0.26.0 | ggml_compute_forward_add_non_quantized |
| 1.79 | libggml-cpu.so.0.26.0 | ggml_compute_forward_rms_norm_mul_fused |
| 1.28 | libggml-cpu.so.0.26.0 | ggml_vec_silu_f32 |
| 1.26 | libggml-cpu.so.0.26.0 | ggml_vec_swiglu_f32 |
| 1.00 | [JIT] tid 1961892 | 0x00007a1c688da165 |
| 0.87 | [JIT] tid 1961892 | 0x00007a1c688da144 |
| 0.82 | [JIT] tid 1961892 | 0x00007a1c688da1b9 |
| 0.82 | [JIT] tid 1961892 | 0x00007a1c688da1b3 |
| 0.82 | [JIT] tid 1961892 | 0x00007a1c688da1ad |
| 0.82 | [JIT] tid 1961892 | 0x00007a1c688da16b |
| 0.81 | [JIT] tid 1961892 | 0x00007a1c688da1a7 |
| 0.79 | [JIT] tid 1961892 | 0x00007a1c688da177 |

### Per ggml op (cpu-clock stacks: 108338 samples, 32 threads, main thread 1961892)

| op (main thread's stack) | % of wall time | threads doing work (of 32) | % of all CPU samples: work | % of all CPU samples: spinning |
|---|---|---|---|---|
| concat | 48.8 | 1.0 | 3.3 | 0.0 |
| ZenDNN matmul (AOCL-DLP GEMM kernels) | 22.3 | 24.9 | 37.1 | 0.1 |
| ZenDNN mul_mat_id | 4.8 | 8.5 | 0.1 | 0.3 |
| flash_attn_ext | 4.8 | 32.5 | 11.6 | 0.0 |
| (ggml barrier: waiting for the slowest thread) | 2.9 | 12.7 | 0.0 | 3.3 |
| gated_delta_net | 2.9 | 31.7 | 6.7 | 0.0 |
| [leaf] ggml: type conversion | 2.1 | 22.3 | 5.0 | 0.0 |
| [leaf] ggml: f32/f16 dot product | 1.6 | 32.5 | 3.4 | 0.0 |
| mul | 1.5 | 29.3 | 3.1 | 0.0 |
| ssm_conv | 1.4 | 31.4 | 3.2 | 0.0 |
| [leaf] libc (memcpy / memset) | 1.4 | 20.4 | 3.3 | 0.0 |
| add | 1.2 | 26.8 | 2.7 | 0.0 |
| rms_norm_mul | 1.1 | 26.1 | 1.9 | 0.0 |
| glu | 0.6 | 26.7 | 1.4 | 0.0 |
| unary | 0.6 | 29.2 | 1.4 | 0.0 |
| rms_norm | 0.5 | 17.9 | 0.5 | 0.0 |
| [leaf] ggml: other ops (norms, elementwise, softmax, ...) | 0.4 | 28.2 | 0.9 | 0.0 |
| [leaf] other | 0.4 | 10.3 | 0.2 | 0.0 |
| scale | 0.2 | 18.5 | 0.3 | 0.0 |
| argsort | 0.1 | 30.1 | 0.3 | 0.0 |
| (OpenMP: idle / fork-join) | 0.1 | 23.2 | 0.0 | 9.6 |
| soft_max | 0.0 | 30.0 | 0.0 | 0.0 |
| get_rows | 0.0 | 18.0 | 0.0 | 0.0 |
| [leaf] ggml: graph loop (ggml_barrier, inlined small ops) | 0.0 | 26.0 | 0.0 | 0.0 |
| div | 0.0 | 25.0 | 0.0 | 0.0 |
| sum_rows | 0.0 | 27.0 | 0.0 | 0.0 |

'% of wall time' is the main thread's samples. 'threads doing work' counts, per main-thread sample in that op, the samples of all threads within 6 ms that were not spinning in libomp.

### Threads: 32, work share min/median/max = 85/86/91% (the rest is spinning in libomp)

## latest_zendnn_fb0_pp8192_ub512

### Where the cycles go (flat profile, all threads)

| category | % of cycles |
|---|---|
| OpenMP runtime (spin-wait at barriers, fork/join) | 31.4 |
| ZenDNN/AOCL-DLP GEMM kernels (incl. JIT code) | 28.6 |
| ggml: flash attention | 10.5 |
| ggml: other ops (norms, elementwise, softmax, ...) | 5.3 |
| ggml: GDN recurrence + conv1d | 5.1 |
| ggml: type conversion | 3.1 |
| ggml: f32/f16 dot product | 2.2 |
| ggml: CONCAT (GDN conv state) | 1.0 |
| libc (memcpy / memset) | 0.9 |
| kernel | 0.4 |
| ggml-base: other | 0.1 |

### Top symbols

| % | object | symbol |
|---|---|---|
| 25.84 | libomp.so.5 | 0x000000000007df00 |
| 10.48 | libggml-cpu.so.0.26.0 | ggml_compute_forward_flash_attn_ext_tiled(ggml_compute_params const*, ggml_tensor*, int, i |
| 3.33 | libggml-cpu.so.0.26.0 | ggml_compute_forward_gated_delta_net |
| 2.68 | libggml-base.so.0.26.0 | ggml_fp32_to_bf16_row_ref |
| 2.17 | libggml-cpu.so.0.26.0 | ggml_vec_dot_f32 |
| 1.75 | libggml-cpu.so.0.26.0 | ggml_compute_forward_ssm_conv |
| 1.57 | libomp.so.5 | 0x00000000000e1140 |
| 1.33 | libggml-cpu.so.0.26.0 | ggml_compute_forward_mul |
| 0.98 | libggml-cpu.so.0.26.0 | ggml_compute_forward_concat |
| 0.93 | libggml-cpu.so.0.26.0 | ggml_compute_forward_rms_norm_mul_fused |
| 0.81 | [JIT] tid 1961039 | 0x00007a614a64f165 |
| 0.78 | [JIT] tid 1961039 | 0x00007a614a64f144 |
| 0.73 | libggml-cpu.so.0.26.0 | ggml_vec_swiglu_f32 |
| 0.72 | libggml-cpu.so.0.26.0 | ggml_vec_silu_f32 |
| 0.66 | [JIT] tid 1961039 | 0x00007a614a64f0f0 |
| 0.66 | [JIT] tid 1961039 | 0x00007a614a64f129 |
| 0.65 | [JIT] tid 1961039 | 0x00007a614a64f177 |
| 0.65 | [JIT] tid 1961039 | 0x00007a614a64f135 |
| 0.64 | [JIT] tid 1961039 | 0x00007a614a64f1d8 |
| 0.64 | [JIT] tid 1961039 | 0x00007a614a64f171 |

### Per ggml op (cpu-clock stacks: 177534 samples, 32 threads, main thread 1961039)

| op (main thread's stack) | % of wall time | threads doing work (of 32) | % of all CPU samples: work | % of all CPU samples: spinning |
|---|---|---|---|---|
| ZenDNN matmul (AOCL-DLP GEMM kernels) | 47.6 | 18.8 | 37.6 | 0.4 |
| concat | 24.8 | 1.8 | 1.0 | 0.0 |
| flash_attn_ext | 5.3 | 31.1 | 7.1 | 0.0 |
| ZenDNN mul_mat_id | 4.6 | 13.3 | 0.1 | 0.2 |
| (ggml barrier: waiting for the slowest thread) | 3.3 | 22.4 | 0.0 | 6.4 |
| gated_delta_net | 2.9 | 28.7 | 4.0 | 0.0 |
| [leaf] ggml: type conversion | 1.9 | 19.0 | 2.7 | 0.0 |
| [leaf] ggml: f32/f16 dot product | 1.6 | 29.3 | 2.1 | 0.0 |
| [leaf] libc (memcpy / memset) | 1.4 | 19.2 | 2.6 | 0.0 |
| ssm_conv | 1.2 | 18.6 | 1.8 | 0.0 |
| mul | 0.9 | 21.0 | 1.3 | 0.0 |
| rms_norm_mul | 0.8 | 26.1 | 1.0 | 0.0 |
| glu | 0.7 | 16.9 | 0.8 | 0.0 |
| [leaf] ggml: other ops (norms, elementwise, softmax, ...) | 0.6 | 22.6 | 0.6 | 0.0 |
| rms_norm | 0.5 | 24.7 | 0.4 | 0.0 |
| unary | 0.5 | 27.6 | 0.7 | 0.0 |
| add | 0.4 | 24.6 | 0.5 | 0.0 |
| [leaf] other | 0.4 | 18.4 | 0.1 | 0.0 |
| get_rows | 0.2 | 15.1 | 0.0 | 0.0 |
| argsort | 0.2 | 19.9 | 0.2 | 0.0 |
| (OpenMP: idle / fork-join) | 0.1 | 19.0 | 0.0 | 28.1 |
| [leaf] ggml-base: other | 0.1 | 6.8 | 0.0 | 0.0 |
| scale | 0.1 | 24.6 | 0.1 | 0.0 |
| sigmoid | 0.1 | 22.2 | 0.1 | 0.0 |
| soft_max | 0.0 | 12.0 | 0.0 | 0.0 |
| [leaf] kernel | 0.0 | 17.0 | 0.0 | 0.0 |

'% of wall time' is the main thread's samples. 'threads doing work' counts, per main-thread sample in that op, the samples of all threads within 6 ms that were not spinning in libomp.

### Threads: 32, work share min/median/max = 59/64/82% (the rest is spinning in libomp)

## latest_zendnn_pp8192_ub512

### Where the cycles go (flat profile, all threads)

| category | % of cycles |
|---|---|
| ggml: bf16 dot product (MUL_MAT_ID experts on CPU) | 45.3 |
| ggml: flash attention | 12.4 |
| ZenDNN/AOCL-DLP GEMM kernels (incl. JIT code) | 10.4 |
| OpenMP runtime (spin-wait at barriers, fork/join) | 6.7 |
| ggml: other ops (norms, elementwise, softmax, ...) | 5.3 |
| ggml: GDN recurrence + conv1d | 5.3 |
| ggml: graph loop (ggml_barrier, inlined small ops) | 5.0 |
| ggml: f32/f16 dot product | 2.3 |
| ggml: type conversion | 2.3 |
| ggml: CONCAT (GDN conv state) | 1.0 |
| ggml: tinyBLAS/llamafile GEMM (dense matmul on CPU) | 0.1 |
| kernel | 0.1 |

### Top symbols

| % | object | symbol |
|---|---|---|
| 45.31 | libggml-cpu.so.0.26.0 | ggml_vec_dot_bf16 |
| 12.42 | libggml-cpu.so.0.26.0 | ggml_compute_forward_flash_attn_ext_tiled(ggml_compute_params const*, ggml_tensor*, int, i |
| 6.11 | libomp.so.5 | 0x000000000007df00 |
| 5.02 | libggml-cpu.so.0.26.0 | ggml_graph_compute_thread.isra.0 |
| 3.45 | libggml-cpu.so.0.26.0 | ggml_compute_forward_gated_delta_net |
| 2.27 | libggml-cpu.so.0.26.0 | ggml_vec_dot_f32 |
| 1.81 | libggml-cpu.so.0.26.0 | ggml_compute_forward_ssm_conv |
| 1.18 | libggml-cpu.so.0.26.0 | ggml_compute_forward_mul |
| 1.00 | libggml-cpu.so.0.26.0 | ggml_compute_forward_concat |
| 0.98 | libggml-base.so.0.26.0 | ggml_fp32_to_bf16_row_ref |
| 0.97 | libggml-cpu.so.0.26.0 | ggml_compute_forward_rms_norm_mul_fused |
| 0.79 | libggml-cpu.so.0.26.0 | ggml_vec_silu_f32 |
| 0.78 | libggml-cpu.so.0.26.0 | ggml_cpu_fp32_to_bf16 |
| 0.73 | libggml-cpu.so.0.26.0 | ggml_vec_swiglu_f32 |
| 0.53 | libggml-cpu.so.0.26.0 | ggml_cpu_fp16_to_fp32 |
| 0.49 | libggml-cpu.so.0.26.0 | ggml_compute_forward_add_non_quantized |
| 0.37 | libomp.so.5 | 0x00000000000e1140 |
| 0.37 | [JIT] tid 1960383 | 0x0000768a6d854165 |
| 0.29 | [JIT] tid 1960383 | 0x0000768a6d854144 |
| 0.29 | [JIT] tid 1960383 | 0x0000768a6d8541b9 |

### Per ggml op (cpu-clock stacks: 181545 samples, 32 threads, main thread 1960383)

| op (main thread's stack) | % of wall time | threads doing work (of 32) | % of all CPU samples: work | % of all CPU samples: spinning |
|---|---|---|---|---|
| [leaf] ggml: bf16 dot product (MUL_MAT_ID experts on CPU) | 36.8 | 32.2 | 48.9 | 0.0 |
| concat | 26.2 | 1.8 | 1.1 | 0.0 |
| ZenDNN matmul (AOCL-DLP GEMM kernels) | 10.1 | 27.3 | 12.6 | 0.0 |
| flash_attn_ext | 4.3 | 31.2 | 5.7 | 0.0 |
| (ggml barrier: waiting for the slowest thread) | 3.3 | 23.4 | 0.0 | 6.9 |
| [leaf] ggml: graph loop (ggml_barrier, inlined small ops) | 3.2 | 31.9 | 3.6 | 0.0 |
| gated_delta_net | 3.1 | 29.3 | 4.2 | 0.0 |
| [leaf] libc (memcpy / memset) | 2.7 | 29.9 | 3.5 | 0.0 |
| [leaf] ggml: type conversion | 1.8 | 28.5 | 1.9 | 0.0 |
| [leaf] ggml: f32/f16 dot product | 1.7 | 29.2 | 2.2 | 0.0 |
| ssm_conv | 1.5 | 17.0 | 2.0 | 0.0 |
| mul | 0.9 | 28.4 | 1.2 | 0.0 |
| rms_norm_mul | 0.8 | 27.8 | 1.1 | 0.0 |
| glu | 0.7 | 30.4 | 0.8 | 0.0 |
| unary | 0.6 | 26.2 | 0.7 | 0.0 |
| rms_norm | 0.6 | 24.2 | 0.4 | 0.0 |
| [leaf] ggml: other ops (norms, elementwise, softmax, ...) | 0.4 | 30.9 | 0.5 | 0.0 |
| add | 0.4 | 30.0 | 0.5 | 0.0 |
| argsort | 0.2 | 31.8 | 0.2 | 0.0 |
| get_rows | 0.2 | 15.1 | 0.0 | 0.0 |
| [leaf] other | 0.2 | 17.7 | 0.1 | 0.0 |
| (OpenMP: idle / fork-join) | 0.1 | 25.0 | 0.0 | 1.3 |
| [leaf] ggml-base: other | 0.1 | 18.6 | 0.0 | 0.0 |
| sigmoid | 0.1 | 25.4 | 0.1 | 0.0 |
| [leaf] ggml: tinyBLAS/llamafile GEMM (dense matmul on CPU) | 0.1 | 28.5 | 0.1 | 0.0 |
| scale | 0.0 | 26.0 | 0.2 | 0.0 |
| mul_mat | 0.0 | 23.5 | 0.0 | 0.1 |
| soft_max | 0.0 | 30.0 | 0.0 | 0.0 |

'% of wall time' is the main thread's samples. 'threads doing work' counts, per main-thread sample in that op, the samples of all threads within 6 ms that were not spinning in libomp.

### Threads: 32, work share min/median/max = 90/92/96% (the rest is spinning in libomp)

