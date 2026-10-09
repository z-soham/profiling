# Qwen3.6-35B-A3B BF16 prefill: ZenDNN llama.cpp vs stock llama.cpp vs ZenDNN vLLM (perf analysis)

Pod turin-xcovoid0021-pod-1 (cores 32-63, NUMA node 1), 2026-09-30, **CPU boost off** (`cpufreq/boost=0`,
measured 2.72 GHz in every window). Only numbers from this run are used.

**Commands:** the optimised per-configuration commands of `Benchmarking/turin-results-2026-09-30.zip`
(`docs/turin-optimised-commands-2026-09-30.md`), on this pod's cores and node:
- llama.cpp, ZenDNN and stock (sacsharm's existing 8fe90e1fb builds, no code changes): **setup B**,
  `KMP_AFFINITY=granularity=fine,compact,1,0 KMP_BLOCKTIME=1 KMP_TPAUSE=0 KMP_*_BARRIER_PATTERN=dist,dist
  OMP_NUM_THREADS=32 OMP_DYNAMIC=FALSE OMP_WAIT_POLICY=ACTIVE` (+ `ZENDNNL_MATMUL_ALGO=1` for ZenDNN),
  tcmalloc + libomp.so.5 preloaded, `-t 32 -b 4096 -ub 512 -fa on -ctk f16 -ctv f16`, own GGUF copy.
- vLLM + zentorch: **setup C** (torch.compile, `VLLM_CPU_KVCACHE_SPACE=32`, `ZENDNNL_MATMUL_WEIGHT_CACHE=1`,
  `--max-num-seqs 1 --max-num-batched-tokens 4096 --dtype auto --kv-cache-dtype auto --language-model-only`).

Deviations: llama-bench instead of llama-server (no `-c 32000`); weights loaded with `--load-mode dio` so they
sit in local memory (65-68 GB on node 1, 0 elsewhere); `vllm bench latency` (batch 1, output 1, prefix
caching off). Raw data and every table: `perf_qwen36_bf16_20260930_0714/analysis.md`. Tools: `perf_llama.sh`
(root), `perf_analyze.py`, `diag_lib.sh`.

## Throughput (prefill tokens/s, measured while profiled)

| | pp256 | pp8192 |
|---|---|---|
| llama.cpp stock | 256 | 278 |
| llama.cpp ZenDNN | 273 | 284 |
| vLLM + zentorch | 462 | 632 |

## Where one prefill pass goes

llama.cpp: main-thread wall-clock samples (cpu-clock, frame-pointer stacks), attributed to the ggml op in the
stack (or the leaf function where the frame chain stops). "Threads" = threads doing work (not spinning or
asleep) while the main thread was in that op. vLLM: torch profiler, one generate call (self CPU time).

**pp8192**

| component | llama.cpp ZenDNN (28.8 s) | llama.cpp stock (29.5 s) | vLLM (13.0 s) |
|---|---|---|---|
| MoE routed experts | 11.1 s (38%, 32 threads) ggml CPU kernel | 11.1 s ggml CPU kernel | 6.1 s zentorch fused MoE |
| GDN conv-state concat | 6.7 s (23%, **2 threads**) | 6.3 s (21%, 2 threads) | 0.1 s (aten::cat) |
| dense weight matmuls | 3.1 s (11%, 27 threads) ZenDNN | 4.2 s (14%) tinyBLAS | 2.4 s zentorch_linear |
| full attention (10 layers) | 1.6 s (6%, 31 threads) | 1.4 s | 3.2 s |
| GDN recurrence + conv1d | 1.2 s (4%) | 1.3 s | 0.5 s |
| barriers, elementwise, norms, copies | 5.1 s (18%) | 5.2 s | 0.7 s |

**pp256**

| component | llama.cpp ZenDNN (937 ms) | llama.cpp stock (1000 ms) | vLLM (581 ms) |
|---|---|---|---|
| MoE routed experts | 448 ms (48%, 32 threads) | 446 ms | 351 ms |
| GDN conv-state concat | 167 ms (18%, **3 threads**) | 197 ms | 2 ms |
| dense weight matmuls | 108 ms (12%) ZenDNN | 131 ms tinyBLAS | 107 ms |
| GDN recurrence + conv1d | 40 ms | 37 ms | 31 ms |
| full attention | 8 ms | 8 ms | 9 ms |
| barriers, elementwise, norms, copies | 164 ms | 180 ms | 80 ms |

## Hardware counters (15 s windows)

| run | cores busy | IPC | retiring | frontend | backend: cpu | backend: memory | FP ops/s | DRAM read (est.) |
|---|---|---|---|---|---|---|---|---|
| llama.cpp ZenDNN pp8192 | 25.7 | 1.90 | 21.8% | 8.7% | 15.9% | 52.7% | 1.66 T | 67 GB/s |
| llama.cpp stock pp8192 | 26.4 | 1.80 | 20.5% | 8.6% | 16.1% | 53.8% | 1.63 T | 73 GB/s |
| vLLM pp8192 | 24.9 | **2.20** | 29.5% | 8.1% | 28.5% | **33.6%** | **3.63 T** | 42 GB/s |
| llama.cpp ZenDNN pp256 | 27.5 | 1.49 | 17.6% | 8.9% | 13.6% | 59.3% | 1.40 T | 90 GB/s |
| llama.cpp stock pp256 | 26.8 | 1.41 | 16.6% | 8.1% | 13.7% | 60.8% | 1.32 T | 92 GB/s |
| vLLM pp256 | 31.9 | 1.07 | 13.7% | 11.2% | 4.6% | 70.3% | **2.36 T** | **117 GB/s** |

Clock 2.72 GHz in all windows. Top-down in % of dispatch slots. FP ops = retired FP operations
(`fp_ret_sse_avx_ops.all`). DRAM read = 64-byte lines from L1 fills sourced from DRAM plus L2 prefetches that
missed L3 (the pods expose no uncore/data-fabric counters); node 1's three DDR5 channels peak around 140 GB/s.
No fills came from the other socket's DRAM (0.0%).

Where the cycles go (flat cycles profile, all threads):
- llama.cpp ZenDNN pp8192: ggml bf16 dot product (experts) 45%, flash attention 14%, ZenDNN/AOCL GEMM 10%,
  OpenMP runtime 6.5%, other ggml ops 6%, GDN 5%.
- vLLM pp8192: ZenDNN/AOCL-DLP GEMM kernels (experts + linear, JIT code) 39%, vLLM attention micro-GEMM
  (`cpu_attention::TileGemm82<BFloat16>`) 32%, OpenMP (libiomp5) spin 14%.

## Findings

### 1. Why ZenDNN llama.cpp is barely faster than stock (+2% at pp8192, +7% at pp256)

- **The MoE experts, the largest cost, run on the same CPU kernel in both builds.** They take 38-48% of wall
  time (11.1 s per pp8192 pass in each build; `ggml_vec_dot_bf16` is 43-55% of all cycles). ggml-zendnn's
  `supports_op` sends MUL_MAT_ID to the CPU when the model has more than 32 experts; Qwen3.6 has 256.
- **ZenDNN takes only the dense weight matmuls, 11-14% of the time,** and runs them 1.2-1.4x faster than
  stock's tinyBLAS (3.1 s vs 4.2 s at pp8192, 108 vs 131 ms at pp256). That is the entire difference between
  the builds: about 1 s of 29 s at pp8192.
- **The rest, about half the time, is ops neither build accelerates**: the conv-state concat, attention, the
  GDN recurrence and the elementwise ops.

### 2. Why vLLM is 2.2x faster at pp8192 (13.0 s vs 28.8 s) and 1.7x at pp256 (581 vs 937 ms)

- **Experts: 5 s of the 15.8 s gap at pp8192.** zentorch's fused MoE (grouped AOCL-DLP GEMMs over 4096-token
  chunks) takes 6.1 s against 11.1 s for ggml's dot-product loop over 512-token ubatches. At pp256 the gap is
  smaller (351 vs 448 ms): both are streaming expert weights, vLLM at an estimated 117 GB/s against 90 GB/s.
- **The conv-state concat: 6.6 s of the gap at pp8192, 165 ms of the 356 ms gap at pp256.** In llama.cpp it
  runs on 2-3 threads while the others sleep at the next barrier (18-23% of wall time; under 1% of cycles,
  so a cycles profile alone hides it). vLLM's equivalent is a 0.1 s `aten::cat`.
- **Everything else: about 4.4 s at pp8192.** llama.cpp spends 5.1 s in barriers, norms, elementwise ops and
  copies, against 0.7 s in vLLM (fused by torch.compile). The GDN recurrence is 1.2 vs 0.5 s.
- **Where llama.cpp is faster: attention** (1.6 s vs 3.2 s at pp8192). vLLM's CPU attention kernel is 32% of
  its cycles; on Zen 5 it uses the generic "vec" path (its AMX path needs AMX).
- **In hardware terms:** vLLM retires 3.6 T FP ops/s against 1.66 T for llama.cpp, at a higher IPC (2.2 vs
  1.9), and is far less memory-bound (34% vs 53% of slots). llama.cpp's expert kernel re-reads the expert
  weights every 512-token ubatch (67 GB/s of estimated DRAM reads at pp8192, against 42 GB/s for vLLM).

### 3. Other observations

- The cores do not boost (2.72 GHz, the EPYC 9755 base clock, in every window), for both engines.
- With setup B's `KMP_BLOCKTIME=1`, llama.cpp's idle threads sleep instead of spinning (OpenMP runtime 6.5%
  of cycles). The single-threaded concat still costs the same wall time, because the other threads have
  nothing to do during it.

## AMD uProf

AMD uProf 5.0 (`/proj/zendnn/ganesh/Deepseek/AMDuProf_Nda_Linux_x64_5.0.1498`) does not run inside the pod:
`AMDuProfPcm` loops forever in `HWCpuTopology::PopulateApicList()` (a segfault that its own signal handler
catches and re-executes), even for `-n` (print topology). The pod's cpuset allows only cores 32-63 of the host's
256. Pinning uProf to core 63 (`taskset -c 63`) does not help: the discovery enumerates all host CPUs.
`AMDuProfCLI` additionally needs `libarchive.so.13`, which the pod image lacks. Running uProf would need its
process to see all host CPUs, i.e. to run outside the pod's pinned cpuset.

## Measurement notes

- The first three vLLM perf attempts measured vLLM's start-up (torch.compile) instead of the benchmark: the
  script waited for "Warming up", which vLLM also prints during compilation. They are kept in
  `perf_qwen36_bf16_20260930_0714/invalid_vllm_attempts_compile_phase/` and not used. The valid vLLM windows
  start once the timed "Bench iterations" run and sample the pod's cores (`perf -C 32-63`), which run only the
  benchmark.
- vLLM's per-op split (torch profiler) runs in a separate pass with the same command; its totals (13.0 s,
  581 ms) match the benchmark's own latencies (12.97 s, 0.56 s).
- DWARF unwinding returned empty stacks for ~96% of samples, so stacks use frame pointers. The bf16/fp32 FP-op
  umasks read 0 on this CPU; only total FP ops are reported.
- An earlier perf run with the compile-ub512 commands is superseded and kept only as
  `superseded_perf_20260930_0244_compile-ub512-cmds/`.
