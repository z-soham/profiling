# Qwen3.6-35B-A3B prefill: ZenDNN llama.cpp vs stock llama.cpp vs ZenDNN vLLM

Turin pod `turin-xcovoid0021-pod-1` (2x EPYC 9755; 32 cores 32-63, NUMA node 1), 2026-09-29.
Settings: `../Benchmarking/sweep.turin-32c-8b-newcmd.yaml`. BF16 weights.
Valid run: `qwen36_bf16_20260929_2252/` (its `summary.md` has every table).
`qwen36_bf16_20260929_2123/` is kept only as evidence of the NUMA problem below; its llama.cpp numbers are invalid.

## Questions

1. Why is ZenDNN llama.cpp barely faster than CPU (stock) llama.cpp on this MoE model?
2. Why is ZenDNN vLLM so much faster than ZenDNN llama.cpp?

## Results (prefill tokens/s)

| configuration | pp256 | pp8192 |
|---|---|---|
| llama.cpp CPU (stock build), `-ub 512` | 253 | 260 |
| llama.cpp ZenDNN, `-ub 512` | 252 | 249 |
| llama.cpp ZenDNN, experts forced onto ZenDNN (`GGML_ZENDNN_ADAPTIVE_FALLBACK=0`), `-ub 512` | 116 (152 once warmed up) | 297 |
| same, `-ub 4096` | 237 | 276 |
| vLLM + zentorch (`vllm bench latency`, batch 1, output 1) | **437** | **617** |

Run-to-run variation between processes is about 5-7%.

## Answer 1: why ZenDNN llama.cpp ≈ CPU llama.cpp

1. **The MoE expert matmuls never reach ZenDNN.** `ggml_backend_zendnn_device_supports_op`
   (`ggml/src/ggml-zendnn/ggml-zendnn.cpp`) rejects `MUL_MAT_ID` when the model has more than 32
   experts (`max_experts = 32`, "adaptive fallback"). Qwen3.6-35B-A3B has 256 experts (8 active).
   The GGML_SCHED_DEBUG placement confirms all 120 `MUL_MAT_ID` nodes per graph run on the CPU.
   These are 37-49% of prefill time.
2. **Most of the remaining time is in ops ZenDNN does not implement:** the conv-state `CONCAT` and
   the recurrence of the Gated-DeltaNet (linear attention) layers, flash attention, the
   attention output-gate `SIGMOID`, norms. ZenDNN only takes dense `MUL_MAT`s, about 12% of the
   time. It runs them ~1.4x faster, saving ~1 s of 33 s at pp8192: within noise.
3. **Forcing the experts onto ZenDNN does not fix it** (`GGML_ZENDNN_ADAPTIVE_FALLBACK=0`):
   - At `-ub 512` each expert gets ~16 rows per call (1 to 80; the routing is skewed). Every
     ZenDNNL grouped-matmul call is clamped to `flat_m_tile_seq_clamp` / exec_algo 1, i.e. experts
     processed one after another.
   - Result: 3x slower than the CPU kernel at pp256, only ~10% faster on the MoE part at pp8192.
   - ZenDNNL packs expert weights per shape on first use: the first pass runs at 52 t/s and the
     process needs ~100 GB extra memory (peak RSS 185 GB).
   - `ZENDNNL_GRP_MATMUL_ALGO` 1/3/5 do not beat the CPU kernel at pp256 (215 / 167 / 226 t/s
     after warm-up). ALGO 4 started 128 threads on 32 cores and made no progress in 11 minutes.

## Answer 2: why vLLM is ~2.4x faster at pp8192 (13.2 s vs 33.5 s)

Per-component time, pp8192 (llama.cpp from the per-op profiler `opprof`, vLLM from the torch profiler):

| component (seconds) | llama.cpp CPU `-ub 512` | llama.cpp ZenDNN `-ub 512` | llama.cpp ZenDNN, experts forced, `-ub 4096` | vLLM |
|---|---|---|---|---|
| GDN conv-state CONCAT | **8.3** | 7.2 | **14.7** | ~0.1 |
| MoE routed experts | **12.3** | 12.3 | 5.7 | 6.1 (zentorch fused MoE) |
| dense weight matmuls | 4.6 | 3.3 | 2.7 | 2.5 |
| GDN recurrence + conv1d | 2.1 | 2.0 | 2.0 | 0.6 |
| attention output-gate SIGMOID | 0.9 | 0.9 | 0.9 | fused into the compiled graph |
| full attention (10 layers) | 2.7 | 2.7 | 3.0 | 3.3 |
| everything else | 2.7 | 2.8 | 2.8 | ~0.8 |
| **total** | **33.5** | **31.1** | **31.8** | **13.2** |

- **Largest single cause (~40% of the gap) is a llama.cpp CPU inefficiency unrelated to ZenDNN.**
  `ggml_compute_forward_concat_f32` (`ggml-cpu/ops.cpp`, "TODO: smarter multi-threading") splits
  work over `ne2`, which is the number of sequences (1). The conv-state concat
  (`delta-net-base.cpp`, `conv_input`) therefore runs on one thread, with transposed strided reads.
  It takes 25% of prefill at `-ub 512` and 40-46% at `-ub 4096` (it gets worse with larger ubatches).
- **MoE is ~30% of the gap.** vLLM uses `zentorch::zentorch_fused_moe` (all experts, gate/up +
  activation + down in one op) on 4096-token chunks. At the same chunk size, ZenDNN's grouped
  matmul inside llama.cpp takes 5.7 s vs vLLM's 6.1 s: the kernel is not the gap once shapes
  match. llama.cpp cannot profit from `-ub 4096`, because the CONCAT cost grows faster than the
  MoE saving.
- Smaller contributors: GDN recurrence (2.1 vs 0.6 s), the output-gate SIGMOID (0.9 s), dense
  matmuls on stock (4.6 vs 2.5 s). llama.cpp's flash attention is slightly faster than vLLM's.
- **pp256 has the same pattern:** llama.cpp 1.13 s vs vLLM 0.61 s; CONCAT 0.29 s, MoE 0.48 vs 0.36 s.

## Other findings

- **mmap + host-wide page cache = remote weights.** The GGUF's page cache sat on NUMA node 5
  (other socket), left there by earlier pod-5 sweeps. Every mmap'd llama.cpp run on this pod
  (node 1) read its weights remotely: stock pp256 151 t/s with mmap vs 271 t/s with
  `--load-mode none` under `numactl --membind`. The script now uses `--load-mode none` and records
  each run's memory placement (`numa_*.txt`). This may also explain the unexplained 17-20% drop
  between runs noted in `Benchmarking/docs/turin-status.md` (not verified).
- **The online sweep understates vLLM at small prompts.** Offline vLLM beats llama.cpp at pp256
  (437 vs 253 t/s), while the online sweep had llama.cpp ahead (268 vs 85). vLLM's small-prompt
  sweep numbers look dominated by server/API overhead, not compute. Worth a separate look.
- ZenDNNL writes its warnings to stdout, so they end up inside llama-bench's JSON output.

## Suggested next steps (code changes, not done)

1. Multi-thread `ggml_compute_forward_concat_f32` over rows (`ne1`), or write the GDN conv state
   without a concat. Largest single win, for both builds.
2. For ZenDNN: raise/remove the 32-expert limit together with a fused MoE path (like zentorch's
   `zentorch_fused_moe`) and larger chunks for the expert GEMMs.
3. Fuse/multi-thread the attention output-gate SIGMOID.

## Caveats

- Both engines use random prompt tokens, which route unevenly across experts.
- vLLM uses 31 OpenMP threads (`VLLM_CPU_OMP_THREADS_BIND=32-62`), llama.cpp 32 (as in the user's commands).
- `opprof` computes one node at a time; its overhead is 1-7% (worst at pp256).
- `vllm bench latency` runs with `--no-enable-prefix-caching` (it reuses one prompt per iteration).

## Contents of this archive

| path | what |
|---|---|
| `FINDINGS.md` | this report |
| `qwen36_moe_diag.sh` | the diagnosis script (run inside the pod via `runas sohroy`) |
| `opprof.cpp` | per-ggml-op wall-time profiler for llama.cpp (perf is not installed in the pods) |
| `summarize.py` | builds `summary.md` from a run directory |
| `qwen36_bf16_20260929_2252/` | valid run: `summary.md`, llama-bench JSON, opprof TSVs, vLLM profiles, placement, NUMA, logs |
| `qwen36_bf16_20260929_2123/` | first run, invalid llama.cpp numbers (weights on remote NUMA node 5) |
