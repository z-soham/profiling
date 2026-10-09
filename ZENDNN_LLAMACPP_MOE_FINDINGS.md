# ZenDNN llama.cpp vs CPU llama.cpp vs vLLM — Qwen3.6-35B-A3B BF16 prefill

Written 2026-10-07. Everything below was measured on 32-core Turin pods (EPYC 9755, boost off, ~2.7 GHz, 1 NUMA node of
memory, 256 GiB limit), model `Qwen3.6-35B-A3B-BF16.gguf` (66 GiB), BF16, prefill only. Statements are labelled
**measured** (from a run listed in section 3), **from source/docs** (read in the code, not run), or **inferred**
(my interpretation; not directly tested).

## 1. Summary

1. **Removing the 32-expert limit (and the "rows per expert <= 32" rule that sits next to it) works as intended: the
   expert matmuls (`MUL_MAT_ID`) run on ZenDNN.** The CPU `ggml_vec_dot_bf16` falls from 43% of cycles to 0.1%.
2. **It does not make prefill faster, because ZenDNN's matmul is not faster than ggml's own BF16 CPU kernels for the
   shapes llama.cpp gives it.** At ub512 each expert sees only ~25 rows; ZenDNN's sequential scheduler (`ALGO_1`)
   runs ~166 tiny GEMMs one after another, each forked across 32 threads. The gate/up projections come out 38% slower
   than stock's CPU kernels, the down projection is about equal (section 5).
3. **ZenDNN does win where the matmuls are big enough**: dense matmuls take 28–36% less time than stock, and the MoE
   matmuls are 17% faster than stock at ub4096 (≈2.4 TFLOP/s, close to vLLM's ≈2.7).
4. **Larger ubatch (ub4096) does not turn that into a win, because ggml's single-threaded `CONCAT` of the Gated-Delta-Net
   conv state gets 2.6× more expensive** (2.8 s → 7.1 s per 4096 tokens) and eats the whole gain.
5. **My first ZenDNN measurements were taken during ZenDNN's warm-up ramp** (default ZenDNN needs ~6 passes to reach
   steady state), which understated it by ~2–5%. The ramp, the +54 GB of memory and the steady page faults have one
   cause (verified in ZenDNN's source): under the default AUTO scheduler ZenDNN also builds the *decode* custom-kernel
   pack of every expert, lazily as experts first fire. Pinning `ZENDNNL_GRP_MATMUL_ALGO=1` removes all three
   (65 GB, ~0 faults); with it ZenDNN is at parity with stock (291–293 vs 293 t/s at pp8192/ub512), and
   `CROSS_WARM=0` reaches 297.
6. **vLLM is still ~2.1× faster (631 t/s)**: one fused MoE call (gate+up+activation+down), ~128 rows/expert from 4096-token
   chunks, a native threaded GDN, no serial CONCAT, and a single copy of the weights.
7. **Fixes that would help (not applied, section 8):** thread the CONCAT; call ZenDNN's fused-MoE API from ggml-zendnn;
   make the offload decision depend on rows-per-expert at the actual ubatch.

## 2. Headline results

pp8192, BF16, prefill t/s, steady state (median over passes after warm-up), unless noted.

| configuration | ub512 | ub4096 | resident memory |
|---|---|---|---|
| llama.cpp CPU (stock) | 293 | 244 | 65 GB |
| llama.cpp ZenDNN, default env | 291 (ramps 278→292) | 278 | 119 GB |
| llama.cpp ZenDNN, `ZENDNNL_GRP_MATMUL_CROSS_WARM=0` | **297** | 279 | 121.7 GB |
| llama.cpp ZenDNN, pinned `ZENDNNL_GRP_MATMUL_ALGO=1` | 291–293 | 279 | **65 GB** |
| llama.cpp ZenDNN, `GRP_MATMUL_AUTO_PROMPT_ALGO=3` | 265–272 | 240 | 119 GB |
| llama.cpp ZenDNN, `GRP_MATMUL_AUTO_PROMPT_ALGO=5` | 268–275 | 251 | 119 GB |
| vLLM + zentorch (4096-token chunks) | — | **631** | n/a |

Source: `investigate_20261007/ub_*.err` (llama.cpp, pass timestamps) and
`perf_qwen36_bf16_20261007_0922/analysis.md` (vLLM). The earlier perf-window run (same pod) measured stock 291, ZenDNN
288, ZenDNN+ALGO3 270, vLLM 631 t/s; the ZenDNN rows there are 2–3 passes taken in the warm-up ramp.

Earlier run on pod `turin-xcovoid0021-pod-1` (`perf_qwen36_bf16_20261007_0531`, before the second rule was removed):
default ZenDNN 280 t/s with experts still on CPU (the rows-per-expert rule rejected them); with
`GGML_ZENDNN_ADAPTIVE_FALLBACK=0` (experts on ZenDNN) 267 t/s at ub512 and 269 at ub4096 (2 passes each, still ramping).

## 3. Tests conducted

All llama.cpp runs: `llama-bench -t 32 -b 4096 -fa on -ctk f16 -ctv f16 -lm dio` pinned with
`numactl --physcpubind=<pod cores> --membind=<pod node>`; env `LD_PRELOAD=tcmalloc:libomp.so.5`,
`KMP_AFFINITY=granularity=fine,compact,1,0`, `KMP_BLOCKTIME=1`, `OMP_WAIT_POLICY=ACTIVE`, `OMP_DYNAMIC=FALSE`
(the "optimised commands" of `Benchmarking/turin-results-2026-09-30.zip`); ZenDNN runs add `ZENDNNL_MATMUL_ALGO=1`.
Random-token prompts (what `llama-bench` and `vllm bench latency` use). Scripts are in `/proj/aigstaff/sohroy/profiling`.

| # | test | pod | scripts | results |
|---|---|---|---|---|
| T1 | Source: where the limits are, how `MUL_MAT_ID` is offloaded | – | – | `ggml/src/ggml-zendnn/ggml-zendnn.cpp` |
| T2 | perf windows, ZenDNN with limit removed (default / fb0 / fb0+ub4096), pp8192 | xcovoid0021-pod-1 | `perf_llama_latest.sh` | `perf_qwen36_bf16_20261007_0531/` |
| T3 | perf windows: stock vs ZenDNN vs ZenDNN+ALGO3 vs vLLM, pp8192 | xcovoid0023-pod-6 | `perf_llama_latest.sh` | `perf_qwen36_bf16_20261007_0922/` |
| T4 | vLLM/zentorch source reading + torch-profiler per-op split | – | – | `vllm_profile_pp8192/profiler_out_0.txt` |
| T5 | Stack-sample timeline (who spins while whom computes) | – | `timeline.py` | `latest_zendnn_pp8192_ub512/stacks_script.txt.gz` |
| T6 | Memory/page-fault sampling per ZenDNN cache setting (pp2048, ub512, 10 passes) | pod-6 | `investigate.sh` (M), `investigate2.sh` | `investigate_20261007/mem_*.{csv,err}` |
| T7 | ub512 vs ub4096 at pp8192 for stock / ZenDNN / ALGO3 / ALGO5 (3 passes) | pod-6 | `investigate.sh` (U) | `investigate_20261007/ub_latest_*.err` |
| T8 | ZenDNNL API log (chosen scheduler, num_ops, max_M, prepack state) at pp4096 ub512/4096 | pod-6 | `investigate.sh` (Z) | `zendnnl_latest_zendnn_pp4096_ub*.log.gz` |
| T9 | Per-ggml-op wall time (`opprof`, scheduler eval callback), pp4096 ub512/4096, stock & ZenDNN | pod-6 | `investigate.sh` (O), `opprof.cpp` | `opprof_latest_*_pp4096_ub*.{log,tsv}` |
| T10 | `CROSS_WARM=0` / `PREPACK=0` memory + throughput (pp2048) | pod-6 | `investigate2.sh` | `mem_crosswarm0*.{csv,err}` |
| T11 | Steady-state ub512/ub4096, `CROSS_WARM=0` vs default, 6 passes | pod-6 | `investigate3.sh` | `ub_zendnn_cw0.err`, `ub_zendnn_default_r6.err` |
| T-pf | perf `page-faults` with call graph + data addresses, steady state (pass ≥12), 10 s | pod-6 | `pf_trace.sh` | `investigate_20261007/pf/latest_zendnn/` |
| T-long | 100-pass RSS / fault-rate sampling, default env | pod-6 | `investigate4.sh` | `long_default.mem.csv` |
| T12 | `CUSTOM_KERNEL=0` and pinned `ZENDNNL_GRP_MATMUL_ALGO=1`: memory, faults, throughput (pp2048 and pp8192) | pod-6 | `investigate5.sh`, `investigate6.sh` | `mem_ck0.*`, `ub_zendnn_ck0.err`, `mem_pinalgo1.*`, `ub_zendnn_pinalgo1.err` |

Perf-window method (T2/T3): after the benchmark's warm-up pass completes (fix: previously the windows started when the
OpenMP pool was created, which could capture weight packing), 10 s settle, then 15 s windows one at a time with
`perf record` (cycles; cpu-clock + frame-pointer stacks) and `perf stat` (basic, top-down, FP ops, cache fill sources, L2).
Only the core PMU is exposed in the pod. DRAM traffic is an estimate from L1-fill and L2-prefetch events.

Builds: llama.cpp `78651c4` (latest master at the time) built natively in the pod, `build/` (stock) and `build_zendnn/`
(`-DGGML_ZENDNN=ON`) against ZenDNN master `e3f4c4dd` (6.1.0_rc3, built with cmake 3.31.6 because the host's 3.26 cannot
evaluate `$<COMPILE_ONLY:>`). vLLM: sacsharm's existing venv (vLLM 0.28.1rc1 CPU, zentorch 2.13.0.0 = 2026-WW30),
used as-is. Exactly one source change: `ggml-zendnn.cpp` `supports_op` (section 4).

## 4. The code change

In `ggml_backend_zendnn_device_supports_op`, `GGML_OP_MUL_MAT_ID`, inside the adaptive-fallback block (default on):

```cpp
// removed (rule 1): max_experts = 32; if (n_experts > max_experts) return false;
// removed (rule 2): if (N / n_experts <= 32) return false;   // N = tokens*n_expert_used
```

Rule 2 matters for Qwen3.6 (256 experts, top-8): at ub512, N = 512×8 = 4096 → 16 rows/expert, so rule 2 alone kept the
experts on CPU after rule 1 was gone (result in section 2, "280 t/s, experts still on CPU"). The remaining generic
rule (`K <= 256 || N <= 128 || M <= 96` → CPU) is unchanged. The edit is uncommitted in
`/proj/aigstaff/sohroy/llama.cpp`.

## 5. Why ZenDNN llama.cpp is not faster than CPU llama.cpp

### 5.1 Per-op accounting (T9, one pass of 4096 tokens, `opprof`, ms)

| op | stock ub512 | ZenDNN ub512 | stock ub4096 | ZenDNN ub4096 |
|---|---|---|---|---|
| MoE experts (gate+up+down) | 5,795 | **7,028** | 4,022 | **3,352** |
|   gate / up / down | 1752 / 1725 / 2318 | 2416 / 2401 / 2210 | 1216 / 1223 / 1582 | 1106 / 1118 / 1128 |
| CONCAT (GDN conv state, 1 thread) | 2,739 | 2,787 | 7,280 | 7,102 |
| dense `MUL_MAT` | 2,243 | 1,621 | 2,144 | 1,381 |
| GATED_DELTA_NET | 798 | 804 | 809 | 789 |
| FLASH_ATTN_EXT | 633 | 617 | 726 | 707 |
| sum of node times | 14,196 | 14,949 | 16,974 | 15,332 |

(opprof's "clean" walls were 308 t/s stock vs 266 t/s ZenDNN at ub512 — that ZenDNN run had only 1 warm-up pass, so
it includes the ramp from 5.5; use the steady-state numbers in section 2.)

At ub512, ZenDNN saves ≈620 ms on dense matmuls but loses ≈1,230 ms on the MoE ops, a net loss. The rest of the graph
(CONCAT, GDN, attention, norms, elementwise) is the same in both builds, about 55% of the pass, so even a large
difference in the matmuls moves the total only a little.

### 5.2 Why the MoE matmuls are slow at ub512

* **Rows per expert (from source arithmetic + T8 log).** A ubatch of T tokens makes T×8 expert assignments over 256
  experts: 16 rows/expert at ub512, 128 at ub4096. The ZenDNN log shows ~166 active experts per call at ub512 (avg
  ~25 rows each) and ~214 at ub4096.
* **Skew (T8, measured).** Average of the per-call *maximum* rows on one expert: 464 of 512 at ub512, 3,699 of 4,096 at
  ub4096. (Random token ids route very unevenly; real text routes differently. vLLM's benchmark uses the same random
  tokens, so the comparison is like for like, but the absolute numbers for real prompts may differ.)
* **Scheduler (T8, measured).** Every group-matmul call chose `ALGO_1`, reason `auto_rule07_prompt_seq`: in
  ZenDNN's auto-selection a prompt batch (`max_M > 32`) runs experts sequentially, each GEMM across all 32 threads
  (docs: `lowoha_group_matmul_operator.md`, "Whole-call selection").
* **Consequence (T5, measured).** For the ZenDNN build, 20% of the main thread's samples are in `GOMP_parallel_end` (the
  join barrier at the end of a ZenDNN parallel region) vs 3.8% for stock. While the main thread is inside a ZenDNN
  matmul, ≈18 of ≈29 sampled other threads are working and ≈11 are spinning in libomp. That is the load-imbalance /
  fork-join overhead of running ~166 small GEMMs × 3 ops × 39 layers per ubatch, each split 32 ways.
* **Shape sensitivity (T9, measured).** At ub512 the gate/up projections (512 output columns, K=2048) take
  2,416/2,401 ms against 1,752/1,725 for stock (+38%); the down projection (2,048 columns, K=512) takes 2,210 vs 2,318
  (−5%). **Inferred:** the 512-column projections give the GEMM too few column tiles to occupy 32 threads. I did not
  test this directly.
* **With more rows it goes away (T9, measured).** At ub4096 gate/up/down are 1,106/1,118/1,128 ms, i.e. equal, and
  17% faster than stock. Achieved rate per call: gate at ub512 ≈ 1.1 TFLOP/s (7.7 ms for 8.6 GFLOP), at ub4096
  ≈ 2.4 TFLOP/s (28.4 ms for 68.7 GFLOP); stock CPU kernels ≈ 1.5 and ≈ 2.2. vLLM's fused call ≈ 2.7 TFLOP/s
  (206 GFLOP in 76.7 ms per layer per 4096-token chunk). Peak BF16 for 32 Zen-5 cores at 2.7 GHz is roughly
  11 TFLOP/s (my estimate), so all of these run at 10–25% of peak.

What "spinning" means: a thread that has no work left polls in a loop (`OMP_WAIT_POLICY=ACTIVE`) instead of sleeping,
so it shows up as CPU time in `libomp`. At a parallel region's end every thread waits for the slowest one.

### 5.3 Why ub4096 does not rescue it

At ub4096 both MoE and dense matmuls get faster for ZenDNN (T9), yet the pass is no faster
(278 vs 291 t/s, section 2): `CONCAT` goes from 2.8 s to 7.1 s for the same 4096 tokens (T9, measured, in both stock and
ZenDNN). `CONCAT` (conv-state concatenation in the Gated Delta Net layers) is threaded over `ne2`, which is 1 for a
single sequence, so one thread does all of it (verified in `ops.cpp` `ggml_compute_forward_concat_f32`: loops `i2 = ith; i2 < ne2; i2 += nth`, with the source comment `// TODO: smarter multi-theading`). It copies one float per iteration with a 4-index address calculation and a branch per element. ≈11.6 ms/call at ub512 vs ≈237 ms/call at ub4096 for ~16.8 MB vs ~134 MB (≈1.4 vs ≈0.6 GB/s, far below memory bandwidth; the scalar strided loop explains the low rate, the superlinear growth with ub is **inferred** from the working set outgrowing cache). Stock also drops
from 293 to 244 t/s at ub4096 for the same reason. Net: ZenDNN beats stock by 14% at ub4096 but is still 5% below its
own ub512 result.

**Estimate, not a measurement:** if CONCAT stayed at its ub512 cost (~2.8 s) at ub4096, ZenDNN's pass would be about
10.9 s instead of 15.3 s, i.e. ≈370 t/s at pp4096.

### 5.4 Memory, page faults and the warm-up ramp (cause verified in the ZenDNN source)

**What a page fault is.** The first access to a 4 KiB page of memory; the kernel then maps a physical page (a *minor*
fault; *major* = needs disk I/O, 0 here). Loading the 65 GB model costs ≈17.3M faults in every build.

**Why only ZenDNN keeps faulting after load.** ggml's CPU kernels read the weights in place in the GGUF layout
(`ggml-cpu.c` has no BF16 repack; `ggml_vec_dot_bf16` / tinyBLAS take the loaded rows as they are), and the graph work
buffers are allocated once. ZenDNN's kernels need the weights in their own packed layouts and cache them.

**The mechanism (ZenDNN `group_matmul_dispatch.cpp` ≈ lines 1250–1460, `prepack/`):**
* With the scheduler on AUTO (`ZENDNNL_GRP_MATMUL_ALGO` unset) and weight cache mode 2 (the default),
  ZenDNN uses a "mixed in-place" mode: the prompt-phase AOCL full-weight reorder rewrites the weight buffer **in place**,
  and the **decode-phase custom-kernel (CK) pack is built out-of-place** from the raw weights before that mutation
  ("cross-warm"). The code comment states the cost: "one extra out-of-place copy of every in-place-eligible weight".
  This mode needs PREPACK on, CROSS_WARM on, CK on and an unlimited LRU. If any is off, ZenDNN downgrades the process to
  out-of-place mode (`set_weight_cache(1)`), which also makes a full extra AOCL copy.
* **A prefill-only run never uses the decode pack** (ALGO_1 runs the AOCL path), yet it pays for it: 26,142 CK pack
  misses × ≈2 MB ≈ 52 GB in one 4096-token pass of the log, matching the +54 GB resident (T6/T8).
* Packing is **lazy per expert**: the prepack warms only the experts that fire in the call (docs: "pre-warms the firing
  experts"; `params[0].active_matmul / total_matmul` in `lowoha_matmul.hpp` is how a caller can pass all experts, and
  ggml-zendnn does not). Experts that are rarely routed first appear late, each triggering a pack into a fresh heap
  buffer. perf shows 86% of the steady-state faults at one instruction of a JIT kernel writing fresh `[heap]` pages,
  each page touched once (T-pf, `investigate_20261007/pf/`). That tail is also the throughput ramp.
* **It decays** (long run, 100 passes, default env): fault rate 245k/s (loading) → 7.9k/s at 3–4 min → 0.2–0.9k/s after
  10 min; RSS saturates at ≈122.5 GB.

**Measured effect of each setting** (pp2048/ub512 unless noted; resident GB; steady t/s; first-pass t/s):

| setting | resident | steady faults | first pass | steady | note |
|---|---|---|---|---|---|
| stock CPU | 65 | 0 | 311 | 315 | no repack |
| ZenDNN default | 119–122.5 | decaying, 7.9k/s→0.3k/s | 43 | ~304 (6+ passes) | mixed in-place + CK decode pack |
| `CROSS_WARM=0` | 121.7 | decaying | 199 | 315 | downgrade → out-of-place AOCL pack (same size), no decode warm |
| `PREPACK=0` | 121.7 | decaying | 196 | 309 | |
| `CUSTOM_KERNEL=0` | **177** | decaying | 135 | ~302 | downgrade + decode regime copy: worse |
| `WEIGHT_CACHE=1` | 176 | decaying | 44 | ~304 | all out-of-place |
| `WEIGHT_CACHE=0` | 65 | ~2.6k/s | 124 | 124 | repack every call |
| **`ZENDNNL_GRP_MATMUL_ALGO=1` (pinned)** | **65.08** | **≈0 (8/s)** | 174 | **313–317** | single in-place AOCL layout, no CK pack |

Pinned ALGO is the fix because the dispatcher treats a pinned generic ALGO as "one layout per weight buffer": WC=2
stays in place, no CK decode pack, no cross-warm (`group_matmul_dispatch.cpp` comment "PINNED algos keep a single layout
per weight and so KEEP WC=2"; cross-warm skips a pinned ALGO). It is the same scheduler the AUTO rule already picks for
prompts (`auto_rule07_prompt_seq` → ALGO 1), so compute is unchanged: pp8192 ub512 291–293 t/s, ub4096 279 t/s
(T12). The trade-off: decode then also runs ALGO 1 instead of the N-tile/CK decode kernel — not measured, so for
mixed prefill+decode use, measure decode first.

A small steady fault floor remains with the weight cache off (≈2.6k/s): the backend's own per-call vectors/buffers
(~14 `std::vector`s in `ggml_zendnn_group_matmul` plus the per-expert row lists in `mul_mat_id`); not traced further.

### 5.5 Reasons, in order of size

1. The matmuls themselves are no faster than ggml's CPU kernels for ~25-row experts (they are bandwidth-bound and the
   scheduler splits them badly); ZenDNN's gain on dense matmuls is cancelled by its loss on gate/up.
2. Half the pass is unaffected by the backend (CONCAT ≈19%, GDN, attention, norms, elementwise).
3. ZenDNN adds its own costs: expert gather/scatter (≈5% of wall), cache packing, a warm-up ramp, +54 GB of memory.
4. Measurement artefact in the first runs: ZenDNN was profiled during its warm-up ramp.

## 6. How vLLM does it (T4, source + profile)

* `ZenCpuPlatform` selects zentorch; dense linears go through `zentorch_linear_unary` with weights prepacked at load
  (`VLLM_ZENTORCH_WEIGHT_PREPACK=1`), 18% of the pass, 4.4 ms/call.
* MoE: one `torch.ops.zentorch.zentorch_fused_moe` call (`cpu_moe.py:407`) = gate+up with fused gated activation, then
  down (ZenDNNL `group_matmul_direct` in parallel mode with internal-alloc scratch buffers; strings in `libzentorch.so`;
  the ZenDNN docs say V1 runs it as two internal passes). 44.5% of the pass, 76.7 ms per layer per 4096-token chunk.
* Attention: vLLM's own `cpu_attention_with_kv_cache` (24.6%). GDN: native `chunk_gated_delta_rule_cpu` (2.8%) and
  `causal_conv1d_fwd_cpu` (0.7%); `aten::cat` only 0.9%. Elementwise ops fused by `torch.compile`.
* Counters (T3): IPC 2.19 vs 1.64–1.74, backend-memory stalls 34% vs 55–61%, ≈3.6 vs ≈1.6 T FP ops/s,
  DRAM-sourced L1 fills 11.8 vs 19–34 GB/s. **Inferred:** expert weights are reused across ~128 rows instead of being
  re-streamed per small slice.

## 7. Open points / limits of this analysis

* Passes per configuration are few (2–6); differences under ~3% (e.g. 297 vs 293) are within noise.
* All runs use random token ids; routing skew with real text will differ (`opprof -f` supports text prompts, not run).
* The "too few column tiles" explanation for slow gate/up is not isolated (inferred only). The memory, ramp and page-fault
  cause is verified (section 5.4); the small fault floor with the cache off (≈2.6k/s) is not traced.
* `CROSS_WARM=0` and pinned `ALGO=1` were only tested on prefill; both drop the decode-regime pack, so decode is likely
  slower (ALGO 1 vs the N-tile/CK decode kernel). Not measured.
* Perf counter windows (T3) are on default-env ZenDNN, i.e. include the warm-up ramp.
* The old `analysis.md` header text still mentions "sacsharm's existing builds"; the runs used the new builds.

## 8. Proposed fixes (none applied except section 4)

**A. Runtime settings, no code**
1. **`ZENDNNL_GRP_MATMUL_ALGO=1` (pin the scheduler) for prefill-only use**: 65 GB resident (no extra copy), no
   steady-state page faults, a short ramp (2nd pass already 300 t/s at pp2048), 313–317 t/s steady at pp2048 and
   291–293 at pp8192/ub512, 279 at ub4096. Verified against the ZenDNN dispatcher source (section 5.4). Check decode
   before using it for mixed workloads (not measured).
   `ZENDNNL_GRP_MATMUL_CROSS_WARM=0` is the weaker alternative: faster start and 297 t/s at pp8192/ub512, but the same
   +55 GB (out-of-place AOCL pack instead of the CK pack). Keep ub512 until C is done.
2. Do not use `GRP_MATMUL_AUTO_PROMPT_ALGO=3/5` here (both slower in T3/T7).
3. Benchmark ZenDNN with ≥6 passes after warm-up.

**B. Offload decision (small code change in `supports_op`)** — make it depend on rows per expert at the *actual* ubatch,
not a fixed expert count. From this data, ZenDNN's MoE path is slower than CPU below roughly 25 rows/expert and faster
from ≈128 (ub4096). A threshold near 64–100 rows/expert would keep experts on CPU at ub512 and move them to ZenDNN at
large ubatches, while dense matmuls always use ZenDNN (they win at both sizes). The cut-off needs a sweep to set.

**C. Thread the CONCAT (ggml-cpu `ggml_compute_forward_concat`)** — highest value. Split the work over rows/elements
instead of `ne2` so all 32 threads copy. It is ≈19% of the ub512 pass and 46% at ub4096. Expected: ub512 ≈ +15–20%
for both builds (if CONCAT time scaled down ~linearly, my estimate); ub4096 ZenDNN ≈ 370 t/s (estimate in 5.3).
This also helps stock and every GDN model.

**D. Use ZenDNN's fused-MoE API from `ggml-zendnn`** — one `group_matmul_direct` call with `fused_moe` (gate+up, gated
activation, down) per layer instead of three gather→GEMM→scatter calls. Needs a graph-level fusion pattern (ggml
fuses neither `MUL_MAT_ID`+GLU+`MUL_MAT_ID`). Cuts expert gather/scatter (≈5% of wall) and the two extra passes over
activations, and mirrors what makes vLLM fast.

**E. Scheduler/packing in ZenDNN or its backend glue**
* Prompt-phase default `ALGO_1` runs hundreds of tiny GEMMs with a 32-thread fork each; an expert-parallel schedule
  with load balancing across the skew would remove most of the barrier waiting (ALGO 5 did not help, but its test had
  2 passes and the ub512 shapes; worth re-testing together with D).
* Second packed copy (+54 GB), ramp and steady faults: avoided at run time by pinning `ZENDNNL_GRP_MATMUL_ALGO=1`
  (A1). In code, ggml-zendnn could pass all 256 experts' weights on the first call through ZenDNN's
  `params[0].active_matmul / total_matmul` "prepack-extras" tail so everything is packed once at start-up. (The
  patched tree `/proj/aigstaff/sohroy/llama.cpp-turin` from 2026-09-30 contained such a change, `GGML_ZENDNN_MOE_PREPACK_ALL`;
  that was my own patch, not an upstream feature, and is not in the current tree.)
* Hoist the per-call vector/buffer allocations in `ggml_zendnn_group_matmul` (about 15 `std::vector`s per call) and
  reuse the work buffer to remove the steady-state page faults.

**F. Re-measure after C and D** with ≥6 passes at ub512/1024/2048/4096, stock vs ZenDNN, and a real-text prompt.

## 9. File index

* Results: `perf_qwen36_bf16_20261007_0922/` (T3, `analysis.md`), `perf_qwen36_bf16_20261007_0531/` (T2),
  `investigate_20261007/` (T6–T11).
* Scripts: `perf_llama_latest.sh`, `diag_lib.sh` (configs `latest_stock`, `latest_zendnn`, `latest_zendnn_algo3/5`,
  `latest_zendnn_fb0`), `investigate.sh`, `investigate2.sh`, `investigate3.sh`, `timeline.py`, `moe_algo_sweep.sh`
  (written, not run), `build_llama_latest_zendnn.sh`, `build_llama_venice.sh` (started, abandoned, no results),
  `opprof.cpp`.
* Earlier work (2026-09-29/30, older builds): `FINDINGS.md`, `PERF_ANALYSIS.md`.
* ZenDNN source/docs used: `/proj/aigstaff/sohroy/ZenDNN-latest/docs/operator/low_overhead_operator/lowoha_group_matmul_operator.md`,
  `docs/runtime_env.md`.

## 10. Verification of the findings against code (2026-10-07)

| claim | verified in | result |
|---|---|---|
| CONCAT is threaded over `ne2` only; scalar per-element copy | `llama.cpp/ggml/src/ggml-cpu/ops.cpp` `ggml_compute_forward_concat_f32` (`// TODO: smarter multi-theading`) | confirmed |
| CPU BF16 matmuls read weights as loaded, no repack | `ggml-cpu/` (no BF16 entry in `repack.cpp`; `ggml_vec_dot_bf16`/tinyBLAS in place) | confirmed |
| ggml-zendnn declares weights constant, no all-expert prepack | `ggml-zendnn.cpp` `is_wei_const(n_experts, true)`; no `active_matmul/total_matmul` use | confirmed |
| Prompt batches use sequential `ALGO_1` | `group_matmul_dispatch.cpp:1012` `pick(1, "auto_rule07_prompt_seq", 1)` (non-W4A8) | confirmed |
| Default AUTO + WC=2 builds an out-of-place CK decode pack, in-place AOCL prompt pack | `group_matmul_dispatch.cpp` ≈1250–1330 "(A) MIXED in-place" | confirmed |
| Otherwise ZenDNN downgrades to out-of-place (extra copy) | same file, "(B) DOWNGRADE", `set_weight_cache(1)` | confirmed; matches CROSS_WARM=0 / PREPACK=0 / CK=0 memory |
| Pinned ALGO keeps one in-place layout (no CK pack) | same file, "PINNED algos keep a single layout per weight and so KEEP WC=2" | confirmed by measurement (65.08 GB, 8 faults/s) |
| Prepack warms only the firing experts unless a tail is passed | `docs/.../lowoha_group_matmul_operator.md:159`, `lowoha_matmul.hpp` `active_matmul/total_matmul` | confirmed (docs/API); the lazy per-expert CK misses are in the T8 log |
| Earlier claim "a previous version of this backend already passed all experts" | it was my own 2026-09-30 patch in `llama.cpp-turin` (`GGML_ZENDNN_MOE_PREPACK_ALL`) | **corrected**: not an upstream feature |
| "tiny-tile / too few column tiles makes gate/up slow" | not found in code, not tested | remains an inference |
