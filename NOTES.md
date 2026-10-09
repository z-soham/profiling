# Working notes: Qwen3.6 / ZenDNN / llama.cpp optimisation

Lives in the private repo `z-soham/profiling` (this directory is now a git repo). Do not copy it into the llama.cpp fork or any PR.
Keep it updated as work continues. Last updated: 2026-10-08.
Plan for getting the changes upstream: `/proj/aigstaff/sohroy/llama.cpp/CONTRIB_PLAN.md` (local file in the llama.cpp checkout, not committed).

## 1. What was found and fixed (one-paragraph version)

Qwen3.6-35B-A3B BF16 prefill in llama.cpp was ~290 t/s with or without ZenDNN; vLLM+zentorch was ~2.1x faster.
Biggest cost: ggml-cpu `CONCAT` of the Gated-Delta-Net conv state ran on ONE thread (it threads over `ne2`, which is 1 for a
single sequence): 19% of the ub512 pass, 46% at ub4096. Threading it over all dst rows: ZenDNN pp8192 ub4096 277 -> 515 t/s
(+85%), ub512 280 -> 352 (+26%); stock CPU 246 -> 417 / 295 -> 366. Qwen's conv concat has a TRANSPOSED second input, so it uses
the strided fallback loop; the gain is from threading, not the memcpy fast path.
Second item: ggml-zendnn's `supports_op` kept MUL_MAT_ID (MoE experts) on CPU for >32 experts / <=32 rows per expert. Removing
that helps only with `ZENDNNL_GRP_MATMUL_ALGO=1` or ZenDNN v6.1.0_rc3 (see section 6); it is NOT upstream-ready.

## 2. Git state

Main checkout: `/proj/aigstaff/sohroy/llama.cpp` (remote `origin` = upstream ggml-org, remote `fork` = z-soham/llama.cpp).
Repo-local git identity: `Soham Roy <soham@zettabolt.com>` (global config untouched). GitHub account: `z-soham` (`gh auth switch`).
Fork (public): https://github.com/z-soham/llama.cpp. All commits are based on upstream master `3d65c90` (2026-10-08).

| branch (local + fork) | commits | content | worktree |
|---|---|---|---|
| `ggml-cpu-concat-threading` | 5e8779d, ffeb1d2 | CONCAT fix only (`ggml-cpu/ops.cpp`) | `llama.cpp-matrix/src_concat` |
| `ggml-zendnn-moe-offload-limits` | a545244 | ZenDNN limit removal only (`ggml-zendnn.cpp`) | `llama.cpp-matrix/src_zenlimits` |
| `zendnn-concat-fix` | 00a87cc, 107ab44, 982809b | both, combined | main checkout |

(Hashes of `zendnn-concat-fix` changed on rebase; check `git log` for the current ones. Older hashes in the report zips,
e.g. f012f20 / 4aa3f1c / 711391c, are pre-rebase / pre-email-rewrite.)
Commit messages are mine (AI-written, one has a `Co-Authored-By: Claude` trailer): REWRITE THEM YOURSELF before any PR.
Upstream policy (CONTRIBUTING.md): features start as an issue, one feature per PR, no AI-written posts/PR text/replies, disclose AI
use, `test-backend-ops` + `llama-perplexity` + `llama-bench` for operator changes, CPU first. CODEOWNERS for
`ggml/src/ggml-zendnn/`: @avinashcpandey @Jiten1parmar.
Not pushed to upstream; no PR opened. Compliance (AMD/Zettabolt approval) is still to be settled.

TODO before a CONCAT PR: search existing issues/PRs; whole-model equivalence (llama-perplexity before/after, expect identical);
a small public GDN model for repro; add a transposed-input case to `test_concat` in `tests/test-backend-ops.cpp` (existing cases
are contiguous-row views only; CPU is the reference backend, so it cannot catch CPU bugs); decide whether to keep the contiguous
memcpy fast path (Qwen never uses it); run `ci/run.sh`; open the issue (profile: CONCAT 19%/46%, single thread) first.
TODO before a ZenDNN PR: replace the fixed limit with a rows-per-expert threshold swept over ub 512/1024/2048/4096 and 2+ MoE
models; report the rc3 mmap segfault to the ZenDNN team; talk to the code owners.

Other repos: `z-soham/profiling` (private; this directory; remote `origin`, branch `main`); `Benchmarking` is a clone of
`sushant-zettabolt/Benchmarking` (public, not yours; z-soham has push access), my changes are committed and pushed on branch
`feat/sweep-reliability-turin` (commit b35397d); its untracked run logs and result zips were left out of that commit.

## 3. Machines and access

Pods (kubectl exec, user sohroy). `runas` refuses if another user holds the pod ("IN USE by uXXXX"): then use another pod.
```
export KUBECONFIG=/proj/zendnn/k8/dev-kubeconfig.yaml
/usr/local/bin/kubectl exec turin-xcovoid0023-pod-6 -n zendnn -- runas sohroy bash -c '<cmd>'
# long jobs: ... -- runas sohroy bash -c 'cd DIR && setsid nohup ./x.sh > log 2>&1 < /dev/null &'   (run the kubectl call in background; it blocks until the job ends)
```
- `turin-xcovoid0023-pod-6`: cpus 192-223, NUMA node 6 (used for everything since 2026-10-07).
- `turin-xcovoid0021-pod-1`: cpus 32-63, node 1 (older work). Sweep specs from earlier used pod-5: cpus 160-191, node 5.
- /tmp is NOT shared between host and pod: put scripts on NFS (`/proj/aigstaff/sohroy/profiling`). Output of piped commands in
  pods is buffered until exit. Builds MUST run in the pod (znver5 `-march=native`). Don't benchmark while a build runs.
- `perf`: `/proj/aigstaff/sohroy/tools/perf-7.0.0-30/perf`, root only (plain kubectl exec, no runas).
- Harness python: `/proj/aigstaff/sohroy/Benchmarking/.venv/bin/python` works only inside the pod.
- cmake: use `/proj/aigstaff/sohroy/tools/cmake-3.31.6-linux-x86_64/bin/cmake` (host's 3.26 is too old for ZenDNN); put its bin dir
  first on PATH when ZenDNN builds from source (its sub-build calls a bare `cmake`).

## 4. Builds (all in /proj/aigstaff/sohroy; build dirs are separate, nothing overwrites another)

llama.cpp-matrix (2026-10-08; script `llama.cpp-matrix/build_matrix.sh [names]`; none contain the CONCAT fix; base 78651c4):

| dir (`llama.cpp-matrix/build_*`) | source | ZenDNN | 32-expert limit |
|---|---|---|---|
| `build_nozen` | src_base (78651c4) | none | - |
| `build_zen_ww28` | src_base | ZenDNN-2026-WW28 (1f399a75; llama.cpp's pinned tag, built from source in `_deps/zendnn-prefix/build/install`) | present |
| `build_zen_ww28_nolimit` | src_nolimit (+limit removal) | same WW28 install | removed |
| `build_zen_latest` | src_base | v6.1.0_rc3 = zendnn-2026-WW40 (e3f4c4dd) at `ZenDNN-latest/build/install` | present |
| `build_zen_latest_nolimit` | src_nolimit | v6.1.0_rc3 | removed |

WW28 library: `llama.cpp-matrix/build_zen_ww28/_deps/zendnn-prefix/build/install/zendnnl/lib` (needed in LD_LIBRARY_PATH).
rc3 library: `ZenDNN-latest/build/install/zendnnl/lib`. ZenDNN WW28 source build takes ~73 min; llama.cpp itself ~10 min.
`build_matrix_rest.sh` = same, but `zen_ww28_nolimit` reuses the WW28 install (saves a rebuild).

llama.cpp (main checkout, 78651c4 era):
- `llama.cpp/build` = stock CPU master. `llama.cpp/build_zendnn` = ZenDNN rc3 + limit removal (pre-CONCAT-fix; the "before" builds).
- `llama.cpp/build_zendnn_concat` (ZenDNN rc3, with CONCAT fix + limit removal) and `llama.cpp/build_concat` (CPU only, with CONCAT fix):
  the "after" builds, built from clean commit 711391c (now rebased away; binaries still valid). Script: `profiling/build_concat_all.sh`
  (refuses a dirty tree). `llama.cpp/build_venice` is from an abandoned attempt.
- `llama.cpp-sacsharm/{build,build_zendnn}`: sacsharm's older builds (8fe90e1fb) used by the 2026-09-29/30 sweeps. Don't modify.
- Rebuilt CPU backend check of the rebased CONCAT tree: `profiling/check_rebased.sh` (compiles + runs both kernel checks).

Worktrees of the main repo (shared .git): `llama.cpp-matrix/src_base` (78651c4), `src_nolimit` (f012f20, detached),
`src_concat`, `src_zenlimits`. `git worktree list` to see them.

## 5. Scripts and what they do

`/proj/aigstaff/sohroy/profiling/`:
- `diag_lib.sh`: shared config for all llama-bench/opprof/vLLM runs (placement from the pod's cpuset, env, `set_cfg <config>`
  configs: latest_stock, latest_zendnn, latest_zendnn_algo3/5, latest_zendnn_fb0; `LATEST_SUFFIX=_concat` selects the build_*_concat dirs).
- `bench_concat.sh`: A/B (llama-bench pp8192, ub512,4096, 3 reps) for before/after the CONCAT fix. `VARIANTS="name cfg suffix;..."`.
- `opprof.cpp` + `opprof_concat.sh`, `opprof_8192.sh`: per-ggml-op wall-time profiler (scheduler eval callback). Build against the
  headers of the tree you profile. Output logs/TSV with op x backend x shape.
- `concat_check.cpp`, `concat_view_check.cpp`: standalone CONCAT correctness checks vs a naive strided reference (contiguous and
  transposed inputs, 5 types, 4 dims, 1/3/32 threads). Compile: `g++ -O1 -std=c++17 X.cpp -I<src>/ggml/include -L<build>/bin -lggml -lggml-base -lggml-cpu -Wl,-rpath,<build>/bin`.
- `investigate*.sh`, `pf_trace.sh`, `timeline.py`, `perf_llama_latest.sh`, `perf_analyze.py`, `moe_algo_sweep.sh` (written, not run):
  earlier perf/memory/page-fault investigation (see prior zips).
- `repro_latest_mmap.sh`: reproduces the rc3 mmap SIGSEGV with llama-bench. `mincore_chk.py <file>`: how much of a file is in page cache.
- `build_concat_zendnn.sh`, `build_llama_latest_zendnn.sh`, `rerun_clean.sh`: builds / reruns.

`/proj/aigstaff/sohroy/Benchmarking/` (llmbench harness; its working tree has other uncommitted changes from earlier work):
- `sweep.turin-zendnn-matrix.yaml` = 9-variant matrix (see section 7); `sweep.turin-zendnn-matrix-smoke.yaml` = quick check.
- `llmbench/suite/coresampler.py`: I ADDED `server_rss_gib / server_anon_gib / server_file_gib` (per 250 ms row, with the per-core CPU
  series); test in `tests/test_suite_coresampler.py` (7 pass). Uncommitted. There is no pod-level memory metric (cgroup files are host-wide).
- `scripts/summarize_cpu_ram.py out/<sweep>`: per-test table of t/s, busy/server/foreign cores, RSS (mean/peak, anon/file), load-peak RSS.
- Docs: `docs/turin-optimised-commands-2026-09-30.md` (command B = the llama.cpp command used), `docs/sweep.md`, `docs/contract.md`.

## 6. Commands that work well

llama-bench (what the profiling used):
```
env -u LD_PRELOAD LD_LIBRARY_PATH=<build>/bin[:<zendnnl lib>] ZENDNNL_MATMUL_ALGO=1 [ZENDNNL_GRP_MATMUL_ALGO=1] OMP_NUM_THREADS=32 \
  numactl --physcpubind=192-223 --membind=6 -- <build>/bin/llama-bench -m /proj/rdi/staff/sohroy/models/Qwen3.6-35B-A3B-BF16.gguf \
  -t 32 -b 4096 -ub 4096 -fa on -ctk f16 -ctv f16 -lm dio -p 8192 -n 0 -r 3 -o json
```
(the diag_lib harness also sets `LD_PRELOAD=tcmalloc:/proj/rdi/staff/sohroy/lib/libomp.so.5`, `KMP_AFFINITY=granularity=fine,compact,1,0`,
`KMP_BLOCKTIME=1 KMP_TPAUSE=0`, `KMP_*_BARRIER_PATTERN=dist,dist`, `OMP_DYNAMIC=FALSE OMP_WAIT_POLICY=ACTIVE`.)
llama-server command B (harness builds it): `... llama-server -m <gguf> --host 127.0.0.1 --port P -t 32 -tb 32 -c 32000 -np 1 -b 4096 -ub 4096 --metrics --alias <name> -fa on -ctk f16 -ctv f16 [-lm dio]`.

Sweep:
```
cd /proj/aigstaff/sohroy/Benchmarking
.venv/bin/llmbench sweep plan --spec sweep.turin-zendnn-matrix.yaml          # validate; 9 deployments, 18 trials
.venv/bin/llmbench sweep run  --spec sweep.turin-zendnn-matrix.yaml           # in the pod, detached (setsid nohup); ~45 min
.venv/bin/llmbench sweep cleanup out/<dir> --dry-run                           # stray servers
python3 scripts/summarize_cpu_ram.py out/turin-zendnn-matrix-dio               # host python is fine for this
```
Prefill-only in a spec: `workload: n_gen: [0]` (an empty list is rejected). `-lm dio` is needed for the rc3 builds.

Pitfalls worth remembering:
- **rc3 + mmap = SIGSEGV** (exit 139, ~3 s after the first request) for both `zen_latest*` builds; works with `-lm dio`. WW28 and CPU are fine with mmap.
  Suspected in-place weight repack on read-only mmap (unverified). rc3's first pass packs all experts: pp256 took ~400 s.
- **ZENDNNL_GRP_MATMUL_ALGO**: leave unset = AUTO scheduler -> extra decode custom-kernel pack, +55 GB RAM, slower on WW28, long ramp (~6 passes).
  `=1` = one in-place layout, ~65-69 GB on rc3 (WW28 nolimit still 125 GB). Decode with ALGO=1 is unmeasured.
- mmap'd GGUF page cache stays on the NUMA node that first read it and `--membind` doesn't move it (1.8x slower runs seen). `-lm dio`
  or check with `mincore_chk.py`. Page cache is host-wide.
- Without the limit removal, Qwen's 256-expert MUL_MAT_ID stays on CPU (ZenDNN then only helps dense matmuls: +2-3%).
- ZenDNN benchmarks need warm-up (>=3, ideally 6 passes); look at per-rep values in `report_reps.csv`.
- In sweeps, "busy cores" ~14-18 of 32 is expected: `KMP_BLOCKTIME=1` lets threads sleep during the serial phases.

## 7. Results (where they are, headline numbers)

Reports (zips in `/proj/aigstaff/sohroy/profiling/`): `qwen36_concat_report.zip` (problem -> CONCAT fix, 3-way vs vLLM),
`zendnn_matrix_report.zip` (9-variant matrix), `qwen36_moe_perf_20261007.zip` (earlier investigation). Older: `qwen36_moe_findings_20260929.zip`, `qwen36_moe_perf_20260930.zip`.

CONCAT fix (pp8192, t/s, same pod; results `concat_run2/`, `opprof_concat_run2/`, `opprof_pp8192/`):
| | ub512 before -> after | ub4096 before -> after |
|---|---|---|
| ZenDNN | 280 -> 352 | 278 -> 515 |
| stock CPU | 295 -> 366 | 246 -> 417 |
| vLLM+zentorch (earlier run) | - | 631 |
After the fix, at pp8192 the gap to vLLM is not MoE (5.96 s vs 6.18 s) but the GDN path (+1.8 s) and the long tail of small
element-wise/norm ops (+3.2 s); attention is faster in llama.cpp. Next optimisation targets: small-op fusion/threading, GDN path, then ub512 MoE.

Matrix (no CONCAT fix, ub4096, `-lm dio`, pp2048 / pp4096 t/s, RAM at pp4096; `Benchmarking/out/turin-zendnn-matrix-dio`):
| variant | t/s | RAM GiB |
|---|---|---|
| CPU | 260.1 / 257.9 | 71 |
| WW28 limit kept (algo1 / unset) | 266.2 / 268.6, 265.3 / 264.9 | 72 |
| WW28 nolimit algo1 | 280.0 / 285.3 | 125 |
| WW28 nolimit unset | 241.7 / 259.5 | 180 |
| rc3 limit kept (algo1 / unset) | 270.0 / 266.9, 264.1 / 266.2 | 69 |
| **rc3 nolimit algo1** | **288.3 / 289.7** | 69 |
| rc3 nolimit unset | 265.8 / 286.1 | 123 |
First (mmap) run with 4 crashed rc3 variants: `Benchmarking/out/turin-zendnn-matrix`.

## 8. Open items / ideas

- Re-measure the matrix WITH the CONCAT fix, and at ub512 (the llama.cpp default); only ub4096 was run.
- Whole-model equivalence check for the CONCAT change (perplexity); transposed `test_concat` case; upstream issue text (write it yourself).
- Rows-per-expert threshold sweep instead of removing the limit; ZenDNN fused-MoE call from ggml-zendnn; ALGO=1 decode check.
- Report rc3 mmap segfault to the ZenDNN team (repro: `repro_latest_mmap.sh`).
- Reword/squash the commits by hand; set up the fork -> upstream PR flow when compliance is settled.
- Regenerate the report zips if you want them to show the new commit hashes / zettabolt identity (they mention amd.com and pre-rebase hashes).
