#!/usr/bin/env python3
"""Turn a perf_llama.sh output dir into analysis.md.

Per run (one sub-dir per config x prompt x ubatch):
  1. throughput while profiled: llama-bench --progress lines, timestamped by perf_llama.sh
  2. counters: IPC, effective clock, cores busy, top-down (retiring / frontend / backend cpu vs
     memory / bad speculation), retired FP ops/s, where L1 fills come from, and an
     estimate of DRAM read traffic (L1 fills from DRAM + L2 prefetches that missed L3, x 64 B)
  3. flat profile (perf record -F 999) grouped into kernel categories, plus the top symbols
  4. per-op profile from the call stacks (cpu-clock, frame pointers): every sample is assigned the ggml
     op in its stack (ggml_compute_forward_* / ggml_zendnn_compute_forward_*), or, where the frame chain
     stops inside ggml (no frame pointers there), the category of its leaf function; samples whose leaf
     is in libomp are spinning (under ggml_graph_compute_thread: a ggml barrier). The main thread takes part in every op
     (thread 0 of ggml's OpenMP team, and the caller of ZenDNN), so its samples give each op's share
     of wall time. Other threads' samples are matched to the main thread's op at the nearest time
     (within 6 ms), which gives how many threads were doing work (not spinning in libomp) during it.
  5. per-thread utilization
vLLM runs (engine=vllm): perf is attached to the engine's worker process; there is no ggml op table,
the per-op split comes from the torch profiler (vllm_profile_pp*/profiler_out_0.txt).
"""
from __future__ import annotations

import bisect
import glob
import gzip
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict

OUT = sys.argv[1]
lines: list[str] = []
p = lines.append


def table(header, rows):
    p("| " + " | ".join(header) + " |")
    p("|" + "|".join("---" for _ in header) + "|")
    for r in rows:
        p("| " + " | ".join(str(c) for c in r) + " |")
    p("")


# ---------------------------------------------------------------- parsing helpers
def read_meta(d):
    m = {}
    try:
        for kv in open(f"{d}/meta.txt").read().split():
            k, v = kv.split("=", 1)
            m[k] = v
    except OSError:
        pass
    return m


def read_stat(path):
    """perf stat -o file -> ({event: value}, {metric: value}, elapsed_s)."""
    ev, met, elapsed = {}, {}, None
    try:
        text = open(path).read()
    except OSError:
        return ev, met, elapsed
    for ln in text.splitlines():
        m = re.match(r"^\s*([\d.,]+)\s+(?:msec\s+)?([\w.:/=,-]+)", ln)
        if m and not ln.strip().startswith("#"):
            try:
                ev[m.group(2)] = float(m.group(1).replace(",", ""))
            except ValueError:
                pass
        for mm in re.finditer(r"#\s+([\d.]+)\s+(%slots|%ops|per_1k_instr)?\s*([A-Za-z_]\w*)", ln):
            met[mm.group(3)] = float(mm.group(1))
        m = re.search(r"([\d.]+) seconds time elapsed", ln)
        if m:
            elapsed = float(m.group(1))
    return ev, met, elapsed


def pass_times(d, windows):
    """Durations of the llama-bench passes that lie inside the profiling windows."""
    starts = []
    try:
        for ln in open(f"{d}/bench.err"):
            m = re.match(r"^([\d.]+) llama-bench: benchmark \d+/\d+: prompt run (\d+)/\d+", ln)
            if m:
                starts.append(float(m.group(1)))
    except OSError:
        return []
    if not windows:
        return []
    lo, hi = min(windows.values(), key=lambda x: x[0])[0], max(windows.values(), key=lambda x: x[1])[1]
    return [b - a for a, b in zip(starts, starts[1:]) if a >= lo - 1 and b <= hi + 1]


def read_windows(d):
    w = {}
    try:
        for ln in open(f"{d}/windows.txt"):
            name, what, t = ln.split()
            w.setdefault(name, [0.0, 0.0])[0 if what == "start" else 1] = float(t)
    except OSError:
        pass
    return w


# ---------------------------------------------------------------- symbol categories
OMP_DSO = re.compile(r"lib(i)?omp")


def kernel_category(dso: str, sym: str) -> str:
    if OMP_DSO.search(dso):
        return "OpenMP runtime (spin-wait at barriers, fork/join)"
    if "libzentorch" in dso:
        if re.search(r"moe", sym, re.I):
            return "zentorch/ZenDNN: fused MoE (incl. its GEMMs)"
        if re.search(r"pack|reorder|unpack|quant", sym, re.I) and "gemm" not in sym:
            return "ZenDNN: weight packing / reorder / quantization"
        return "zentorch/ZenDNN GEMM kernels"
    if dso.startswith("[JIT]") or "libzendnnl" in dso:
        if re.search(r"pack|reorder|unpack|quant", sym, re.I) and "gemm" not in sym:
            return "ZenDNN: weight packing / reorder / quantization"
        return "ZenDNN/AOCL-DLP GEMM kernels (incl. JIT code)"
    if "libggml-cpu" in dso:
        if "vec_dot_bf16" in sym:
            return "ggml: bf16 dot product (MUL_MAT_ID experts on CPU)"
        if re.search(r"tinyBLAS|llamafile|sgemm|gemm", sym):
            return "ggml: tinyBLAS/llamafile GEMM (dense matmul on CPU)"
        if re.search(r"vec_dot_(f32|f16)", sym):
            return "ggml: f32/f16 dot product"
        if "concat" in sym:
            return "ggml: CONCAT (GDN conv state)"
        if "flash_attn" in sym:
            return "ggml: flash attention"
        if re.search(r"gated_delta|ssm_conv", sym):
            return "ggml: GDN recurrence + conv1d"
        if "ggml_graph_compute_thread" in sym:
            return "ggml: graph loop (ggml_barrier, inlined small ops)"
        if re.search(r"to_bf16|bf16_to|fp32_to|fp16|from_float", sym):
            return "ggml: type conversion"
        return "ggml: other ops (norms, elementwise, softmax, ...)"
    if re.search(r"vllm/_C|_C\.abi3|vllm.*\.so", dso):
        if re.search(r"attn|attention", sym, re.I):
            return "vLLM C++: attention"
        if re.search(r"gated_delta|gdn|conv1d|causal_conv", sym, re.I):
            return "vLLM C++: GDN / conv1d"
        return "vLLM C++: other kernels"
    if "torchinductor" in dso or "inductor" in dso:
        return "Inductor-generated kernels (fused elementwise)"
    if "libtorch" in dso or "libc10" in dso:
        return "PyTorch runtime / aten ops"
    if re.search(r"python|\.cpython-", dso):
        return "Python interpreter / extension glue"
    if "libggml-base" in dso:
        return "ggml: type conversion" if re.search(r"bf16|fp16|fp32", sym) else "ggml-base: other"
    if "libc.so" in dso:
        return "libc (memcpy / memset)"
    if "kernel" in dso:
        return "kernel"
    return "other"


def parse_flat(d):
    """Symbols and their share of the non-idle cycles (-C runs also sample the idle loop)."""
    rows = []
    try:
        for ln in open(f"{d}/flat_sym.txt"):
            m = re.match(r"^\s+([\d.]+)%\s+(\S+(?: tid \d+)?)\s+\[[.k]\]\s+(.*)$", ln)
            if m:
                rows.append((float(m.group(1)), m.group(2), m.group(3).strip()))
    except OSError:
        pass
    idle = sum(p for p, dso, sym in rows if dso == "[kernel.kallsyms]" and
               re.search(r"^(acpi_idle|intel_idle|mwait_idle|cpuidle|poll_idle|default_idle|do_idle|native_safe_halt|io_idle)", sym))
    if idle and idle < 100:
        rows = [(p * 100 / (100 - idle), dso, sym) for p, dso, sym in rows
                if not (dso == "[kernel.kallsyms]" and re.search(r"idle|safe_halt", sym))]
    return rows


# ---------------------------------------------------------------- DWARF stacks
OP_RE = re.compile(r"^(ggml_zendnn_compute_forward|ggml_compute_forward)_(\w+)")
OP_SUFFIX = re.compile(r"(_one_chunk|_f32|_f16|_bf16|_tiled|_non_quantized|_fused|_ext_f16|_thread|_q8_0)+$")


def op_of(frames) -> str | None:
    for sym, dso in frames:
        m = OP_RE.match(sym)
        if m:
            name = OP_SUFFIX.sub("", m.group(2))
            if m.group(1).startswith("ggml_zendnn"):
                # the ZenDNN backend's own frame: same work as the GEMM-kernel samples below it
                return "ZenDNN matmul (AOCL-DLP GEMM kernels)" if name == "mul_mat" else "ZenDNN " + name
            return name
        if "apply_unary_op" in sym or "unary_op<" in sym:
            return "unary (sigmoid, softplus, exp, ...)"
    return None


def parse_script(path):
    """-> list of (pid, tid, time, frames[(sym, dso)] leaf first)."""
    samples, cur = [], None  # (samples of the idle task, pid 0, are dropped below)
    with gzip.open(path, "rt", errors="replace") as f:
        for ln in f:
            if not ln.strip():
                continue
            if not ln.startswith("\t") and not ln.startswith(" " * 8):
                m = re.match(r"^\s*(?:(\d+)/)?(\d+)\s+([\d.]+):", ln)
                if m:  # "pid/tid time:" (or "tid time:" without -F pid: pid filled in below)
                    cur = (int(m.group(1) or 0), int(m.group(2)), float(m.group(3)), [])
                    samples.append(cur)
                continue
            if cur is None:
                continue
            # "<addr> <symbol> (<dso>)": the symbol may itself contain parentheses (C++ signatures,
            # "(anonymous namespace)"), so the dso is the last parenthesised group
            m = re.match(r"^\s+[0-9a-f]+\s+(.*)\s+\(([^()]*)\)\s*$", ln)
            if m:
                cur[3].append((m.group(1), m.group(2)))
    samples = [x for x in samples if not (x[0] == 0 and x[1] == 0)]   # the idle loop (-C runs)
    if samples and samples[0][0] == 0:  # no pid in the script: the main thread has the lowest tid
        pid = min(t for _, t, _, _ in samples)
        samples = [(pid, t, ts, fr) for _, t, ts, fr in samples]
    return samples


def classify(frames):
    user = [f for f in frames if f[1] != "[kernel.kallsyms]"]
    leaf_sym, leaf_dso = user[0] if user else ("kernel", "[kernel.kallsyms]")
    op = op_of(frames)
    waiting = bool(OMP_DSO.search(leaf_dso))
    if op is None:
        if any("libzendnnl" in dso or dso.startswith("[JIT]") for _, dso in user):
            op = "ZenDNN matmul (AOCL-DLP GEMM kernels)"
        elif waiting and any("ggml_graph_compute_thread" in s for s, _ in user):
            op = "(ggml barrier: waiting for the slowest thread)"
        elif waiting:
            op = "(OpenMP: idle / fork-join)"
        else:
            op = "[leaf] " + kernel_category(leaf_dso, leaf_sym)
    return op, waiting, kernel_category(leaf_dso, leaf_sym)


def analyze_stacks(d):
    path = f"{d}/stacks_script.txt.gz"
    if not os.path.exists(path):
        return None
    samples = parse_script(path)
    if not samples:
        return None
    pid = samples[0][0]
    main = sorted((t, classify(fr)) for pp, tid, t, fr in samples if tid == pid)
    main_t = [t for t, _ in main]
    per_op = defaultdict(lambda: Counter())    # op -> counters
    per_tid = defaultdict(lambda: Counter())
    for pp, tid, t, fr in samples:
        op, waiting, kcat = classify(fr)
        per_tid[tid]["work" if not waiting else "wait"] += 1
        per_op[op]["cpu_" + ("wait" if waiting else "work")] += 1
        if tid == pid:
            per_op[op]["main"] += 1
            if not waiting:
                per_op[op]["aligned_work"] += 1
            continue
        i = bisect.bisect_left(main_t, t)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(main_t) and (best is None or abs(main_t[j] - t) < abs(main_t[best] - t)):
                best = j
        if best is not None and abs(main_t[best] - t) <= 0.006:
            mop = main[best][1][0]
            per_op[mop]["aligned_" + ("wait" if waiting else "work")] += 1
    return {"pid": pid, "n": len(samples), "n_main": len(main), "per_op": per_op, "per_tid": per_tid}


# ---------------------------------------------------------------- report
runs = sorted(x for x in os.listdir(OUT) if os.path.isdir(f"{OUT}/{x}") and os.path.exists(f"{OUT}/{x}/meta.txt"))
p(f"# perf analysis: {os.path.basename(os.path.abspath(OUT))}\n")
p("Qwen3.6-35B-A3B BF16 prefill with the optimised commands of turin-optimised-commands-2026-09-30.md "
  "(boost off): llama.cpp setup B on sacsharm's existing stock and ZenDNN builds (weights loaded with "
  "--load-mode dio into local memory), vLLM setup C. 32 cores / 1 NUMA node. perf attached after load and "
  "warm-up (vLLM: to its worker process); one measurement per window. See perf_llama.sh.\n")

summary_rows, counter_rows, mem_rows = [], [], []
for r in runs:
    d = f"{OUT}/{r}"
    meta = read_meta(d)
    pp = int(meta.get("pp", 0) or 0)
    wins = read_windows(d)
    if meta.get("engine") == "vllm":
        try:
            lat = json.load(open(f"{d}/bench.json"))["latencies"]
            pts = lat
            tps = f"{pp / statistics.median(lat):.1f}"
        except (OSError, ValueError, KeyError):
            # no JSON (it could not be written): the average the benchmark printed
            m = re.search(r"Avg latency: ([\d.]+) seconds", open(f"{d}/bench.log", errors="replace").read()) \
                if os.path.exists(f"{d}/bench.log") else None
            pts, tps = ([], f"{pp / float(m.group(1)):.1f} (avg)") if m else ([], "?")
    else:
        pts = pass_times(d, wins)
        tps = f"{pp / statistics.median(pts):.1f}" if pts else "?"
    numa = open(f"{d}/numa.txt").read().strip() if os.path.exists(f"{d}/numa.txt") else ""
    if meta.get("engine") == "vllm" and "N1=1.1GB" in numa:
        # this run read numa_maps of the benchmark's Python process, not the model worker; the worker
        # was read in an earlier attempt of the same command (invalid_vllm_attempts_compile_phase/first)
        numa = "worker: N1=68.5GB N5=0.2GB, rest 0 (read in an earlier attempt, same command)"
    summary_rows.append([r, tps, len(pts), meta.get("load_s", "?"), numa])

    b, _, eb = read_stat(f"{d}/stat_basic.txt")
    _, td, _ = read_stat(f"{d}/stat_topdown.txt")
    fl, _, efl = read_stat(f"{d}/stat_flops.txt")
    mm, _, emm = read_stat(f"{d}/stat_mem.txt")
    l2, _, el2 = read_stat(f"{d}/stat_l2.txt")
    tc, cyc, ins = b.get("task-clock"), b.get("cycles"), b.get("instructions")
    if "scope" in meta:  # vLLM runs: counters over the pod's cores (-C), which run only the benchmark
        # -C <cpus>: task-clock is wall time of every cpu; cycles stop when a core idles (halted).
        # The clock is fixed (boost off, measured 2.72 GHz in the -p runs): busy cores = cycles / (t x 2.72 GHz)
        ghz = 2.72
        cores = cyc / eb / 2.72e9 if cyc and eb else None
    else:
        cores = tc / 1000 / eb if tc and eb else None
        ghz = cyc / (tc / 1000) / 1e9 if cyc and tc else None
    ipc = ins / cyc if ins and cyc else None
    mac = fl.get("fp_ret_sse_avx_ops.mac_flops")
    allf = fl.get("fp_ret_sse_avx_ops.all")
    counter_rows.append([
        r, f"{cores:.1f}" if cores else "?", f"{ghz:.2f}" if ghz else "?", f"{ipc:.2f}" if ipc else "?",
        *(f"{td.get(k, float('nan')):.1f}" for k in ("retiring", "frontend_bound", "bad_speculation",
                                                       "backend_bound_by_cpu", "backend_bound_by_memory")),
        f"{allf / efl / 1e9:.0f}" if allf and efl else "?",
        f"{mac / efl / 1e9:.0f}" if mac and efl else "?",
    ])
    if emm:
        keys = ["local_l2", "local_ccx", "near_cache", "dram_io_near", "dram_io_far"]
        vals = {k: mm.get(f"ls_any_fills_from_sys.{k}", 0.0) for k in keys}
        tot = sum(vals.values()) or 1.0
        pf_dram = (l2.get("l2_pf_miss_l2_l3.l2_hwpf", 0.0) + l2.get("l2_pf_miss_l2_l3.l1_dc_l2_hwpf", 0.0))
        dram_gbs = (vals["dram_io_near"] + vals["dram_io_far"]) * 64 / emm / 1e9
        pf_gbs = pf_dram * 64 / el2 / 1e9 if el2 else 0.0
        mem_rows.append([r] + [f"{100 * vals[k] / tot:.1f}" for k in keys]
                        + [f"{dram_gbs:.1f}", f"{pf_gbs:.1f}", f"{dram_gbs + pf_gbs:.0f}"])

p("## Runs\n")
table(["run", "prefill t/s while profiled (median pass)", "passes", "start-up s", "resident memory per NUMA node"],
      summary_rows)
p("## Counters (whole process, during the windows)\n")
table(["run", "cores busy", "GHz", "IPC", "retiring %", "frontend %", "bad spec %", "backend: cpu %",
       "backend: memory %", "FP G-ops/s (all)", "FP G-ops/s (mul-add)"], counter_rows)
p("Top-down columns are % of dispatch slots (PipelineL1/L2). FP columns: retired FP operations per second "
  "(fp_ret_sse_avx_ops.all / .mac_flops), across the whole process. Their bf16 and packed-fp32 umasks read "
  "0 on this CPU, so the split by data type is not available.\n")
if mem_rows:
    p("## Where L1 data-cache fills come from (% of fills) and estimated DRAM read traffic\n")
    table(["run", "L2 %", "L3 same CCX %", "other CCX %", "DRAM this socket %",
           "DRAM other socket %", "L1 fills from DRAM GB/s", "L2 prefetch misses to DRAM GB/s", "DRAM read est. GB/s"],
          mem_rows)
    p("DRAM near/far = this socket / the other socket. The estimate counts 64-byte lines from L1 fills "
      "that came from DRAM plus L2 hardware prefetches that missed L3; it excludes writes. The pods "
      "expose no uncore (data-fabric) counters to measure DRAM bandwidth directly. NUMA node 1 has 3 of "
      "the socket's 12 DDR5 channels (NPS4), roughly 140 GB/s peak.\n")

for r in runs:
    d = f"{OUT}/{r}"
    p(f"## {r}\n")
    flat = parse_flat(d)
    if flat:
        cats = Counter()
        for pct, dso, sym in flat:
            cats[kernel_category(dso, sym)] += pct
        p("### Where the cycles go (flat profile, all threads)\n")
        table(["category", "% of cycles"], [[k, f"{v:.1f}"] for k, v in cats.most_common()])
        p("### Top symbols\n")
        table(["%", "object", "symbol"],
              [[f"{pct:.2f}", dso, sym[:90]] for pct, dso, sym in flat[:20]])
    st = analyze_stacks(d)
    if st and read_meta(d).get("engine") == "vllm":
        # CPU-wide sampling of a Python engine: the ggml op table and per-thread shares do not apply;
        # the flat profile above and the torch-profiler split below cover vLLM
        st = None
    if st:
        n_main = st["n_main"] or 1
        n_threads = len(st["per_tid"])
        rows = []
        for op, c in sorted(st["per_op"].items(), key=lambda kv: -kv[1]["main"]):
            if not c["main"]:
                continue
            busy = (c["aligned_work"]) / c["main"]
            rows.append([op, f"{100 * c['main'] / n_main:.1f}", f"{busy:.1f}",
                         f"{100 * c['cpu_work'] / st['n']:.1f}", f"{100 * c['cpu_wait'] / st['n']:.1f}"])
        p(f"### Per ggml op (cpu-clock stacks: {st['n']} samples, {n_threads} threads, main thread {st['pid']})\n")
        table(["op (main thread's stack)", "% of wall time", f"threads doing work (of {n_threads})",
               "% of all CPU samples: work", "% of all CPU samples: spinning"], rows)
        p("'% of wall time' is the main thread's samples. 'threads doing work' counts, per main-thread "
          "sample in that op, the samples of all threads within 6 ms that were not spinning in libomp.\n")
        trows = []
        for i, (tid, c) in enumerate(sorted(st["per_tid"].items(), key=lambda kv: (kv[0] != st["pid"], kv[0]))):
            tot = c["work"] + c["wait"]
            trows.append([("main " if tid == st["pid"] else "") + str(tid), tot, f"{100 * c['work'] / tot:.0f}"])
        works = [100 * c["work"] / (c["work"] + c["wait"]) for c in st["per_tid"].values() if c["work"] + c["wait"]]
        p(f"### Threads: {n_threads}, work share min/median/max = {min(works):.0f}/{statistics.median(works):.0f}/"
          f"{max(works):.0f}% (the rest is spinning in libomp)\n")

# ---------------------------------------------------------------- vLLM torch profiles
TORCH_BUCKETS = [
    ("MoE experts (zentorch fused MoE incl. its copies)", re.compile(r"fused_moe|moe", re.I)),
    ("GDN / linear attention (chunk rule, conv1d, gating)", re.compile(r"gdn|gated_delta|causal_conv|conv1d", re.I)),
    ("attention", re.compile(r"attention|attn|sdpa|paged", re.I)),
    ("linear / GEMM (zentorch_linear)", re.compile(r"linear|addmm|\bmm\b|aten::mm|matmul|bmm|gemm", re.I)),
    ("concatenation (aten::cat)", re.compile(r"aten::cat$")),
]


def to_ms(x):
    m = re.match(r"([\d.]+)(us|ms|s)$", x.strip())
    if not m:
        return 0.0
    v = float(m.group(1))
    return v / 1000 if m.group(2) == "us" else v * 1000 if m.group(2) == "s" else v


for f in sorted(glob.glob(f"{OUT}/vllm_profile_pp*/profiler_out_*.txt")):
    tag = os.path.basename(os.path.dirname(f))
    txt = open(f).read().splitlines()
    total = next((l.split(":", 1)[1].strip() for l in txt if "Self CPU time total" in l), "?")
    hdr = next((i for i, l in enumerate(txt) if l.strip().startswith("Name")), None)
    if hdr is None:
        continue
    rows = []
    for l in txt[hdr + 2:]:
        if l.startswith("---") or not l.strip():
            break
        parts = re.split(r"\s{2,}", l.strip())
        if len(parts) >= 7:
            rows.append(parts)
    b = defaultdict(float)
    for r in rows:
        name, ms = r[0], to_ms(r[2])
        if name.startswith("## Call CompiledFxGraph") or name.startswith("execute_context"):
            b["(compiled-graph / step wrappers: self time)"] += ms
            continue
        for k, rx in TORCH_BUCKETS:
            if rx.search(name):
                b[k] += ms
                break
        else:
            b["other ops (top-50 rows)"] += ms
    p(f"## vLLM per-op split: {tag} (torch profiler, one generate call; self CPU total {total})\n")
    table(["bucket", "self CPU ms"], [[k, f"{v:.0f}"] for k, v in sorted(b.items(), key=lambda kv: -kv[1])])
    table(["op", "self CPU %", "self CPU", "calls"], [[r[0], r[1], r[2], r[-1]] for r in rows[:15]])

open(f"{OUT}/analysis.md", "w").write("\n".join(lines) + "\n")
print(f"wrote {OUT}/analysis.md")
