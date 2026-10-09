#!/usr/bin/env python3
"""Summarise one qwen36_moe_diag.sh output directory into <out>/summary.md.

Sections: clean prefill throughput (llama-bench, vllm bench latency), the llama.cpp per-op
profile (opprof) bucketed by model component, the vLLM torch-profiler top ops bucketed the same
way, GGML_SCHED_DEBUG op placement, the ZenDNNL group-matmul log, NUMA placement of each run's
memory, and the step list.
"""
import csv
import glob
import json
import os
import re
import statistics
import sys
from collections import defaultdict

out = sys.argv[1] if len(sys.argv) > 1 else "."
ZLOG = re.compile(r"^\[[A-Za-z ]+\]\[(info|warning|error|verbose|debug)\]")
lines = []
p = lines.append


def table(header, rows):
    p("| " + " | ".join(header) + " |")
    p("|" + "|".join("---" for _ in header) + "|")
    for r in rows:
        p("| " + " | ".join(str(c) for c in r) + " |")
    p("")


# ------------------------------------------------------------------ clean throughput
p(f"# Qwen3.6-35B-A3B prefill diagnosis: {os.path.basename(os.path.abspath(out))}\n")
p("## Clean prefill throughput (tokens/s, mean ± sd)\n")
tps = defaultdict(dict)  # (pp) -> config -> str
for f in sorted(glob.glob(f"{out}/bench_*.json")):
    cfg = os.path.basename(f)[len("bench_"):-len(".json")]
    try:
        # ZenDNNL prints "[API    ][warning][..]" lines to stdout, i.e. into llama-bench's JSON
        data = json.loads("".join(l for l in open(f) if not ZLOG.match(l)))
    except Exception as e:  # noqa: BLE001
        p(f"- {f}: unreadable ({e})")
        continue
    for t in data:
        if t.get("n_gen", 0) != 0:
            continue
        cell = f"{t['avg_ts']:.1f} ± {t['stddev_ts']:.1f}"
        sm = t.get("samples_ts", [])
        if len(sm) > 3:  # warming-up configs: also show the last 3 passes and the first
            cell += f" (last 3: {statistics.mean(sm[-3:]):.1f}; 1st {sm[0]:.1f}; n={len(sm)})"
        tps[t["n_prompt"]][f"llama.cpp {cfg} ub{t['n_ubatch']}"] = cell
for f in sorted(glob.glob(f"{out}/vllm_*pp*.json")):
    m = re.search(r"vllm_(?:(\w+?)_)?pp(\d+)\.json$", f)
    if not m:
        continue
    tag, pp = m.group(1) or "base", int(m.group(2))
    d = json.load(open(f))
    lat = d["latencies"]
    t = [pp / x for x in lat]
    sd = statistics.stdev(t) if len(t) > 1 else 0.0
    tps[pp][f"vLLM {tag} (bench latency, out=1)"] = f"{statistics.mean(t):.1f} ± {sd:.1f}"
cols = sorted({c for v in tps.values() for c in v})
table(["config"] + [f"pp{pp}" for pp in sorted(tps)],
      [[c] + [tps[pp].get(c, "") for pp in sorted(tps)] for c in cols])
p("vLLM t/s = prompt / latency of a 1-token generation (prefill + one sample), batch 1. "
  "mnbt8192 = --max-num-batched-tokens 8192 (the whole prompt in one chunk).\n")

tg = []
for f in sorted(glob.glob(f"{out}/tg_*.json")):
    cfg = os.path.basename(f)[3:-5]
    try:
        data = json.loads("".join(l for l in open(f) if not ZLOG.match(l)))
    except Exception as e:  # noqa: BLE001
        p(f"- {f}: unreadable ({e})")
        continue
    for t in data:
        if t.get("n_gen", 0) > 0 and t.get("n_prompt", 0) == 0:
            tg.append([cfg, f"tg{t['n_gen']}", f"{t['avg_ts']:.2f} ± {t['stddev_ts']:.2f}"])
if tg:
    p("## Decode (llama-bench tg, tokens/s)\n")
    table(["config", "test", "t/s"], tg)

vt = f"{out}/vllm_text.json"
if os.path.exists(vt):
    d = json.load(open(vt))
    p(f"## vLLM: random vs real-text prompts (one engine, vllm_prompt_bench.py; text = {d['text_tokens']} tokens)\n")
    table(["prompt", "kind", "t/s (mean)", "latencies s"],
          [[f"pp{r['n_prompt']}", r["kind"], f"{r['tps_mean']:.1f}", " ".join(f"{x:.3f}" for x in r["latencies"])]
           for r in d["runs"]])

# ------------------------------------------------------------------ llama.cpp op profiles
GDN_OPS = {"GATED_DELTA_NET", "SSM_CONV", "SOLVE_TRI", "CUMSUM", "TRI", "FILL", "SSM_SCAN"}
GDN_STEMS = re.compile(r"^(alpha|a_softplus|beta|beta_sigmoid|b_in|conv_|decay_mask|dnet_|g_|gate|"
                       r"k_|key_gdiff|linear_attn|new_state|output_state|q_|qkv_mixed|state_|v_|z$|"
                       r"attn_inter|attn_pre_solve|attn_gated|attn_residual|k_cumdecay)")


def bucket_llama(op, stem, wbuf):
    if op == "MUL_MAT_ID":
        return "MoE routed experts (MUL_MAT_ID)"
    if op == "MUL_MAT":
        if wbuf == "act":
            return "activation x activation matmul"
        if "shexp" in stem or "shared_expert" in stem:
            return "weight matmul: shared expert"
        if stem.startswith("ffn_moe"):
            return "weight matmul: MoE router"
        if stem == "result_output":
            return "weight matmul: lm head"
        return "weight matmul: attention / GDN projections"
    if op == "FLASH_ATTN_EXT":
        return "flash attention"
    if op == "CONCAT" and stem == "conv_input":
        return "GDN conv-state CONCAT (single-threaded in ggml-cpu)"
    if op == "SIGMOID" and stem == "gate_sigmoid":
        return "attention output-gate SIGMOID"
    if op in ("GATED_DELTA_NET", "SSM_CONV"):
        return "GDN recurrence + conv1d"
    if stem.startswith("ffn_moe"):
        return "MoE glue (topk, act, weighting)"
    if op in GDN_OPS or GDN_STEMS.match(stem):
        return "GDN other elementwise"
    return "other (norms, rope, adds, copies)"


profiles = sorted(glob.glob(f"{out}/opprof_*.tsv"))
if profiles:
    p("## llama.cpp per-op profile (opprof: per-node wall time, ms per prompt)\n")
    summ = {}
    heads = {}
    detail = {}
    for f in profiles:
        name = os.path.basename(f)[len("opprof_"):-len(".tsv")]
        meta = {}
        rows = []
        with open(f) as fh:
            for ln in fh:
                if ln.startswith("#"):
                    meta.update(dict(kv.split("=", 1) for kv in ln[1:].split() if "=" in kv))
                    continue
                rows.append(ln.rstrip("\n").split("\t"))
        if not rows:
            continue
        hdr, rows = rows[0], rows[1:]
        idx = {h: i for i, h in enumerate(hdr)}
        # unnamed nodes ("node_3354", e.g. flash attention) -> one row per (op, backend, shape)
        merged = {}
        for r in rows:
            if re.fullmatch(r"node_\d+", r[idx["stem"]]):
                r = r[:]
                r[idx["stem"]] = "(unnamed)"
            key = tuple(r[: idx["calls"]])
            if key in merged:
                m = merged[key]
                for c in ("calls", "ms", "pct"):
                    m[idx[c]] = f"{float(m[idx[c]]) + float(r[idx[c]]):g}"
            else:
                merged[key] = r[:]
        rows = sorted(merged.values(), key=lambda r: -float(r[idx["ms"]]))
        b = defaultdict(lambda: [0.0, 0.0])  # bucket -> [ms total, ms on ZenDNN]
        for r in rows:
            ms = float(r[idx["ms"]])
            k = bucket_llama(r[idx["op"]], r[idx["stem"]], r[idx["wbuf"]])
            b[k][0] += ms
            if r[idx["backend"]].startswith("ZenDNN"):
                b[k][1] += ms
        summ[name] = b
        heads[name] = meta
        detail[name] = (idx, rows)
    names = list(summ)
    table(["run", "clean t/s", "profiled t/s", "sum(nodes) ms"],
          [[n, heads[n].get("clean_tps", "?"), heads[n].get("profiled_tps", "?"),
            heads[n].get("sum_node_ms", "?")] for n in names])
    buckets = sorted({k for b in summ.values() for k in b},
                     key=lambda k: -max(summ[n].get(k, [0, 0])[0] for n in names))
    table(["bucket (ms, % of nodes; [ms on ZenDNN])"] + names,
          [[k] + [(lambda v, tot: f"{v[0]:.0f} ({100 * v[0] / tot:.0f}%) [{v[1]:.0f}]")(
              summ[n].get(k, [0, 0]), sum(x[0] for x in summ[n].values()) or 1) for n in names]
           for k in buckets])
    for n in names:
        idx, rows = detail[n]
        p(f"### top rows: {n}\n")
        table(["op", "stem", "backend", "wtype", "shape", "calls", "ms", "%"],
              [[r[idx[c]] for c in ("op", "stem", "backend", "wtype", "shape", "calls", "ms", "pct")]
               for r in rows[:15]])

# ------------------------------------------------------------------ vLLM torch profiles
VLLM_BUCKETS = [
    ("MoE routed experts (zentorch_fused_moe / fused_moe)", re.compile(r"fused_moe|moe", re.I)),
    ("GDN / linear attention", re.compile(r"gdn|gated_delta|causal_conv|conv1d|delta", re.I)),
    ("attention", re.compile(r"attention|attn|sdpa|paged", re.I)),
    ("linear / GEMM (zentorch_linear, mm, addmm)", re.compile(r"linear|addmm|\bmm\b|aten::mm|matmul|bmm|gemm", re.I)),
]
for f in sorted(glob.glob(f"{out}/vllm_profile_pp*/profiler_out_*.txt")):
    tag = os.path.basename(os.path.dirname(f))
    txt = open(f).read().splitlines()
    total = next((l for l in txt if "Self CPU time total" in l), "")
    p(f"## vLLM torch profile: {tag}  ({total.strip()})\n")
    hdr_i = next((i for i, l in enumerate(txt) if l.strip().startswith("Name")), None)
    if hdr_i is None:
        p("(no table)\n")
        continue
    # columns are fixed-width; split on 2+ spaces
    rows = []
    for l in txt[hdr_i + 2:]:
        if l.startswith("---") or not l.strip():
            break
        parts = re.split(r"\s{2,}", l.strip())
        if len(parts) >= 7:
            rows.append(parts)

    def to_ms(s):
        m = re.match(r"([\d.]+)(us|ms|s)", s)
        if not m:
            return 0.0
        v = float(m.group(1))
        return v / 1000 if m.group(2) == "us" else v * 1000 if m.group(2) == "s" else v

    b = defaultdict(float)
    for r in rows:
        name, self_ms = r[0], to_ms(r[2])
        for k, rx in VLLM_BUCKETS:
            if rx.search(name):
                b[k] += self_ms
                break
        else:
            b["other (top-50 rows only)"] += self_ms
    table(["bucket (self CPU ms, top-50 ops)", "ms"], [[k, f"{v:.0f}"] for k, v in sorted(b.items(), key=lambda x: -x[1])])
    table(["op", "self CPU %", "self CPU", "calls"], [[r[0], r[1], r[2], r[-1]] for r in rows[:20]])

# ------------------------------------------------------------------ placement, clocks, failures
for f in sorted(glob.glob(f"{out}/placement_*.txt")):
    p(f"## Op placement (GGML_SCHED_DEBUG=2, last graph): {os.path.basename(f)}\n")
    p("```")
    p(open(f).read().rstrip())
    p("```\n")

numa = sorted(glob.glob(f"{out}/numa_*.txt"))
if numa:
    p("## Where each llama.cpp run's memory lives (numa_maps once the weights are loaded)\n")
    table(["run", "resident per node"], [[os.path.basename(f)[5:-4], open(f).read().strip()] for f in numa])

import gzip  # noqa: E402
for f in sorted(glob.glob(f"{out}/zendnnl_*.log.gz")):
    grp = defaultdict(lambda: [0, 0.0, []])  # (mode, algo) -> [calls, ms, num_ops list]
    dense = defaultdict(lambda: [0, 0.0])    # kernel -> [calls, ms]
    for ln in gzip.open(f, "rt", errors="replace"):
        if not ln.startswith("[PROF"):
            continue
        t = re.search(r"time=([\d.]+)ms", ln)
        ms = float(t.group(1)) if t else 0.0
        if "[GRP_MATMUL.CALL]" in ln:
            m = re.search(r"num_ops=(\d+) mode=(\S+) exec_algo=(\d+)", ln)
            if m:
                g = grp[(m.group(2), m.group(3))]
                g[0] += 1
                g[1] += ms
                g[2].append(int(m.group(1)))
        elif "matmul_direct" in ln:
            k = re.search(r"kernel=([\w]+)", ln)
            d = dense[k.group(1) if k else "?"]
            d[0] += 1
            d[1] += ms
    p(f"## ZenDNNL calls: {os.path.basename(f)} (one cold pass, includes first-use weight packing)\n")
    table(["group matmul mode", "exec_algo", "calls", "ms", "active experts (min/median/max)"],
          [[k[0], k[1], v[0], f"{v[1]:.0f}", f"{min(v[2])}/{statistics.median(v[2]):.0f}/{max(v[2])}"]
           for k, v in grp.items()])
    table(["dense matmul kernel", "calls", "ms"], [[k, v[0], f"{v[1]:.0f}"] for k, v in dense.items()])

mhz = f"{out}/cpu_mhz.tsv"
if os.path.exists(mhz):
    per = defaultdict(list)
    for ln in open(mhz):
        parts = ln.rstrip("\n").split("\t")
        if len(parts) == 3 and parts[1] not in ("", "idle"):
            per[parts[1]].append(int(parts[2]))
    if per:
        p("## Effective clock of the benchmark cores per step (cpuinfo_avg_freq, MHz)\n")
        table(["step", "mean", "min", "max", "samples"],
              [[k, f"{statistics.mean(v):.0f}", min(v), max(v), len(v)] for k, v in per.items()])

steps = f"{out}/steps.tsv"
if os.path.exists(steps):
    p("## Steps\n")
    rows = list(csv.reader(open(steps), delimiter="\t"))
    table(["step", "rc", "seconds"], rows)

open(f"{out}/summary.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
