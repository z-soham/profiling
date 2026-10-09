#!/usr/bin/env python3
"""Prefill latency of vLLM for random-token and real-text prompts, several lengths, one engine.

`vllm bench latency` only sends random token ids (np.random.randint(10000)), and each prompt
length costs a full engine start. This builds the engine once, like `vllm bench latency` does
(EngineArgs from the same command-line flags, prefix caching off), then for every length and
prompt kind times `--iters` generate calls of one prompt with max_tokens=1 after `--warmup`
untimed ones. Real text routes tokens to MoE experts differently from random ids.

    python vllm_prompt_bench.py --prompt-file prompt_text.txt --lens 256,8192 \
        --output-json out.json <vllm engine args, e.g. --model ... --max-model-len 16384>
"""
import argparse
import json
import time

import numpy as np

from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import TokensPrompt
from vllm.utils.argparse_utils import FlexibleArgumentParser


def main() -> None:
    parser = FlexibleArgumentParser(description=__doc__)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--lens", default="256,8192")
    parser.add_argument("--kinds", default="random,text")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output-json", required=True)
    parser = EngineArgs.add_cli_args(parser)
    parser.set_defaults(enable_prefix_caching=False)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    llm = LLM.from_engine_args(EngineArgs.from_cli_args(args))
    tok = llm.get_tokenizer()
    with open(args.prompt_file) as f:
        text_ids = tok.encode(f.read(), add_special_tokens=False)
    sampling = SamplingParams(n=1, temperature=1.0, top_p=1.0, ignore_eos=True, max_tokens=1)
    rng = np.random.default_rng(0)

    def prompt(kind: str, n: int, i: int) -> TokensPrompt:
        if kind == "random":  # as vllm bench latency: ids below 10000
            ids = rng.integers(0, 10000, size=n).tolist()
        else:  # the text from a different offset each call, wrapped to length
            off = (i * 997) % len(text_ids)
            ids = [text_ids[(off + j) % len(text_ids)] for j in range(n)]
        return TokensPrompt(prompt_token_ids=ids)

    results = {"text_tokens": len(text_ids), "runs": []}
    for n in [int(x) for x in args.lens.split(",")]:
        for kind in args.kinds.split(","):
            lat = []
            for i in range(args.warmup + args.iters):
                p = prompt(kind, n, i)
                t0 = time.perf_counter()
                llm.generate([p], sampling_params=sampling, use_tqdm=False)
                if i >= args.warmup:
                    lat.append(time.perf_counter() - t0)
            run = {"n_prompt": n, "kind": kind, "latencies": lat,
                   "tps": [n / x for x in lat], "tps_mean": n * len(lat) / sum(lat)}
            results["runs"].append(run)
            print(f"pp{n} {kind}: {run['tps_mean']:.1f} t/s  {[round(x, 3) for x in lat]}", flush=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
