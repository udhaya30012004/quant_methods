"""Step 6 (B1): speed & memory benchmark of every format on the SAME L4 GPU via vLLM.

POST-TRAINING ONLY. Synthetic fixed-seed random token prompts (identical across formats), greedy decoding,
ignore_eos=True so every request generates exactly BENCH_OUTPUT_TOKENS tokens. Prefix caching is OFF.

Per (batch, input_len) cell we time, with warmup then BENCH_REPEATS repeats (median reported):
  * prefill proxy / TTFT : generate(max_tokens=1)
  * end-to-end latency   : generate(max_tokens=OUT)
  * decode TPOT          : (e2e - prefill) / (OUT - 1)
  * output tok/s and total tok/s
plus model weight GiB, KV-cache capacity lines, kernel log lines, compute capability.

Run from E:\\quantt\\fp_8 (formats run one after another so timings are not disturbed by each other):
    modal run bench_speed.py --fmts bf16 --smoke      # tiny grid, nothing saved
    modal run bench_speed.py --fmts all
Outputs: E:\\quantt\\results\\bench\\<fmt>.json
"""
import json

import modal

import config
from modal_app import app, vllm_image, volume, VOLUMES


@app.function(image=vllm_image, gpu=config.GPU, volumes=VOLUMES, timeout=3600)
def bench_one(fmt: str, smoke: bool) -> dict:
    import statistics
    import time

    import numpy as np
    from vllm import SamplingParams
    try:
        from vllm.inputs import TokensPrompt
    except ImportError:
        from vllm import TokensPrompt

    from vllm_common import make_llm

    config.assert_inference_only()
    volume.reload()

    batches = [1, 8] if smoke else config.BENCH_BATCH_SIZES
    inputs = [128] if smoke else config.BENCH_INPUT_TOKENS
    repeats = 2 if smoke else config.BENCH_REPEATS
    warmup = 1 if smoke else config.BENCH_WARMUP
    out_tok = config.BENCH_OUTPUT_TOKENS

    llm, info = make_llm(
        fmt, max_model_len=max(inputs) + out_tok + 8,
        max_num_seqs=max(batches), enable_prefix_caching=False,
    )
    print(f"[{fmt}] loaded in {info['load_seconds']}s quant={info['quantization']} "
          f"cc={info['compute_capability']} weights={info['model_weight_gib']} GiB")
    vocab_hi = 100000      # stay well inside the vocab, avoid special tokens

    def make_prompts(bs, inp, seed):
        rng = np.random.default_rng(seed)
        return [TokensPrompt(prompt_token_ids=rng.integers(1000, vocab_hi, size=inp).tolist()) for _ in range(bs)]

    def timed(prompts, max_tokens):
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True, seed=config.SEED)
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        n_out = sum(len(o.outputs[0].token_ids) for o in outs)
        return dt, n_out

    cells = []
    for bs in batches:
        for inp in inputs:
            for w in range(warmup):
                timed(make_prompts(bs, inp, 1000 + w), out_tok)
            pre, e2e = [], []
            n_out_last = 0
            for r in range(repeats):
                prompts = make_prompts(bs, inp, 2000 + r)        # same seeds for every format
                dt_p, _ = timed(prompts, 1)
                dt_e, n_out_last = timed(prompts, out_tok)
                pre.append(dt_p)
                e2e.append(dt_e)
            m_pre, m_e2e = statistics.median(pre), statistics.median(e2e)
            tpot = max(m_e2e - m_pre, 0.0) / max(out_tok - 1, 1)
            cell = {
                "batch": bs, "input_tokens": inp, "output_tokens": out_tok,
                "generated_tokens_last_run": n_out_last,
                "prefill_s_median": m_pre, "e2e_s_median": m_e2e,
                "tpot_ms": tpot * 1000.0,
                "output_tok_per_s": bs * out_tok / m_e2e,
                "total_tok_per_s": bs * (inp + out_tok) / m_e2e,
                "prefill_s_all": pre, "e2e_s_all": e2e,
            }
            cells.append(cell)
            print(f"[{fmt:<14}] bs={bs:<4} in={inp:<4} prefill={m_pre:.3f}s e2e={m_e2e:.3f}s "
                  f"TPOT={tpot * 1000:.2f}ms out_tok/s={cell['output_tok_per_s']:.1f}")

    import torch
    free_b, total_b = torch.cuda.mem_get_info(0)
    info["cuda_mem_used_gib_after_bench"] = round((total_b - free_b) / (1024 ** 3), 3)
    return {"info": info, "grid": {"batches": batches, "inputs": inputs, "repeats": repeats,
                                   "warmup": warmup, "output_tokens": out_tok}, "cells": cells}


@app.local_entrypoint()
def main(fmts: str = "bf16", smoke: bool = False) -> None:
    from vllm_common import parse_fmts

    fmt_list = parse_fmts(fmts)
    outdir = config.RESULTS_DIR / "bench"
    results = {}
    for f in fmt_list:                       # sequential on purpose: fair, undisturbed timings
        print(f"\n===== benchmarking {f} =====")
        res = bench_one.remote(f, smoke)
        results[f] = res
        if not smoke:
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / f"{f}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    print(f"\n{'fmt':<16} {'bs':>4} {'in':>5} {'prefill_s':>10} {'e2e_s':>8} {'TPOT_ms':>8} {'out_tok/s':>10}")
    for f, res in results.items():
        for c in res["cells"]:
            print(f"{f:<16} {c['batch']:>4} {c['input_tokens']:>5} {c['prefill_s_median']:>10.3f} "
                  f"{c['e2e_s_median']:>8.3f} {c['tpot_ms']:>8.2f} {c['output_tok_per_s']:>10.1f}")
        i = res["info"]
        print(f"  {f}: weights={i['model_weight_gib']} GiB  mem_used_after={i['cuda_mem_used_gib_after_bench']} GiB  "
              f"cc={i['compute_capability']}")
    if smoke:
        print("\n[smoke mode] results not saved to disk.")
    else:
        print(f"\nsaved -> {outdir}")
