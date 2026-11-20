"""Step 5 (E5): greedy generation agreement on the 25 frozen prompts, BF16 vs every format.

POST-TRAINING ONLY. Greedy decoding (temperature 0), MAX_TOKENS=128, prompts from /vol/data/prompts.json.
Per prompt we record: exact match, first divergence token, matching-prefix fraction.

Run from E:\\quantt\\fp_8:
    modal run sanity_check.py --fmts bf16 --smoke      # 3 prompts, nothing saved
    modal run sanity_check.py --fmts all
Outputs: E:\\quantt\\results\\sanity25\\<fmt>.json  and  agreement_vs_bf16.json
(The old 5-prompt files were archived in results/_archive_sanity5/.)
"""
import json

import modal

import config
from modal_app import app, vllm_image, volume, VOLUMES

MAX_TOKENS = 128


@app.function(image=vllm_image, gpu=config.GPU, volumes=VOLUMES, timeout=1800)
def generate(fmt: str, n_prompts: int) -> dict:
    from vllm import SamplingParams

    from vllm_common import make_llm

    config.assert_inference_only()
    volume.reload()
    prompts = json.load(open(f"{config.REMOTE_DATA}/prompts.json", encoding="utf-8"))[:n_prompts]

    llm, info = make_llm(fmt)
    print(f"[{fmt}] loaded in {info['load_seconds']}s quant={info['quantization']} cc={info['compute_capability']}")
    params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, seed=config.SEED)
    outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], params, use_tqdm=False)
    results = [
        {"prompt": p, "text": o.outputs[0].text, "token_ids": list(o.outputs[0].token_ids)}
        for p, o in zip(prompts, outs)
    ]
    return {"info": info, "results": results}


def compare(base: list, other: list) -> dict:
    per, exact, fracs = [], 0, []
    for a, b in zip(base, other):
        ta, tb = a["token_ids"], b["token_ids"]
        n = min(len(ta), len(tb))
        first = next((i for i in range(n) if ta[i] != tb[i]), None)
        same = first is None and len(ta) == len(tb)
        denom = max(len(ta), len(tb), 1)
        prefix = (n if first is None else first) / denom
        exact += same
        fracs.append(prefix)
        per.append({"prompt": a["prompt"], "exact": same, "first_divergence": first, "prefix_frac": prefix})
    return {"exact_matches": exact, "n": len(per), "exact_rate": exact / max(len(per), 1),
            "mean_prefix_frac": sum(fracs) / max(len(fracs), 1), "per_prompt": per}


@app.local_entrypoint()
def main(fmts: str = "all", smoke: bool = False) -> None:
    from vllm_common import parse_fmts

    fmt_list = parse_fmts(fmts)
    n = 3 if smoke else 25
    args = [(f, n) for f in fmt_list]
    results = list(generate.starmap(args)) if len(args) > 1 else [generate.remote(*args[0])]

    outdir = config.RESULTS_DIR / "sanity25"
    by_fmt = {r["info"]["fmt"]: r for r in results}
    if not smoke:
        outdir.mkdir(parents=True, exist_ok=True)
        for f, r in by_fmt.items():
            (outdir / f"{f}.json").write_text(json.dumps(r, indent=2), encoding="utf-8")

    base = by_fmt.get("bf16")
    if base is None and (outdir / "bf16.json").exists():
        base = json.loads((outdir / "bf16.json").read_text(encoding="utf-8"))
    if base is None:
        print("no bf16 baseline available to compare against (run with bf16 included)")
        return

    summary = {}
    print(f"\n{'fmt':<16} {'exact':>9} {'rate':>6} {'mean prefix':>12} {'load_s':>7}")
    for f, r in by_fmt.items():
        if f == "bf16":
            continue
        c = compare(base["results"], r["results"])
        summary[f] = c
        print(f"{f:<16} {c['exact_matches']:>4}/{c['n']:<4} {c['exact_rate']:>6.2f} "
              f"{c['mean_prefix_frac']:>12.3f} {r['info']['load_seconds']:>7.1f}")
    if smoke:
        print("\n[smoke mode] results not saved to disk.")
        return
    (outdir / "agreement_vs_bf16.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nsaved -> {outdir}")
