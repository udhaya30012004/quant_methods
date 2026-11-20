"""Step 2 (E1 — Track B Perplexity): evaluate real checkpoints on frozen windows via vLLM.

POST-TRAINING ONLY. Feeds the exact frozen 2048-token windows from /vol/data/ to vLLM
with prompt_logprobs=0 to obtain exact per-token NLL, loss, and perplexity.

Run from E:\\quantt\\fp_8:
    modal run eval_ppl_vllm.py --fmts bf16 --smoke
    modal run eval_ppl_vllm.py --fmts all
Outputs: E:\\quantt\\results\\trackb_ppl\\<fmt>__<set>.json and .npy (per-token NLL)
"""
import json

import modal

import config
from modal_app import app, vllm_image, volume, VOLUMES

DEFAULT_SETS = ("ppl_wikitext", "kl_chat")
BATCH_WINDOWS = 8


@app.function(image=vllm_image, gpu=config.GPU, volumes=VOLUMES, timeout=3600)
def eval_one(fmt: str, sets: list[str], n_windows: int) -> dict:
    import numpy as np
    from vllm import SamplingParams
    try:
        from vllm.inputs import TokensPrompt
    except ImportError:
        from vllm import TokensPrompt

    from vllm_common import make_llm

    config.assert_inference_only()
    volume.reload()

    # Window is 2048 tokens; vLLM needs room for at least 1 generated token.
    llm, info = make_llm(fmt, max_model_len=config.MAX_MODEL_LEN + 8)
    print(
        f"[{fmt}] loaded in {info['load_seconds']}s  quant={info['quantization']}  "
        f"cc={info['compute_capability']}  weights={info['model_weight_gib']} GiB"
    )

    params = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0, seed=config.SEED)
    out = {"info": info, "sets": {}}

    for s in sets:
        arr = np.load(f"{config.REMOTE_DATA}/{s}.npy")[:n_windows]
        nll_chunks = []
        win_losses = []
        for i in range(0, len(arr), BATCH_WINDOWS):
            batch = arr[i:i + BATCH_WINDOWS]
            prompts = [TokensPrompt(prompt_token_ids=w.astype(int).tolist()) for w in batch]
            outputs = llm.generate(prompts, params, use_tqdm=False)
            for w, o in zip(batch, outputs):
                plp = o.prompt_logprobs
                assert plp is not None and len(plp) == len(w), (
                    f"unexpected prompt_logprobs length: {None if plp is None else len(plp)} vs {len(w)}"
                )
                nlls = []
                for pos in range(1, len(w)):
                    tok = int(w[pos])
                    entry = plp[pos][tok]
                    lp = float(getattr(entry, "logprob", entry))
                    nlls.append(-lp)
                nll_arr = np.asarray(nlls, dtype=np.float32)
                nll_chunks.append(nll_arr)
                win_losses.append(float(nll_arr.mean()))

        all_nll = np.concatenate(nll_chunks)
        mean_loss = float(all_nll.mean())
        ppl = float(np.exp(mean_loss))
        stats = {
            "fmt": fmt,
            "set": s,
            "n_windows": int(len(arr)),
            "n_tokens": int(all_nll.size),
            "loss": mean_loss,
            "ppl": ppl,
            "window_losses": win_losses,
        }
        out["sets"][s] = {"stats": stats, "nll": all_nll}
        print(f"[{fmt:<14}][{s:<12}] windows={len(arr)} tokens={all_nll.size} loss={mean_loss:.4f} ppl={ppl:.3f}")

    return out


@app.local_entrypoint()
def main(fmts: str = "bf16", sets: str = "ppl_wikitext,kl_chat", smoke: bool = False) -> None:
    import numpy as np
    from vllm_common import parse_fmts

    fmt_list = parse_fmts(fmts)
    set_list = ["ppl_wikitext"] if smoke else [s.strip() for s in sets.split(",") if s.strip()]
    n_windows = 4 if smoke else 64

    outdir = config.RESULTS_DIR / "trackb_ppl"
    outdir.mkdir(parents=True, exist_ok=True)

    args = [(f, set_list, n_windows) for f in fmt_list]
    results = list(eval_one.starmap(args)) if len(args) > 1 else [eval_one.remote(*args[0])]

    print(f"\n{'fmt':<15} {'set':<13} {'windows':>7} {'loss':>8} {'ppl':>8} {'load_s':>7} {'wt_GiB':>7}")
    for res in results:
        info = res["info"]
        fmt = info["fmt"]
        for s, payload in res["sets"].items():
            st = payload["stats"]
            if not smoke:
                (outdir / f"{fmt}__{s}.json").write_text(
                    json.dumps({**st, "info": info}, indent=2), encoding="utf-8"
                )
                np.save(outdir / f"{fmt}__{s}.npy", payload["nll"])
            wt = f"{info['model_weight_gib']:.2f}" if info["model_weight_gib"] is not None else "n/a"
            print(f"{fmt:<15} {s:<13} {st['n_windows']:>7} {st['loss']:>8.4f} {st['ppl']:>8.3f} {info['load_seconds']:>7.1f} {wt:>7}")

    if smoke:
        print("\n[smoke mode] results not saved to disk.")
    else:
        print(f"\nsaved -> {outdir}")
