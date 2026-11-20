"""Step 7 (E4): downstream accuracy with lm-evaluation-harness on real vLLM checkpoints.

POST-TRAINING ONLY. Zero-shot by default, identical settings for every format. We build lm-eval's vLLM
wrapper DIRECTLY so the Marlin KernelConfig pin (W8A16) reaches the engine.
Per-example correctness is saved so Step 8 can run paired tests (McNemar) and bootstrap CIs.

Run from E:\\quantt\\fp_8:
    modal run eval_lm.py --fmts bf16 --smoke              # 50 examples, 2 tasks, nothing saved
    modal run eval_lm.py --fmts all                       # hellaswag, arc_easy, arc_challenge, winogrande, piqa
    modal run eval_lm.py --fmts all --tasks mmlu --fewshot 5     # optional second pass
Outputs: E:\\quantt\\results\\lm_eval\\<fmt>.json and <fmt>__samples.npz
"""
import json

import modal

import config
from modal_app import app, vllm_image, volume, VOLUMES


@app.function(image=vllm_image, gpu=config.GPU, volumes=VOLUMES, timeout=7200)
def eval_one(fmt: str, tasks: list, fewshot: int, limit: int) -> dict:
    import os
    import time

    import numpy as np

    # keep downloaded datasets on the Volume so every format reuses them
    os.environ.setdefault("HF_DATASETS_CACHE", f"{config.VOLUME_MOUNT}/hf-cache/datasets")
    os.environ.setdefault("HF_HOME", f"{config.VOLUME_MOUNT}/hf-cache")

    import lm_eval
    from lm_eval.models.vllm_causallms import VLLM
    from vllm.config import KernelConfig

    config.assert_inference_only()
    volume.reload()

    t0 = time.time()
    lm = VLLM(
        pretrained=config.FORMATS[fmt],
        dtype=config.DTYPE,
        max_model_len=config.MAX_MODEL_LEN,
        gpu_memory_utilization=config.GPU_MEM_UTIL,
        batch_size=config.LM_EVAL_BATCH,
        seed=config.SEED,
        kernel_config=KernelConfig(linear_backend_per_quant={"w8a16_fp8": "marlin"}),
    )
    load_s = time.time() - t0
    print(f"[{fmt}] lm-eval vLLM wrapper loaded in {load_s:.1f}s")

    out = {"fmt": fmt, "load_seconds": load_s, "fewshot": fewshot, "limit": limit or None,
           "tasks": {}, "versions": {}, "errors": {}}
    arrays = {}
    for task in tasks:
        t1 = time.time()
        try:
            res = lm_eval.simple_evaluate(
                model=lm, tasks=[task], num_fewshot=fewshot, limit=(limit or None),
                log_samples=True, random_seed=config.SEED, numpy_random_seed=config.SEED,
                torch_random_seed=config.SEED, fewshot_random_seed=config.SEED,
            )
        except Exception as e:                       # one broken task must not kill the whole run
            out["errors"][task] = f"{type(e).__name__}: {e}"
            print(f"[{fmt}][{task}] FAILED: {out['errors'][task]}")
            continue
        out["tasks"][task] = res["results"][task]
        out["versions"][task] = res.get("versions", {}).get(task)
        samples = res.get("samples", {}).get(task, [])
        for metric in ("acc", "acc_norm"):
            vals = [s[metric] for s in samples if metric in s]
            if vals and len(vals) == len(samples):
                arrays[f"{task}__{metric}"] = np.asarray(vals, dtype=np.float32)
        if samples:
            arrays[f"{task}__doc_id"] = np.asarray([s.get("doc_id", i) for i, s in enumerate(samples)], dtype=np.int64)
        r = out["tasks"][task]
        print(f"[{fmt:<14}][{task:<14}] n={len(samples)} "
              f"acc={r.get('acc,none')} acc_norm={r.get('acc_norm,none')} ({time.time() - t1:.0f}s)")
    return {"json": out, "npz": arrays}


@app.local_entrypoint()
def main(fmts: str = "bf16", tasks: str = "default", fewshot: int = -1, smoke: bool = False) -> None:
    import numpy as np
    from vllm_common import parse_fmts

    fmt_list = parse_fmts(fmts)
    if tasks == "default":
        task_list = list(config.EVAL_TASKS)
        tag = ""
    else:
        task_list = [t.strip() for t in tasks.split(",") if t.strip()]
        tag = "__" + "+".join(task_list)
    nshot = config.LM_EVAL_FEWSHOT if fewshot < 0 else fewshot
    if smoke:
        task_list, limit = ["hellaswag", "arc_easy"], 50
    else:
        limit = 0

    args = [(f, task_list, nshot, limit) for f in fmt_list]
    results = list(eval_one.starmap(args)) if len(args) > 1 else [eval_one.remote(*args[0])]

    outdir = config.RESULTS_DIR / "lm_eval"
    print(f"\n{'fmt':<16} {'task':<15} {'acc':>8} {'+/-':>7} {'acc_norm':>9}")
    for res in results:
        j = res["json"]
        for t, r in j["tasks"].items():
            f = lambda k: f"{r[k]:.4f}" if isinstance(r.get(k), (int, float)) else "-"
            print(f"{j['fmt']:<16} {t:<15} {f('acc,none'):>8} {f('acc_stderr,none'):>7} {f('acc_norm,none'):>9}")
        for t, e in j["errors"].items():
            print(f"{j['fmt']:<16} {t:<15} ERROR {e[:120]}")
        if not smoke:
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / f"{j['fmt']}{tag}.json").write_text(json.dumps(j, indent=2, default=str), encoding="utf-8")
            if res["npz"]:
                np.savez_compressed(outdir / f"{j['fmt']}{tag}__samples.npz", **res["npz"])
    if smoke:
        print("\n[smoke mode] results not saved to disk.")
    else:
        print(f"\nsaved -> {outdir}")
