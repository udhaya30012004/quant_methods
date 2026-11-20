"""Step 2 (Track B): quantize the BF16 baseline to one FP8 PTQ recipe with llm-compressor.

POST-TRAINING ONLY: one-shot rounding to FP8 E4M3. No gradients, no fine-tuning.
Checkpoint recipes come from config.CHECKPOINT_RECIPES:

  w8a16_pt       : weights FP8 per-tensor,  activations BF16
  w8a16_pc       : weights FP8 per-channel, activations BF16
  w8a8_pt_dyn    : weights FP8 per-tensor,  activations FP8 per-tensor DYNAMIC (no calibration)
  w8a8_pt_static : weights FP8 per-tensor,  activations FP8 per-tensor STATIC  (calibrated on calib_chat)
  w8a8_pc        : weights FP8 per-channel, activations FP8 per-token DYNAMIC  (== FP8_DYNAMIC preset)

NOTE Track A's w8a8_pt (dynamic per-tensor activations) differs from Track B's
w8a8_pt_static. Per-tensor *dynamic* activations are evaluated in sim_eval.py.
Embeddings, norms and lm_head stay BF16.

Run from E:\\quantt\\fp_8 (one recipe per call):
    modal run quantize_fp8.py --recipe w8a16_pc
"""
import modal

import config
from modal_app import app, quant_image, volume, VOLUMES

N_CALIB = 256          # samples used to record static activation ranges (no training)


def _build_recipe(name: str):
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
    from llmcompressor.modifiers.quantization import QuantizationModifier

    r = config.CHECKPOINT_RECIPES[name]
    if name == "w8a8_pc":
        return QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=config.QUANT_IGNORE)

    w = QuantizationArgs(
        num_bits=8, type="float", symmetric=True, observer="minmax",
        strategy="tensor" if r["wscale"] == "per_tensor" else "channel",
    )
    a = None
    if name == "w8a8_pt_static":
        a = QuantizationArgs(num_bits=8, type="float", symmetric=True, observer="minmax",
                             strategy="tensor", dynamic=False)
    elif name == "w8a8_pt_dyn":
        a = QuantizationArgs(num_bits=8, type="float", symmetric=True,
                             strategy="tensor", dynamic=True)
    scheme = QuantizationScheme(targets=["Linear"], weights=w, input_activations=a)
    return QuantizationModifier(config_groups={"group_0": scheme}, ignore=config.QUANT_IGNORE)


@app.function(image=quant_image, gpu=config.GPU, volumes=VOLUMES, timeout=3600)
def quantize(recipe_name: str) -> dict:
    import collections
    import glob
    import json
    import os

    import numpy as np
    import torch
    from llmcompressor import oneshot
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert recipe_name in config.CHECKPOINT_RECIPES, f"unknown checkpoint recipe {recipe_name}; choose from {list(config.CHECKPOINT_RECIPES)}"
    volume.reload()
    src = config.FORMATS["bf16"]
    dst = config.FORMATS[recipe_name]
    assert os.path.isdir(src), f"{src} missing - run download_model.py first"

    torch.manual_seed(config.SEED)
    config.assert_inference_only()

    print(f"Loading BF16 model from {src}")
    model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained(src)
    config.assert_inference_only(model)

    r = config.CHECKPOINT_RECIPES[recipe_name]
    needs_calib = r["ascale"] == "per_tensor_static"
    recipe = _build_recipe(recipe_name)
    print(f"Recipe {recipe_name}: {r}   calibration={needs_calib}")

    if needs_calib:
        from datasets import Dataset
        ids = np.load(f"{config.REMOTE_DATA}/calib_chat.npy")[:N_CALIB]
        lens = np.load(f"{config.REMOTE_DATA}/calib_chat_lens.npy")[:N_CALIB]
        rows = [x[:n].tolist() for x, n in zip(ids, lens)]
        ds = Dataset.from_dict({"input_ids": rows, "attention_mask": [[1] * len(x) for x in rows]})
        oneshot(model=model, recipe=recipe, dataset=ds, max_seq_length=config.MAX_MODEL_LEN,
                num_calibration_samples=len(rows))
    else:
        oneshot(model=model, recipe=recipe)

    os.makedirs(dst, exist_ok=True)
    model.save_pretrained(dst, save_compressed=True)
    tokenizer.save_pretrained(dst)
    volume.commit()

    # ---- inspect what was actually saved
    dtype_params, dtype_bytes = collections.Counter(), collections.Counter()
    n_wscale = n_iscale = 0
    for shard in glob.glob(os.path.join(dst, "*.safetensors")):
        with safe_open(shard, framework="pt") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                dtype_params[str(t.dtype)] += t.numel()
                dtype_bytes[str(t.dtype)] += t.numel() * t.element_size()
                n_wscale += key.endswith("weight_scale")
                n_iscale += key.endswith("input_scale")

    with open(os.path.join(dst, "config.json")) as f:
        qcfg = json.load(f).get("quantization_config", {})

    total = sum(os.path.getsize(p) for p in glob.glob(os.path.join(dst, "*.safetensors")))
    src_total = sum(os.path.getsize(p) for p in glob.glob(os.path.join(src, "*.safetensors")))
    n_expected = model.config.num_hidden_layers * 7

    groups = {}
    for name, g in qcfg.get("config_groups", {}).items():
        w, a = g.get("weights") or {}, g.get("input_activations") or {}
        groups[name] = {
            "weights": {k: w.get(k) for k in ("type", "num_bits", "strategy", "dynamic")},
            "acts": {k: a.get(k) for k in ("type", "num_bits", "strategy", "dynamic")} if a else None,
        }

    report = {
        "recipe": recipe_name, "spec": r, "path": dst,
        "quant_bytes": total, "bf16_bytes": src_total,
        "size_ratio": total / src_total, "compression_pct": 100 * (1 - total / src_total),
        "bytes_by_dtype": dict(dtype_bytes), "elems_by_dtype": dict(dtype_params),
        "weight_scale_tensors": n_wscale, "input_scale_tensors": n_iscale, "expected_linears": n_expected,
        "format": qcfg.get("format"), "ignore": qcfg.get("ignore"), "config_groups": groups,
    }
    with open(os.path.join(dst, "size_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    volume.commit()

    print("\n--- saved checkpoint ---")
    print(f"path                : {dst}")
    print(f"safetensors size    : {total / 1e6:.2f} MB   (BF16 baseline: {src_total / 1e6:.2f} MB)")
    print(f"size ratio / saved  : {total / src_total:.3f} / {report['compression_pct']:.1f}%")
    for dt in dtype_params:
        print(f"  {dt:<22} {dtype_params[dt] / 1e6:>9.2f} M elems  {dtype_bytes[dt] / 1e6:>9.2f} MB")
    print(f"weight_scale tensors: {n_wscale}  (expect {n_expected})   input_scale tensors: {n_iscale}")
    print(f"quant format        : {qcfg.get('format')}  ignore={qcfg.get('ignore')}")
    print(json.dumps(groups, indent=2))
    return report


@app.local_entrypoint()
def main(recipe: str = "w8a16_pc") -> None:
    import json

    rep = quantize.remote(recipe)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS_DIR / f"size_{recipe}.json"
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"saved -> {out}")
