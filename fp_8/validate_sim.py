"""Step 1 (Validation Gate): verify Track A fake-quant weights match Track B checkpoint weights.

POST-TRAINING ONLY. Loads the BF16 baseline and each Track B checkpoint's safetensors,
dequantizes the checkpoint FP8 weights (w_fp8 * weight_scale), and compares them
element-by-element against Track A's qdq(W) across all 252 quantized Linear layers.

Run from E:\\quantt\\fp_8:
    modal run validate_sim.py
    modal run validate_sim.py --recipes w8a16_pc,w8a16_pt
Outputs: E:\\quantt\\results\\validate\\sim_vs_ckpt.json
"""
import json

import modal

import config
from modal_app import app, quant_image, volume, VOLUMES

FP8_MAX = 448.0


@app.function(image=quant_image, gpu=config.GPU, volumes=VOLUMES, timeout=1800)
def validate(recipe_names: list[str]) -> dict:
    import glob
    import os

    import torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM

    config.assert_inference_only()
    volume.reload()

    def to_scale(amax):
        return amax.clamp(min=1e-12) / FP8_MAX

    def sim_quantize(w32, wscale):
        if wscale == "per_tensor":
            s = to_scale(w32.abs().amax())
        elif wscale == "per_channel":
            s = to_scale(w32.abs().amax(dim=1, keepdim=True))
        else:
            raise ValueError(wscale)
        q_fp8 = (w32 / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        return q_fp8, s, q_fp8.float() * s

    print(f"Loading BF16 baseline from {config.FORMATS['bf16']}")
    base = AutoModelForCausalLM.from_pretrained(
        config.FORMATS["bf16"], dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    config.assert_inference_only(base)
    print(f"baseline dtype={next(base.parameters()).dtype}")

    def load_ckpt_tensors(ckpt_dir: str) -> dict[str, torch.Tensor]:
        tensors = {}
        for shard in sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors"))):
            with safe_open(shard, framework="pt", device="cpu") as f:
                for k in f.keys():
                    if k.endswith(".weight") or k.endswith(".weight_scale"):
                        tensors[k] = f.get_tensor(k)
        return tensors

    results = {}
    for recipe in recipe_names:
        spec = config.CHECKPOINT_RECIPES[recipe]
        ckpt_dir = config.FORMATS[recipe]
        assert os.path.isdir(ckpt_dir), f"checkpoint missing: {ckpt_dir}"
        ckpt = load_ckpt_tensors(ckpt_dir)

        total_elems = 0
        exact_fp8 = 0
        exact_bf16 = 0
        max_abs_diff_f32 = 0.0
        max_scale_rel_diff = 0.0
        sq_err_sum = 0.0
        n_layers = 0
        worst_layer = {"name": None, "max_abs_diff": 0.0}

        for layer_idx, layer in enumerate(base.model.layers):
            for sub_name, parent in (("self_attn", layer.self_attn), ("mlp", layer.mlp)):
                for proj_name, child in parent.named_children():
                    if not isinstance(child, torch.nn.Linear):
                        continue
                    prefix = f"model.layers.{layer_idx}.{sub_name}.{proj_name}"
                    w_key, s_key = f"{prefix}.weight", f"{prefix}.weight_scale"
                    assert w_key in ckpt and s_key in ckpt, f"missing {w_key} or {s_key} in {ckpt_dir}"

                    w_bf16 = child.weight.data
                    w32 = w_bf16.float()
                    sim_fp8, sim_s, sim_dq = sim_quantize(w32, spec["wscale"])

                    ck_fp8 = ckpt[w_key].cuda()
                    ck_s = ckpt[s_key].cuda().float()
                    if ck_s.ndim == 1 and ck_s.numel() == ck_fp8.shape[0]:
                        ck_s = ck_s.unsqueeze(1)
                    ck_dq = ck_fp8.float() * ck_s

                    diff_f32 = (sim_dq - ck_dq).abs()
                    layer_max = float(diff_f32.max().item())
                    if layer_max > worst_layer["max_abs_diff"]:
                        worst_layer = {"name": prefix, "max_abs_diff": layer_max}

                    max_abs_diff_f32 = max(max_abs_diff_f32, layer_max)
                    sq_err_sum += float((diff_f32 ** 2).sum().item())

                    s_ref = sim_s.expand_as(ck_s) if sim_s.numel() == 1 and ck_s.numel() == 1 else sim_s
                    s_rel = float(((s_ref - ck_s).abs() / ck_s.abs().clamp(min=1e-12)).max().item())
                    max_scale_rel_diff = max(max_scale_rel_diff, s_rel)

                    exact_fp8 += int((sim_fp8.view(torch.uint8) == ck_fp8.view(torch.uint8)).sum().item())
                    exact_bf16 += int((sim_dq.to(torch.bfloat16) == ck_dq.to(torch.bfloat16)).sum().item())
                    total_elems += w32.numel()
                    n_layers += 1

        rmse_f32 = (sq_err_sum / total_elems) ** 0.5
        frac_fp8 = exact_fp8 / total_elems
        frac_bf16 = exact_bf16 / total_elems
        passed = frac_bf16 >= 0.999 and max_scale_rel_diff < 1e-4
        results[recipe] = {
            "spec": spec,
            "ckpt_dir": ckpt_dir,
            "layers_checked": n_layers,
            "total_elems": total_elems,
            "exact_fp8_frac": frac_fp8,
            "exact_bf16_frac": frac_bf16,
            "max_abs_diff_f32": max_abs_diff_f32,
            "rmse_f32": rmse_f32,
            "max_scale_rel_diff": max_scale_rel_diff,
            "worst_layer": worst_layer,
            "passed": passed,
        }
        status = "PASS" if passed else "FAIL"
        print(
            f"[{status}] {recipe:<14} layers={n_layers}  exact_fp8={frac_fp8:.6f}  "
            f"exact_bf16={frac_bf16:.6f}  max_diff={max_abs_diff_f32:.3e}  "
            f"scale_rel={max_scale_rel_diff:.3e}"
        )
    return results


@app.local_entrypoint()
def main(recipes: str = "w8a16_pc,w8a16_pt") -> None:
    names = [r.strip() for r in recipes.split(",") if r.strip()]
    for n in names:
        assert n in config.CHECKPOINT_RECIPES, f"unknown checkpoint recipe '{n}'"
    res = validate.remote(names)

    outdir = config.RESULTS_DIR / "validate"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "sim_vs_ckpt.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out}")
