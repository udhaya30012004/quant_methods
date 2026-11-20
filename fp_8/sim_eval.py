"""Track A: simulated FP8 PTQ (fake-quant) -> loss, perplexity and FULL-VOCAB KL vs the BF16 baseline.

POST-TRAINING ONLY. Weights are rounded once to FP8 E4M3 and de-quantized back to BF16; activations are
fake-quantized inside forward(). No gradients, no training.

Recipes (config.RECIPES) plus two checks:
  none      : copy of the baseline. KL MUST be exactly 0 (determinism / harness self-check)
  fp16cast  : baseline cast to FP16. Gives the "noise floor" of a harmless precision change

Run from E:\\quantt\\fp_8:
    modal run sim_eval.py --recipes none --windows 4            # quick smoke test
    modal run sim_eval.py --recipes none,w8a16_pt,w8a16_pc,w8a8_pt,w8a8_pc,fp16cast
Outputs: E:\\quantt\\results\\sim\\<recipe>__<set>.json  and  .npy (per-token KL)
"""
import modal

import config
from modal_app import app, quant_image, volume, VOLUMES

FP8_MAX = 448.0   # E4M3
SETS = ["ppl_wikitext", "kl_chat"]
CHUNK = 512       # tokens per KL chunk (keeps float32 [chunk, vocab] small)


@app.function(image=quant_image, gpu=config.GPU, volumes=VOLUMES, timeout=7200)
def evaluate(recipe_names: list, n_windows: int) -> dict:
    import copy

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM

    torch.manual_seed(config.SEED)
    config.assert_inference_only()
    volume.reload()

    # ------------------------------------------------------------ fake-quant helpers
    def qdq(x32, scale):
        """FP8 E4M3 quantize -> dequantize in float32."""
        return (x32 / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).to(torch.float32) * scale

    def to_scale(amax):
        return amax.clamp(min=1e-12) / FP8_MAX

    class QLinear(nn.Module):
        def __init__(self, lin, wscale, ascale, static_input_scale=None):
            super().__init__()
            w = lin.weight.data.float()
            if wscale == "per_tensor":
                s = to_scale(w.abs().amax())
            elif wscale == "per_channel":
                s = to_scale(w.abs().amax(dim=1, keepdim=True))   # one scale per output channel
            else:
                raise ValueError(wscale)
            self.weight = nn.Parameter(qdq(w, s).to(lin.weight.dtype), requires_grad=False)
            self.bias = None if lin.bias is None else nn.Parameter(lin.bias.data.clone(), requires_grad=False)
            self.ascale = ascale
            if ascale == "per_tensor_static":
                assert static_input_scale is not None, "per_tensor_static requires input_scale from checkpoint"
                self.register_buffer("input_scale", static_input_scale.float().clamp(min=1e-12))

        def forward(self, x):
            if self.ascale != "none":
                xf = x.float()
                if self.ascale == "per_tensor_dynamic":
                    s = to_scale(xf.abs().amax())
                elif self.ascale == "per_token_dynamic":
                    s = to_scale(xf.abs().amax(dim=-1, keepdim=True))
                elif self.ascale == "per_tensor_static":
                    s = self.input_scale
                else:
                    raise ValueError(self.ascale)
                x = qdq(xf, s).to(x.dtype)
            return F.linear(x, self.weight, self.bias)

    def load_static_input_scales(ckpt_dir: str) -> dict:
        import glob
        import os
        from safetensors import safe_open

        scales = {}
        for shard in sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors"))):
            with safe_open(shard, framework="pt", device="cuda") as f:
                for k in f.keys():
                    if k.endswith(".input_scale"):
                        scales[k] = f.get_tensor(k)
        assert scales, f"no input_scale tensors found in {ckpt_dir}"
        return scales

    def quantize_model(model, spec, static_scales=None):
        """Replace every Linear inside the decoder layers. Embeddings, norms, lm_head untouched."""
        n = 0
        for layer_idx, layer in enumerate(model.model.layers):
            for sub_name, parent in (("self_attn", layer.self_attn), ("mlp", layer.mlp)):
                for name, child in list(parent.named_children()):
                    if isinstance(child, nn.Linear):
                        s_in = None
                        if spec["ascale"] == "per_tensor_static":
                            key = f"model.layers.{layer_idx}.{sub_name}.{name}.input_scale"
                            s_in = static_scales[key]
                        setattr(parent, name, QLinear(child, spec["wscale"], spec["ascale"], s_in))
                        n += 1
        return n

    # ------------------------------------------------------------ data + baseline
    path = config.FORMATS["bf16"]
    data = {s: np.load(f"{config.REMOTE_DATA}/{s}.npy")[:n_windows] for s in SETS}
    base = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
    config.assert_inference_only(base)
    print(f"baseline dtype={next(base.parameters()).dtype}  windows={n_windows}")

    @torch.inference_mode()
    def run_set(sim, ids_np, dtype_sim):
        kl_all, top1_all, nll_b_all, nll_q_all = [], [], [], []
        win_loss_b, win_loss_q = [], []
        for w in ids_np:
            ids = torch.from_numpy(w.astype(np.int64)).cuda().unsqueeze(0)
            lb = base(ids).logits[0, :-1]                          # [T-1, V] bf16
            lq = sim(ids).logits[0, :-1]
            tgt = ids[0, 1:]
            w_nb, w_nq = [], []
            for i in range(0, lb.shape[0], CHUNK):
                a = lb[i:i + CHUNK].float()
                b = lq[i:i + CHUNK].float()
                t = tgt[i:i + CHUNK]
                lpa, lpb = F.log_softmax(a, -1), F.log_softmax(b, -1)
                kl_all.append((lpa.exp() * (lpa - lpb)).sum(-1).cpu())
                top1_all.append((a.argmax(-1) == b.argmax(-1)).cpu())
                nb_c = -lpa.gather(-1, t[:, None])[:, 0].cpu()
                nq_c = -lpb.gather(-1, t[:, None])[:, 0].cpu()
                nll_b_all.append(nb_c)
                nll_q_all.append(nq_c)
                w_nb.append(nb_c)
                w_nq.append(nq_c)
            win_loss_b.append(float(torch.cat(w_nb).mean().item()))
            win_loss_q.append(float(torch.cat(w_nq).mean().item()))
        kl = torch.cat(kl_all).numpy()
        nb, nq = torch.cat(nll_b_all).numpy(), torch.cat(nll_q_all).numpy()
        top1 = torch.cat(top1_all).numpy()
        q = lambda p: float(np.quantile(kl, p))
        return {
            "n_tokens": int(kl.size),
            "loss_base": float(nb.mean()), "loss_quant": float(nq.mean()),
            "ppl_base": float(np.exp(nb.mean())), "ppl_quant": float(np.exp(nq.mean())),
            "delta_loss": float(nq.mean() - nb.mean()),
            "kl_mean": float(kl.mean()), "kl_median": q(0.5), "kl_p90": q(0.9), "kl_p99": q(0.99),
            "kl_max": float(kl.max()), "top1_agree": float(top1.mean()),
            "window_loss_base": win_loss_b, "window_loss_quant": win_loss_q,
        }, kl.astype(np.float32)

    out = {}
    for name in recipe_names:
        if name == "none":
            sim, spec = copy.deepcopy(base), {"note": "baseline copy"}
        elif name == "fp16cast":
            sim, spec = copy.deepcopy(base).half(), {"note": "fp16 cast"}
        else:
            spec = config.RECIPES[name]
            sim = copy.deepcopy(base)
            static_scales = (
                load_static_input_scales(config.FORMATS["w8a8_pt_static"])
                if spec["ascale"] == "per_tensor_static" else None
            )
            n = quantize_model(sim, spec, static_scales=static_scales)
            print(f"{name}: replaced {n} Linear layers  spec={spec}")
        sim.eval()
        for p in sim.parameters():
            p.requires_grad_(False)
        out[name] = {"spec": spec}
        for s in SETS:
            stats, kl = run_set(sim, data[s], None)
            out[name][s] = {"stats": stats, "kl": kl}
            print(f"[{name:<9}][{s:<12}] loss {stats['loss_base']:.4f}->{stats['loss_quant']:.4f} "
                  f"ppl {stats['ppl_base']:.3f}->{stats['ppl_quant']:.3f} KL {stats['kl_mean']:.6f} "
                  f"top1 {stats['top1_agree']:.4f}")
        del sim
        torch.cuda.empty_cache()
    return out


@app.local_entrypoint()
def main(recipes: str = "none", windows: int = 64) -> None:
    import json

    import numpy as np

    names = [r.strip() for r in recipes.split(",") if r.strip()]
    for n in names:
        assert n in ("none", "fp16cast") or n in config.RECIPES, f"unknown recipe {n}"
    res = evaluate.remote(names, windows)

    outdir = config.RESULTS_DIR / "sim"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'recipe':<10} {'set':<13} {'loss_b':>8} {'loss_q':>8} {'dLoss':>8} {'ppl_q':>8} {'KL mean':>10} {'KL p99':>9} {'top1':>7}")
    for name, d in res.items():
        for s in SETS:
            st = d[s]["stats"]
            (outdir / f"{name}__{s}.json").write_text(
                json.dumps({"recipe": name, "spec": d["spec"], "set": s, **st}, indent=2), encoding="utf-8")
            np.save(outdir / f"{name}__{s}.npy", d[s]["kl"])
            print(f"{name:<10} {s:<13} {st['loss_base']:>8.4f} {st['loss_quant']:>8.4f} {st['delta_loss']:>+8.4f} "
                  f"{st['ppl_quant']:>8.3f} {st['kl_mean']:>10.6f} {st['kl_p99']:>9.5f} {st['top1_agree']:>7.4f}")
    print(f"\nsaved -> {outdir}")
