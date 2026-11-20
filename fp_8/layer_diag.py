"""E3: per-layer diagnostics ("why does each recipe behave as it does").

POST-TRAINING ONLY. Pure inference/measurement on a FIXED diagnostic batch (first 4 WikiText + first 4 chat
windows). Nothing here feeds back into quantization. No gradients.

Run from E:\\quantt\\fp_8 (one part per command):
    modal run layer_diag.py --part weights
    modal run layer_diag.py --part acts
    modal run layer_diag.py --part layererr
    modal run layer_diag.py --part sensitivity
    (optional) modal run layer_diag.py --part weights --smoke     # tiny check, nothing saved

Outputs -> E:\\quantt\\results\\diag\\
    weights.json, weight_samples.npz, acts.json, layer_err.json, sensitivity.json
"""
import modal

import config
from modal_app import app, quant_image, volume, VOLUMES

FP8_MAX = 448.0
FP8_MIN_NORMAL = 2.0 ** -6          # E4M3 smallest normal (in the scaled domain)
N_DIAG = 4                          # windows per set
SAMPLE_LAYERS = [0, 9, 18, 27, 35]
PROJS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
SAMPLES_TOTAL = 200_000
CHUNK = 512
SENS_RECIPES = ["w8a8_pt", "w8a8_pc"]


def _helpers():
    """Build torch-dependent helpers lazily (torch only exists inside the container)."""
    import types

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def qdq(x32, scale):
        return (x32 / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).to(torch.float32) * scale

    def to_scale(amax):
        return amax.clamp(min=1e-12) / FP8_MAX

    def weight_scale(w, wscale):
        if wscale == "per_tensor":
            return to_scale(w.abs().amax())
        return to_scale(w.abs().amax(dim=1, keepdim=True))

    def act_scale(xf, ascale, static=None):
        if ascale == "per_tensor_dynamic":
            return to_scale(xf.abs().amax())
        if ascale == "per_token_dynamic":
            return to_scale(xf.abs().amax(dim=-1, keepdim=True))
        if ascale == "per_tensor_static":
            return static
        raise ValueError(ascale)

    class QLinear(nn.Module):
        def __init__(self, lin, wscale, ascale, static=None):
            super().__init__()
            w = lin.weight.data.float()
            self.weight = nn.Parameter(qdq(w, weight_scale(w, wscale)).to(lin.weight.dtype), requires_grad=False)
            self.bias = None if lin.bias is None else nn.Parameter(lin.bias.data.clone(), requires_grad=False)
            self.ascale = ascale
            self.static = None if static is None else static.float().clamp(min=1e-12)

        def forward(self, x):
            if self.ascale != "none":
                xf = x.float()
                x = qdq(xf, act_scale(xf, self.ascale, self.static)).to(x.dtype)
            return F.linear(x, self.weight, self.bias)

    def iter_linears(model):
        for li, layer in enumerate(model.model.layers):
            for sub, parent in (("self_attn", layer.self_attn), ("mlp", layer.mlp)):
                for name, child in list(parent.named_children()):
                    if isinstance(child, nn.Linear):
                        yield li, sub, name, parent, child

    def load_static_scales():
        import glob
        import os

        from safetensors import safe_open
        out = {}
        for shard in sorted(glob.glob(os.path.join(config.FORMATS["w8a8_pt_static"], "*.safetensors"))):
            with safe_open(shard, framework="pt", device="cuda") as f:
                for k in f.keys():
                    if k.endswith(".input_scale"):
                        out[k] = f.get_tensor(k)
        assert out, "no input_scale tensors in static checkpoint"
        return out

    def swap(model, spec, layers=None, projs=None, static=None):
        """Replace selected Linears with fake-quant versions. Returns list to pass to restore()."""
        saved = []
        for li, sub, name, parent, child in list(iter_linears(model)):
            if layers is not None and li not in layers:
                continue
            if projs is not None and name not in projs:
                continue
            s_in = None
            if spec["ascale"] == "per_tensor_static":
                s_in = static[f"model.layers.{li}.{sub}.{name}.input_scale"]
            saved.append((parent, name, child))
            setattr(parent, name, QLinear(child, spec["wscale"], spec["ascale"], s_in))
        return saved

    def restore(saved):
        for parent, name, child in saved:
            setattr(parent, name, child)

    def sqnr_db(sig, err):
        return float(10.0 * torch.log10(torch.as_tensor(max(sig, 1e-30) / max(err, 1e-30))))

    return types.SimpleNamespace(
        torch=torch, nn=nn, F=F, qdq=qdq, to_scale=to_scale, weight_scale=weight_scale,
        act_scale=act_scale, QLinear=QLinear, iter_linears=iter_linears,
        load_static_scales=load_static_scales, swap=swap, restore=restore, sqnr_db=sqnr_db)


def _load(H, n_diag):
    import numpy as np
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        config.FORMATS["bf16"], dtype=H.torch.bfloat16, device_map="cuda").eval()
    config.assert_inference_only(model)
    for p in model.parameters():
        p.requires_grad_(False)
    wins = []
    for s in ("ppl_wikitext", "kl_chat"):
        arr = np.load(f"{config.REMOTE_DATA}/{s}.npy")[:n_diag]
        wins += [H.torch.from_numpy(w.astype(np.int64)).cuda().unsqueeze(0) for w in arr]
    return model, wins


# ------------------------------------------------------------------ part: weights
def part_weights(H, model, smoke):
    import numpy as np
    torch = H.torch
    gen = torch.Generator().manual_seed(config.SEED)
    per_mat = SAMPLES_TOTAL // (len(SAMPLE_LAYERS) * len(PROJS))
    rows, samples = [], {}
    for li, sub, name, parent, lin in H.iter_linears(model):
        if smoke and li > 1:
            break
        w = lin.weight.data.float()
        sig = float((w ** 2).mean())
        row = {"layer": li, "proj": name, "shape": list(w.shape), "tensor_amax": float(w.abs().max())}
        ch = w.abs().amax(dim=1)
        qs = torch.quantile(ch, torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], device=ch.device)).tolist()
        row["pc_amax_quantiles"] = qs                      # min, p10, median, p90, max
        row["pc_amax_ratio_max_over_median"] = qs[4] / max(qs[2], 1e-12)
        for tag, mode in (("pt", "per_tensor"), ("pc", "per_channel")):
            s = H.weight_scale(w, mode)
            u = w / s
            q8 = u.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).to(torch.float32)
            wq = q8 * s
            mse = float(((wq - w) ** 2).mean())
            row[f"mse_{tag}"] = mse
            row[f"sqnr_db_{tag}"] = H.sqnr_db(sig, mse)
            row[f"flush_zero_frac_{tag}"] = float(((q8 == 0) & (w != 0)).float().mean())
            row[f"subnormal_frac_{tag}"] = float(((q8.abs() < FP8_MIN_NORMAL) & (q8 != 0)).float().mean())
            if li in SAMPLE_LAYERS:
                idx = torch.randperm(w.numel(), generator=gen)[:per_mat].to(w.device)
                samples[f"L{li}.{name}.{tag}_q"] = wq.flatten()[idx].cpu().numpy().astype(np.float32)
                if tag == "pt":
                    samples[f"L{li}.{name}.w"] = w.flatten()[idx].cpu().numpy().astype(np.float32)
        rows.append(row)
    print(f"[weights] {len(rows)} matrices")
    return {"json": {"n_layers_matrices": len(rows), "rows": rows}, "npz": samples}


# ------------------------------------------------------------------ part: acts
def part_acts(H, model, wins, smoke):
    torch = H.torch
    st = {}

    def mk(key):
        def f(mod, args):
            x = args[0].float().reshape(-1, args[0].shape[-1])
            d = st.setdefault(key, dict(amax=0.0, ch=None, tok=[], kurt=[], med=[], sp=0.0, ep=0.0, st=0.0, et=0.0))
            ax = x.abs()
            d["amax"] = max(d["amax"], float(ax.max()))
            ch = ax.amax(0)
            d["ch"] = ch if d["ch"] is None else torch.maximum(d["ch"], ch)
            d["tok"].append(ax.amax(-1).cpu())
            d["kurt"].append(float((((x - x.mean()) / x.std()) ** 4).mean()))
            idx = torch.randint(0, ax.numel(), (1_000_000,), device=ax.device)
            d["med"].append(float(ax.flatten()[idx].median()))
            sig = float((x ** 2).sum())
            d["sp"] += sig
            d["ep"] += float(((H.qdq(x, H.to_scale(ax.max())) - x) ** 2).sum())
            d["st"] += sig
            d["et"] += float(((H.qdq(x, H.to_scale(ax.amax(-1, keepdim=True))) - x) ** 2).sum())
        return f

    hooks = []
    for li, sub, name, parent, lin in H.iter_linears(model):
        if smoke and li > 1:
            continue
        hooks.append(lin.register_forward_pre_hook(mk((li, sub, name))))
    for ids in (wins[:1] if smoke else wins):
        model(ids)
    for h in hooks:
        h.remove()
    rows = []
    for (li, sub, name), d in sorted(st.items()):
        tok = torch.cat(d["tok"])
        q = torch.quantile(tok, torch.tensor([0.5, 0.99, 1.0])).tolist()
        top = d["ch"].topk(10)
        med = sum(d["med"]) / len(d["med"])
        rows.append({
            "layer": li, "proj": name, "tensor_amax": d["amax"],
            "tok_amax_p50": q[0], "tok_amax_p99": q[1], "tok_amax_max": q[2],
            "tok_amax_max_over_median": q[2] / max(q[0], 1e-12),
            "ch_top10_amax": top.values.tolist(), "ch_top10_idx": top.indices.tolist(),
            "kurtosis": sum(d["kurt"]) / len(d["kurt"]),
            "outlier_factor": d["amax"] / max(med, 1e-12),
            "act_sqnr_db_per_tensor": H.sqnr_db(d["sp"], d["ep"]),
            "act_sqnr_db_per_token": H.sqnr_db(d["st"], d["et"]),
        })
    print(f"[acts] {len(rows)} linears")
    return {"json": {"n_linears": len(rows), "n_windows": len(wins), "rows": rows}, "npz": {}}


# ------------------------------------------------------------------ part: layererr
def part_layererr(H, model, wins, smoke):
    import copy
    torch, F = H.torch, H.F
    static = H.load_static_scales()
    recipes = list(config.RECIPES)
    loc = {}

    def mk(li, sub, name, lin):
        def f(mod, args):
            x = args[0]
            ref = F.linear(x, lin.weight, lin.bias).float().reshape(-1, lin.weight.shape[0])
            d = loc.setdefault(f"L{li}.{name}", {r: dict(sig=0.0, err=0.0, cos=0.0, n=0) for r in recipes})
            for r in recipes:
                spec = config.RECIPES[r]
                w = lin.weight.float()
                wq = H.qdq(w, H.weight_scale(w, spec["wscale"])).to(lin.weight.dtype)
                if spec["ascale"] == "none":
                    xq = x
                else:
                    xf = x.float()
                    s_in = static[f"model.layers.{li}.{sub}.{name}.input_scale"] if spec["ascale"] == "per_tensor_static" else None
                    xq = H.qdq(xf, H.act_scale(xf, spec["ascale"], s_in)).to(x.dtype)
                y = F.linear(xq, wq, lin.bias).float().reshape(-1, lin.weight.shape[0])
                e = d[r]
                e["sig"] += float((ref ** 2).sum())
                e["err"] += float(((ref - y) ** 2).sum())
                e["cos"] += float(F.cosine_similarity(ref, y, dim=-1).sum())
                e["n"] += ref.shape[0]
        return f

    hooks = []
    for li, sub, name, parent, lin in H.iter_linears(model):
        if smoke and li > 1:
            continue
        hooks.append(lin.register_forward_pre_hook(mk(li, sub, name, lin)))
    use = wins[:1] if smoke else wins
    for ids in use:
        model(ids)
    for h in hooks:
        h.remove()
    local = {k: {r: {"sqnr_db": H.sqnr_db(e["sig"], e["err"]), "cos_mean": e["cos"] / e["n"]}
                 for r, e in v.items()} for k, v in loc.items()}

    # accumulated error: residual stream after each block, whole model quantized
    n_layers = len(model.model.layers)
    acc = {}
    base_out = {}

    def bh(i):
        def f(m, a, o):
            base_out[i] = (o[0] if isinstance(o, tuple) else o).detach()
        return f
    bhooks = [l.register_forward_hook(bh(i)) for i, l in enumerate(model.model.layers)]
    for r in (recipes[:1] if smoke else recipes):
        spec = config.RECIPES[r]
        saved = H.swap(model, spec, static=static)
        cos = [0.0] * n_layers
        rel = [0.0] * n_layers
        nums = [0.0] * n_layers
        dens = [0.0] * n_layers
        q_out = {}

        def qh(i):
            def f(m, a, o):
                q_out[i] = (o[0] if isinstance(o, tuple) else o).detach()
            return f
        qhooks = [l.register_forward_hook(qh(i)) for i, l in enumerate(model.model.layers)]
        # base_out must come from the unquantized model: recompute per window with restore trick
        for ids in use:
            H.restore(saved)
            model(ids)
            snap = {i: base_out[i].clone() for i in range(n_layers)}
            saved = H.swap(model, spec, static=static)
            model(ids)
            for i in range(n_layers):
                a = snap[i].float().reshape(-1, snap[i].shape[-1])
                b = q_out[i].float().reshape(-1, a.shape[-1])
                cos[i] += float(F.cosine_similarity(a, b, dim=-1).mean())
                nums[i] += float(((a - b) ** 2).sum())
                dens[i] += float((a ** 2).sum())
        for h in qhooks:
            h.remove()
        H.restore(saved)
        acc[r] = {"cos": [c / len(use) for c in cos], "rel_err": [(n / max(d_, 1e-30)) ** 0.5 for n, d_ in zip(nums, dens)]}
        print(f"[layererr] {r}: final-block cos={acc[r]['cos'][-1]:.5f}")
        torch.cuda.empty_cache()
    for h in bhooks:
        h.remove()
    return {"json": {"recipes": recipes, "local": local, "accumulated": acc}, "npz": {}}


# ------------------------------------------------------------------ part: sensitivity
def part_sensitivity(H, model, wins, smoke):
    torch, F = H.torch, H.F
    n_layers = len(model.model.layers)
    use = wins[:1] if smoke else wins
    with torch.inference_mode():
        base_logits = [model(ids).logits[0, :-1].clone() for ids in use]   # bf16, on GPU
    half = len(use) // 2 if len(use) > 1 else 1

    def kl_eval():
        per_win = []
        for ids, lb in zip(use, base_logits):
            lq = model(ids).logits[0, :-1]
            tot = 0.0
            for i in range(0, lb.shape[0], CHUNK):
                a = F.log_softmax(lb[i:i + CHUNK].float(), -1)
                b = F.log_softmax(lq[i:i + CHUNK].float(), -1)
                tot += float((a.exp() * (a - b)).sum())
            per_win.append(tot / lb.shape[0])
        return {"kl_mean": sum(per_win) / len(per_win),
                "kl_wiki": sum(per_win[:half]) / max(half, 1),
                "kl_chat": sum(per_win[half:]) / max(len(per_win) - half, 1)}

    out = {"windows": len(use), "recipes": SENS_RECIPES, "per_layer": {}, "per_proj": {}, "all_layers": {}}
    for r in SENS_RECIPES:
        spec = config.RECIPES[r]
        saved = H.swap(model, spec)
        out["all_layers"][r] = kl_eval()
        H.restore(saved)
        print(f"[sens] {r} ALL layers kl={out['all_layers'][r]['kl_mean']:.6f}")
        out["per_layer"][r], out["per_proj"][r] = {}, {}
        for li in (range(2) if smoke else range(n_layers)):
            saved = H.swap(model, spec, layers={li})
            out["per_layer"][r][str(li)] = kl_eval()
            H.restore(saved)
            torch.cuda.empty_cache()
        print(f"[sens] {r} per-layer done")
        for p in (PROJS[:2] if smoke else PROJS):
            saved = H.swap(model, spec, projs={p})
            out["per_proj"][r][p] = kl_eval()
            H.restore(saved)
            torch.cuda.empty_cache()
        print(f"[sens] {r} per-proj done")
    return {"json": out, "npz": {}}


@app.function(image=quant_image, gpu=config.GPU, volumes=VOLUMES, timeout=7200)
def run_diag(part: str, smoke: bool) -> dict:
    H = _helpers()
    config.assert_inference_only()
    volume.reload()
    model, wins = _load(H, 1 if smoke else N_DIAG)
    print(f"[{part}] dtype={next(model.parameters()).dtype} windows={len(wins)} smoke={smoke}")
    with H.torch.inference_mode():
        if part == "weights":
            return part_weights(H, model, smoke)
        if part == "acts":
            return part_acts(H, model, wins, smoke)
        if part == "layererr":
            return part_layererr(H, model, wins, smoke)
        if part == "sensitivity":
            return part_sensitivity(H, model, wins, smoke)
    raise ValueError(part)


@app.local_entrypoint()
def main(part: str = "weights", smoke: bool = False) -> None:
    import json

    import numpy as np

    assert part in ("weights", "acts", "layererr", "sensitivity"), part
    res = run_diag.remote(part, smoke)
    j = res["json"]
    if part == "weights":
        rows = j["rows"]
        for tag in ("pt", "pc"):
            print(f"  mean SQNR dB [{tag}] = {np.mean([r['sqnr_db_' + tag] for r in rows]):.2f}   "
                  f"mean flush-to-zero = {np.mean([r['flush_zero_frac_' + tag] for r in rows]):.2e}   "
                  f"mean subnormal = {np.mean([r['subnormal_frac_' + tag] for r in rows]):.2e}")
        worst = sorted(rows, key=lambda r: -r["pc_amax_ratio_max_over_median"])[:5]
        print("  largest per-channel amax spread (max/median):",
              [(r["layer"], r["proj"], round(r["pc_amax_ratio_max_over_median"], 1)) for r in worst])
    elif part == "acts":
        worst = sorted(j["rows"], key=lambda r: -r["outlier_factor"])[:6]
        print("  top outlier-factor inputs:", [(r["layer"], r["proj"], round(r["outlier_factor"], 1)) for r in worst])
    elif part == "layererr":
        for r, a in j["accumulated"].items():
            print(f"  accumulated cos after last block [{r}] = {a['cos'][-1]:.5f}  rel_err={a['rel_err'][-1]:.4f}")
    else:
        for r in j["recipes"]:
            pl = j["per_layer"][r]
            top = sorted(pl.items(), key=lambda kv: -kv[1]["kl_mean"])[:5]
            print(f"  [{r}] all-layers KL={j['all_layers'][r]['kl_mean']:.5f}  "
                  f"most sensitive layers={[(k, round(v['kl_mean'], 5)) for k, v in top]}")
            pp = sorted(j["per_proj"][r].items(), key=lambda kv: -kv[1]["kl_mean"])[:3]
            print(f"      most sensitive projections={[(k, round(v['kl_mean'], 5)) for k, v in pp]}")
    if smoke:
        print("\n[smoke mode] results not saved to disk.")
        return
    outdir = config.RESULTS_DIR / "diag"
    outdir.mkdir(parents=True, exist_ok=True)
    fname = {"weights": "weights.json", "acts": "acts.json", "layererr": "layer_err.json",
             "sensitivity": "sensitivity.json"}[part]
    (outdir / fname).write_text(json.dumps(j, indent=1), encoding="utf-8")
    if res["npz"]:
        np.savez_compressed(outdir / "weight_samples.npz", **res["npz"])
    print(f"\nsaved -> {outdir / fname}")
