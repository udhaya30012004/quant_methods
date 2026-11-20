"""Step 8 (R) helpers: result loading, plot style, and statistics (bootstrap CIs, paired tests, McNemar).

Pure numpy, no scipy. Run `python fp_8/stats.py --selftest` to verify the statistics on synthetic data.
Everything here is post-hoc analysis of saved results; nothing touches a model.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

FMTS = ["w8a16_pt", "w8a16_pc", "w8a8_pt_dyn", "w8a8_pc", "w8a8_pt_static"]    # quantized Track B formats
ALL_FMTS = ["bf16"] + FMTS
A_NAME = {"w8a8_pt_dyn": "w8a8_pt"}                      # Track A name for each Track B format
SETS = ["ppl_wikitext", "kl_chat"]
SET_LABEL = {"ppl_wikitext": "WikiText-2", "kl_chat": "UltraChat"}
TASKS = ["hellaswag", "arc_easy", "arc_challenge", "winogrande", "piqa"]
PRIMARY = {"hellaswag": "acc_norm", "arc_easy": "acc", "arc_challenge": "acc_norm", "winogrande": "acc", "piqa": "acc"}

# fixed colour / marker / linestyle / hatch per recipe in every plot
STYLE = {
    "bf16":           dict(color="black",   marker="*", ls="-",  hatch="",   label="BF16"),
    "w8a16_pt":       dict(color="#1f77b4", marker="o", ls="-",  hatch="",   label="W8A16 per-tensor"),
    "w8a16_pc":       dict(color="#6baed6", marker="s", ls="--", hatch="",   label="W8A16 per-channel"),
    "w8a8_pt_dyn":    dict(color="#ff7f0e", marker="^", ls="-",  hatch="",   label="W8A8 per-tensor dyn"),
    "w8a8_pc":        dict(color="#d62728", marker="D", ls="--", hatch="",   label="W8A8 per-channel/token"),
    "w8a8_pt_static": dict(color="#8c2d04", marker="v", ls="-",  hatch="//", label="W8A8 per-tensor static"),
}


# ------------------------------------------------------------------ statistics
def bootstrap_mean_ci(x, n_boot=2000, seed=0, alpha=0.05):
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap(a, b, stat=None, n_boot=2000, seed=0, alpha=0.05):
    """Paired bootstrap of stat(a[idx], b[idx]) (default: mean(a-b)). Returns (est, lo, hi, p_two_sided)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape
    stat = stat or (lambda x, y: float(np.mean(x - y)))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    vals = np.array([stat(a[i], b[i]) for i in idx])
    est = stat(a, b)
    p = 2 * min((vals <= 0).mean(), (vals >= 0).mean())
    return float(est), float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)), float(min(p, 1.0))


def mcnemar_exact(a, b):
    """Exact two-sided McNemar on paired 0/1 correctness. Returns (n_a_only, n_b_only, p)."""
    a, b = np.asarray(a).astype(bool), np.asarray(b).astype(bool)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    n = n10 + n01
    if n == 0:
        return n10, n01, 1.0
    k = min(n10, n01)
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return n10, n01, float(min(p, 1.0))


# ------------------------------------------------------------------ loading
def _j(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_results():
    R = {"sim": {}, "trackb": {}, "size": {}, "bench": {}, "lm": {}, "sanity": None, "diag": {}, "validate": None}
    for r in ["none", "fp16cast", "w8a16_pt", "w8a16_pc", "w8a8_pt", "w8a8_pc", "w8a8_pt_static"]:
        for s in SETS:
            p = RES / "sim" / f"{r}__{s}.json"
            if p.exists():
                R["sim"].setdefault(r, {})[s] = {"stats": _j(p), "kl": np.load(RES / "sim" / f"{r}__{s}.npy")}
    for f in ALL_FMTS:
        for s in SETS:
            p = RES / "trackb_ppl" / f"{f}__{s}.json"
            if p.exists():
                R["trackb"].setdefault(f, {})[s] = {"stats": _j(p), "nll": np.load(RES / "trackb_ppl" / f"{f}__{s}.npy")}
        p = RES / "size_" f"{f}.json"
        if p.exists():
            R["size"][f] = _j(p)
        p = RES / "bench" / f"{f}.json"
        if p.exists():
            R["bench"][f] = _j(p)
        p = RES / "lm_eval" / f"{f}.json"
        if p.exists():
            z = RES / "lm_eval" / f"{f}__samples.npz"
            R["lm"][f] = {"json": _j(p), "npz": dict(np.load(z)) if z.exists() else {}}
    if FMTS[0] in R["size"]:
        R["size"]["bf16"] = {"quant_bytes": R["size"][FMTS[0]]["bf16_bytes"], "compression_pct": 0.0, "size_ratio": 1.0}
    p = RES / "sanity25" / "agreement_vs_bf16.json"
    R["sanity"] = _j(p) if p.exists() else None
    p = RES / "validate" / "sim_vs_ckpt.json"
    R["validate"] = _j(p) if p.exists() else None
    for name in ("weights", "acts", "layer_err", "sensitivity"):
        p = RES / "diag" / f"{name}.json"
        if p.exists():
            R["diag"][name] = _j(p)
    z = RES / "diag" / "weight_samples.npz"
    R["diag"]["samples"] = dict(np.load(z)) if z.exists() else {}
    return R


def trackb_delta(R, fmt, s, n_boot=2000):
    """Paired (per-window) delta vs BF16 for Track B: dloss and dppl with 95% CI."""
    q = np.asarray(R["trackb"][fmt][s]["stats"]["window_losses"])
    b = np.asarray(R["trackb"]["bf16"][s]["stats"]["window_losses"])
    dl = paired_bootstrap(q, b, n_boot=n_boot)
    dp = paired_bootstrap(q, b, stat=lambda x, y: float(np.exp(x.mean()) - np.exp(y.mean())), n_boot=n_boot)
    return {"dloss": dl[0], "dloss_lo": dl[1], "dloss_hi": dl[2], "dppl": dp[0], "dppl_lo": dp[1], "dppl_hi": dp[2], "p": dl[3]}


def kl_windows(R, recipe, s):
    kl = R["sim"][recipe][s]["kl"]
    n_win = len(R["sim"][recipe][s]["stats"].get("window_loss_base", [])) or 64
    return kl.reshape(n_win, -1)


def lm_metric(R, fmt, task):
    r = R["lm"][fmt]["json"]["tasks"].get(task)
    if r is None:
        return None, None
    m = PRIMARY[task]
    return r.get(f"{m},none"), r.get(f"{m}_stderr,none")


def lm_correct(R, fmt, task):
    return R["lm"][fmt]["npz"].get(f"{task}__{PRIMARY[task]}")


def selftest():
    rng = np.random.default_rng(1)
    x = rng.normal(1.0, 0.5, size=200)
    m, lo, hi = bootstrap_mean_ci(x)
    assert lo < 1.0 < hi and abs(m - x.mean()) < 1e-12, (m, lo, hi)
    a = rng.normal(0.10, 0.05, size=64)
    b = a - 0.02 + rng.normal(0, 0.005, size=64)
    est, lo, hi, p = paired_bootstrap(a, b)
    assert lo < 0.02 < hi and p < 0.01, (est, lo, hi, p)
    est, lo, hi, p = paired_bootstrap(a, a + rng.normal(0, 0.05, size=64))
    assert p > 0.01
    # hand-computed: n10=10, n01=2 -> p = 2*(1+12+66)/4096
    a = np.array([1] * 10 + [0] * 2 + [1] * 5 + [0] * 3)
    b = np.array([0] * 10 + [1] * 2 + [1] * 5 + [0] * 3)
    n10, n01, p = mcnemar_exact(a, b)
    assert (n10, n01) == (10, 2) and abs(p - 158 / 4096) < 1e-12, (n10, n01, p)
    assert mcnemar_exact(a, a)[2] == 1.0
    print("stats selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        R = load_results()
        print({k: (list(v) if isinstance(v, dict) else v is not None) for k, v in R.items()})
