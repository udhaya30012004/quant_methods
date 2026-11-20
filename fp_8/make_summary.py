"""Step 8 (R): build results/summary/summary.md + summary.csv and run the scientific consistency gates.

Run locally:  python fp_8/make_summary.py
"""
import csv
import math

import numpy as np

from stats import (ALL_FMTS, A_NAME, FMTS, PRIMARY, RES, SETS, SET_LABEL, TASKS, STYLE, bootstrap_mean_ci,
                   kl_windows, lm_correct, lm_metric, load_results, mcnemar_exact, paired_bootstrap, trackb_delta)

OUT = RES / "summary"
LM_RANGE = {"hellaswag": (0.70, 0.78), "arc_easy": (0.70, 0.80), "arc_challenge": (0.40, 0.52),
            "winogrande": (0.62, 0.72), "piqa": (0.76, 0.81)}


def cell(R, f, bs, inp):
    for c in R["bench"].get(f, {}).get("cells", []):
        if c["batch"] == bs and c["input_tokens"] == inp:
            return c
    return None


def sig(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def build_rows(R):
    rows = []
    for f in ALL_FMTS:
        row = {"format": f, "label": STYLE[f]["label"]}
        sz = R["size"].get(f)
        if sz:
            row["size_MB"] = sz["quant_bytes"] / 1e6
            row["weights_GiB"] = sz["quant_bytes"] / 2 ** 30
            row["compression_pct"] = sz["compression_pct"]
        for s in SETS:
            tag = "wiki" if s == "ppl_wikitext" else "chat"
            tb = R["trackb"].get(f, {}).get(s)
            if tb:
                row[f"ppl_B_{tag}"] = tb["stats"]["ppl"]
                if f != "bf16":
                    d = trackb_delta(R, f, s)
                    row[f"dloss_B_{tag}"] = d["dloss"]
                    row[f"dloss_B_{tag}_ci"] = f"[{d['dloss_lo']:+.4f}, {d['dloss_hi']:+.4f}]"
                    row[f"dppl_B_{tag}"] = d["dppl"]
                    row[f"dloss_B_{tag}_p"] = d["p"]
            a = R["sim"].get(A_NAME.get(f, f), {}).get(s)
            if a:
                st = a["stats"]
                row[f"ppl_A_{tag}"] = st["ppl_quant"]
                if f != "bf16":
                    m, lo, hi = bootstrap_mean_ci(kl_windows(R, A_NAME.get(f, f), s).mean(axis=1))
                    row[f"kl_A_{tag}"] = st["kl_mean"]
                    row[f"kl_A_{tag}_ci"] = f"[{lo:.5f}, {hi:.5f}]"
                    row[f"kl_p99_A_{tag}"] = st["kl_p99"]
                    row[f"top1_A_{tag}"] = st["top1_agree"]
        accs = []
        for t in TASKS:
            v, se = lm_metric(R, f, t) if f in R["lm"] else (None, None)
            if v is None:
                continue
            row[f"acc_{t}"] = v
            accs.append(v)
            if f != "bf16" and "bf16" in R["lm"]:
                a, b = lm_correct(R, f, t), lm_correct(R, "bf16", t)
                if a is not None and b is not None and len(a) == len(b):
                    d = a - b
                    row[f"dacc_{t}"] = float(d.mean())
                    row[f"dacc_{t}_p"] = mcnemar_exact(a, b)[2]
        if accs:
            row["acc_avg"] = float(np.mean(accs))
        for bs in (1, 32, 128):
            c = cell(R, f, bs, 128)
            if c:
                row[f"tok_s_b{bs}"] = c["output_tok_per_s"]
        c = cell(R, f, 1, 128)
        if c:
            row["tpot_ms_b1"] = c["tpot_ms"]
        c = cell(R, f, 128, 512)
        if c:
            row["prefill_s_b128_in512"] = c["prefill_s_median"]
            row["tok_s_b128_in512"] = c["output_tok_per_s"]
        if R["sanity"] and f in R["sanity"]:
            row["gen_exact"] = f"{R['sanity'][f]['exact_matches']}/{R['sanity'][f]['n']}"
            row["gen_prefix"] = R["sanity"][f]["mean_prefix_frac"]
        rows.append(row)
    # proper average-accuracy delta vs bf16
    base = next(r for r in rows if r["format"] == "bf16").get("acc_avg")
    for r in rows:
        r.pop("dacc_avg", None)
        if base is not None and "acc_avg" in r and r["format"] != "bf16":
            r["dacc_avg"] = r["acc_avg"] - base
    return rows


def gates(R, rows):
    out = []

    def add(name, ok, detail):
        out.append((name, "PASS" if ok else ("SKIP" if ok is None else "FAIL"), detail))
    v = R["validate"]
    if v:
        rm = [x.get("rmse_f32") for x in v.values() if isinstance(x, dict) and "rmse_f32" in x]
        add("1. Sim weights == checkpoint dequantized weights", (max(rm) < 1e-3) if rm else None,
            f"max RMSE {max(rm):.2e} (scale stored in BF16 explains residual)" if rm else "no rmse field")
    else:
        add("1. Sim weights == checkpoint dequantized weights", None, "validate/sim_vs_ckpt.json missing")
    a, b = R["sim"]["none"]["ppl_wikitext"]["stats"]["ppl_base"], R["trackb"]["bf16"]["ppl_wikitext"]["stats"]["ppl"]
    add("2. BF16 PPL Track A vs B within 1%", abs(a - b) / a < 0.01, f"A={a:.3f} B={b:.3f} rel={abs(a - b) / a:.2%}")
    bad = []
    for f in FMTS:
        for s in SETS:
            da = R["sim"][A_NAME.get(f, f)][s]["stats"]["delta_loss"] if A_NAME.get(f, f) in R["sim"] else None
            if da is None:
                continue
            db = R["trackb"][f][s]["stats"]["loss"] - R["trackb"]["bf16"][s]["stats"]["loss"]
            if da * db <= 0 or not (0.33 < db / da < 3.0):
                bad.append(f"{f}/{s} A={da:+.4f} B={db:+.4f}")
    add("3. Delta-loss sign & magnitude agree across tracks", not bad, "; ".join(bad) or "all 10 pairs agree within 3x")
    sizes = {f: R["size"][f]["quant_bytes"] for f in FMTS if f in R["size"]}
    spread = (max(sizes.values()) - min(sizes.values())) / min(sizes.values())
    add("4. All FP8 checkpoints same size (scales only differ)", spread < 0.002, f"spread {spread:.3%}")
    add("5. BF16 `none` KL == 0", all(float(R["sim"]["none"][s]["stats"]["kl_mean"]) == 0.0 for s in SETS), "kl_mean exactly 0")
    add("6. vLLM kernel_log shows intended kernel", None,
        "kernel_log not captured (EngineCore subprocess logs bypass the python logger); W8A16 Marlin pin verified in code, W8A8 path inferred from speedup")
    if "bf16" in R["lm"]:
        msgs, ok = [], True
        for t, (lo, hi) in LM_RANGE.items():
            v, _ = lm_metric(R, "bf16", t)
            good = v is not None and lo <= v <= hi
            ok &= good
            msgs.append(f"{t}={v:.3f}{'' if good else ' OUT'}")
        add("7. BF16 lm-eval scores plausible for Qwen2.5-3B", ok, ", ".join(msgs))
    return out


def pairwise(R):
    pairs = [("w8a16_pt", "w8a16_pc"), ("w8a8_pt_dyn", "w8a8_pc"), ("w8a8_pt_dyn", "w8a8_pt_static"),
             ("w8a8_pc", "w8a8_pt_static"), ("w8a16_pc", "w8a8_pc"), ("w8a16_pt", "w8a8_pt_dyn")]
    lines = []
    for x, y in pairs:
        for s in SETS:
            qx = np.asarray(R["trackb"][x][s]["stats"]["window_losses"])
            qy = np.asarray(R["trackb"][y][s]["stats"]["window_losses"])
            est, lo, hi, p = paired_bootstrap(qx, qy)
            kx = kl_windows(R, A_NAME.get(x, x), s).mean(axis=1)
            ky = kl_windows(R, A_NAME.get(y, y), s).mean(axis=1)
            ke, klo, khi, kp = paired_bootstrap(kx, ky)
            lines.append(f"| {STYLE[x]['label']} - {STYLE[y]['label']} | {SET_LABEL[s]} | "
                         f"{est:+.4f} [{lo:+.4f}, {hi:+.4f}] {sig(p)} | {ke:+.5f} [{klo:+.5f}, {khi:+.5f}] {sig(kp)} |")
    return lines


def fmt(v, spec):
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else format(v, spec)


def main():
    R = load_results()
    rows = build_rows(R)
    OUT.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    g = gates(R, rows)
    md = ["# FP8 PTQ summary - Qwen2.5-3B on Modal L4 (auto-generated by make_summary.py)", "",
          "Delta values are vs the BF16 baseline. CIs are 95% bootstrap over the 64 evaluation windows (paired). "
          "Significance: `*` p<0.05, `**` p<0.01, `***` p<0.001, `ns` not significant.", ""]
    md += ["## 1. Size", "", "| Format | Size MB | Weights GiB | Compression % |", "|---|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['label']} | {fmt(r.get('size_MB'), '.1f')} | {fmt(r.get('weights_GiB'), '.2f')} | {fmt(r.get('compression_pct'), '.1f')} |")
    md += ["", "## 2. Quality (Track B real vLLM: loss / PPL; Track A simulation: full-vocab KL)", "",
           "| Format | PPL wiki (B) | dLoss wiki [95% CI] | dPPL wiki | PPL chat (B) | dLoss chat [95% CI] | mean KL wiki (A) [CI] | p99 KL wiki | top-1 wiki | mean KL chat (A) | top-1 chat |",
           "|---|---:|---|---:|---:|---|---|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['label']} | {fmt(r.get('ppl_B_wiki'), '.3f')} | {fmt(r.get('dloss_B_wiki'), '+.4f')} {r.get('dloss_B_wiki_ci', '')} | "
                  f"{fmt(r.get('dppl_B_wiki'), '+.3f')} | {fmt(r.get('ppl_B_chat'), '.3f')} | {fmt(r.get('dloss_B_chat'), '+.4f')} {r.get('dloss_B_chat_ci', '')} | "
                  f"{fmt(r.get('kl_A_wiki'), '.5f')} {r.get('kl_A_wiki_ci', '')} | {fmt(r.get('kl_p99_A_wiki'), '.4f')} | {fmt(r.get('top1_A_wiki'), '.4f')} | "
                  f"{fmt(r.get('kl_A_chat'), '.5f')} | {fmt(r.get('top1_A_chat'), '.4f')} |")
    fp16 = R["sim"].get("fp16cast", {}).get("ppl_wikitext")
    if fp16:
        md.append(f"\nNoise floor (BF16 -> FP16 cast): mean KL {fp16['stats']['kl_mean']:.5f}, top-1 {fp16['stats']['top1_agree']:.4f}.")
    md += ["", "## 3. Downstream accuracy (lm-eval, 0-shot; headline metric per task)", "",
           "| Format | " + " | ".join(f"{t} ({PRIMARY[t]})" for t in TASKS) + " | avg | d avg |", "|---|" + "---:|" * (len(TASKS) + 2)]
    for r in rows:
        cells = []
        for t in TASKS:
            v = r.get(f"acc_{t}")
            d, p = r.get(f"dacc_{t}"), r.get(f"dacc_{t}_p")
            cells.append("-" if v is None else f"{v:.4f}" + ("" if d is None else f" ({d:+.4f} {sig(p)})"))
        md.append(f"| {r['label']} | " + " | ".join(cells) + f" | {fmt(r.get('acc_avg'), '.4f')} | {fmt(r.get('dacc_avg'), '+.4f')} |")
    md += ["", "## 4. Speed (vLLM, L4, 128 output tokens, median of repeats)", "",
           "| Format | tok/s B=1 | tok/s B=32 | tok/s B=128 | TPOT ms B=1 | prefill s (B=128, in=512) | tok/s (B=128, in=512) |", "|---|---:|---:|---:|---:|---:|---:|"]
    base = next(r for r in rows if r["format"] == "bf16")
    for r in rows:
        sp = ""
        if r["format"] != "bf16" and r.get("tok_s_b128") and base.get("tok_s_b128"):
            sp = f" ({r['tok_s_b128'] / base['tok_s_b128']:.2f}x)"
        md.append(f"| {r['label']} | {fmt(r.get('tok_s_b1'), '.1f')} | {fmt(r.get('tok_s_b32'), '.1f')} | {fmt(r.get('tok_s_b128'), '.1f')}{sp} | "
                  f"{fmt(r.get('tpot_ms_b1'), '.2f')} | {fmt(r.get('prefill_s_b128_in512'), '.2f')} | {fmt(r.get('tok_s_b128_in512'), '.1f')} |")
    md += ["", "## 5. Generation agreement vs BF16 (25 prompts, 128 greedy tokens)", "", "| Format | exact matches | mean matching prefix |", "|---|---:|---:|"]
    for r in rows:
        if r["format"] != "bf16":
            md.append(f"| {r['label']} | {r.get('gen_exact', '-')} | {fmt(r.get('gen_prefix'), '.3f')} |")
    md += ["", "## 6. Pairwise recipe comparisons (first minus second; negative = first is better)", "",
           "| Pair | Set | d window loss (Track B) [95% CI] | d mean KL (Track A) [95% CI] |", "|---|---|---|---|"] + pairwise(R)
    md += ["", "## 7. Scientific consistency gates", "", "| Gate | Status | Detail |", "|---|---|---|"]
    md += [f"| {n} | **{s}** | {d} |" for n, s, d in g]
    (OUT / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nwrote {OUT / 'summary.md'} and {OUT / 'summary.csv'}")


if __name__ == "__main__":
    main()
