# FP8 Post-Training Quantization Study: Qwen2.5-3B on Modal L4

> **Hard constraint: post-training quantization only.** No QAT, no fine-tuning, no gradient updates, no LoRA, no distillation, anywhere in this study. Weights are changed only by a one-shot rounding to FP8. Calibration (only for the optional static-activation recipe) just *records activation ranges* on a few samples with no training. Any code path that calls `.backward()` or an optimizer is a bug. Phase 0 adds a guard for this (see below).

Goal: take one BF16 baseline (Qwen2.5-3B) and quantize it with several FP8 PTQ recipes. Compare each against the baseline on accuracy, perplexity/loss, KL divergence, size and speed. Then analyse the graph patterns to explain *why* each recipe behaves as it does. INT8 comes later, using the same harness.

---

## 1. Study Status (updated 2026-11-20)

> *Note: The original 2026-11-19 audit targeted an older `Qwen2.5-0.5B-Instruct` / FP16 prototype. The repository was migrated to **`Qwen/Qwen2.5-3B` (BF16 baseline)** and the full FP8 PTQ evaluation and benchmark suite is now complete.*

### Completed Artifacts
| Item | State |
|---|---|
| `fp_8/config.py`, `fp_8/modal_app.py`, `fp_8/vllm_common.py` | Central 3B / BF16 config, Modal L4 images (`llmcompressor 0.15.0`, `vLLM 0.34.0`, `lm-eval 0.4.13`), shared Marlin-pinned vLLM loader. |
| `fp_8/download_model.py`, `fp_8/prepare_data.py` | BF16 baseline (`/vol/models/qwen2.5-3b-bf16`) and frozen datasets (`/vol/data/`: `ppl_wikitext`, `kl_chat`, `calib_chat`, 25 `prompts.json`) built. |
| `fp_8/quantize_fp8.py`, `fp_8/validate_sim.py` | All 5 FP8 PTQ Track B checkpoints (`w8a16_pt`, `w8a16_pc`, `w8a8_pc`, `w8a8_pt_dyn`, `w8a8_pt_static`) quantized (~45% compression, 252 layers) and validated against Track A (`RMSE 4e-4`). |
| `fp_8/sim_eval.py`, `fp_8/eval_ppl_vllm.py` | Track A full-vocab KL + PPL and Track B vLLM PPL completed across 64 WikiText + 64 chat windows (BF16 tracks agree within 0.06%). |
| `fp_8/layer_diag.py` | All 4 per-layer diagnostics (`weights`, `acts`, `layererr`, `sensitivity`) completed → `results/diag/`. |
| `fp_8/sanity_check.py`, `fp_8/bench_speed.py`, `fp_8/eval_lm.py` | 25-prompt generation agreement (`results/sanity25/`), 8-cell speed benchmark (`results/bench/`), and 5-task 0-shot lm-eval (`results/lm_eval/`) completed for all 6 formats. |
| `fp_8/stats.py`, `fp_8/make_summary.py`, `fp_8/make_plots.py`, `docs/findings.md` | Bootstrap CIs, McNemar tests, 13 PNG+SVG plots (`results/plots/`), summary tables (`results/summary/`), and final research findings (`docs/findings.md`). |
| Next phase | `int_8/` — INT8 PTQ comparison using the same harness. |

---

## 2. Experiment design

### 2.1 Baseline and model choice
**Recommendation: `Qwen/Qwen2.5-3B` (the base model, BF16) as the primary model.** This is my judgement from the constraints, not a benchmark I ran.

| Candidate | Fits L4 (24 GB)? | Verdict |
|---|---|---|
| **Qwen2.5-3B (base)** | 6.2 GB BF16, plenty of room for logits, hooks and vLLM | **Best fit.** Stored natively in BF16, small enough to run dozens of recipe and layer sweeps cheaply, and big enough that FP8 effects are not drowned in noise. Qwen2.5 has strong activation outliers, which makes it a good stress test for W8A8 per-tensor vs per-token. Base is the right choice for perplexity and KL (clean next-token distribution) and for lm-eval log-likelihood tasks. |
| Qwen2.5-3B-Instruct | Same | Good for chat sanity prompts, but instruction tuning makes WikiText perplexity less meaningful. Use it only as a secondary check. |
| Qwen2.5-1.5B | Yes | Cheaper, but small models quantize worse and hide the effect size. Use only for debugging. |
| Qwen2.5-7B | 15 GB BF16 fits for vLLM, but full-vocab KL in simulation gets tight | Larger models are more FP8-robust. Good **optional scaling check** after the 3B study, not the first target. |
| Llama-3.2-3B | Yes | Valid alternative, but gated on Hugging Face (needs a token) and brings no advantage here. |

So: primary = Qwen2.5-3B base; sanity chat = Qwen2.5-3B-Instruct with the identical recipes (one extra run of the same code); optional later = Qwen2.5-7B. Both 3B variants share a tokenizer, so the same prepared data works for both.

The baseline runs in **BF16** for weights and activations. Every run uses the same GPU (L4), seed (42), max length (2048) and data.

### 2.2 Recipe matrix
FP8 here means E4M3 (max 448). Linear layers are quantized. `lm_head`, embeddings and norms stay BF16.

| ID | Weights | Activations | Weight scale | Act scale | Needs calibration? |
|---|---|---|---|---|---|
| `bf16` | BF16 | BF16 | n/a | n/a | n/a |
| `w8a16_pt` | FP8 | BF16 | per-tensor | n/a | No |
| `w8a16_pc` | FP8 | BF16 | per-channel | n/a | No |
| `w8a8_pt` *(Track A simulation)* | FP8 | FP8 | per-tensor | per-tensor dynamic | No |
| `w8a8_pc` | FP8 | FP8 | per-channel | per-token dynamic | No |
| `w8a8_pt_static` *(Track B checkpoint; extra)* | FP8 | FP8 | per-tensor | per-tensor **static** | Yes (calib_chat) |

Notes:
- The four core rows form the 2x2 you asked for (A16 vs A8, per-tensor vs per-channel). Each pair isolates one variable.
- "Per-channel" for weights means one scale per output channel. For activations, the natural counterpart is per-token, since you cannot do per-channel on activations at runtime.
- The static-activation row is optional. It is the realistic fast-inference case and it shows the cost of calibration error. Add it after the core four work.
- Optional later: a mixed recipe (keep the most sensitive layers in BF16) driven by the per-layer analysis.

### 2.3 Two tracks
This is the key design decision. vLLM returns only top-k logprobs, so it cannot give a **full-vocabulary KL**. Real FP8 kernels also hide per-layer internals. So run both tracks and cross-check them:

| | Track A: simulated (fake-quant, PyTorch/HF) | Track B: real checkpoints (llm-compressor + vLLM) |
|---|---|---|
| Purpose | KL, loss, per-layer error, all the graphs | Real size on disk, GPU memory, throughput, lm-eval accuracy |
| How | Quantize to FP8 and dequantize back inside the forward pass using `torch.float8_e4m3fn`. Scale is `amax/448`. | `QuantizationModifier` with custom schemes, saved with `save_compressed=True`, served in vLLM. |
| Strength | Full control over granularity, full-vocab logits, forward hooks | What you would actually deploy |
| Weakness | Does not measure speed or real memory | Opaque internals |

**Sanity check linking them**: Track A perplexity should match Track B perplexity within noise for the same recipe. If it does not, a recipe is misconfigured.

---

## 2b. Test data (generated once, frozen, reused by every recipe)

`fp_8/prepare_data.py` builds everything below, saves it to the Volume under `/vol/data/`, and writes a `manifest.json` (dataset name, split, sample counts, token counts, SHA-256 of each array). Every recipe reads these exact files. That is what makes the comparisons fair and repeatable.

| Set | Source | Size | Used for | Notes |
|---|---|---|---|---|
| `ppl_wikitext` | WikiText-2 raw, **test** split | 64 windows x 2048 tokens (~131k tokens) | Perplexity, loss, **KL**, per-token KL | Standard PPL benchmark, so numbers are comparable with published ones. |
| `kl_chat` | UltraChat-200k, **test_sft** split | 64 windows x 2048 tokens | KL / loss on a second domain | Checks the damage is not WikiText-specific. |
| `calib_chat` | UltraChat-200k, **train_sft** split | 512 samples x up to 2048 tokens | Activation range calibration for the static recipe **only** | Different split and domain from the eval sets, so there is no leakage. Not used for training of any kind. |
| `prompts_fixed` | The 5 sanity prompts plus ~20 more (factual, code, math, translation, reasoning) written in the script | 25 prompts | Greedy-output diff between BF16 and each recipe | Reports the fraction of identical outputs and the first token where they diverge. |
| lm-eval tasks | hellaswag, arc_easy, arc_challenge, winogrande, piqa (downloaded by lm-eval itself) | Full sets, fixed few-shot settings | Downstream accuracy | MMLU later. |

Rules: eval sets are never used for calibration; the same windows are used for every recipe; token counts are asserted equal across runs.

---

## 3. Metrics

### Accuracy and loss
- **Perplexity and cross-entropy loss** on WikiText-2 test, 2048-token windows. Loss is the main continuous quality signal. Report the delta vs BF16 (ΔPPL, Δloss).
- **lm-eval tasks**: hellaswag, arc_easy, arc_challenge, winogrande, piqa (MMLU later). Report absolute accuracy and the drop vs BF16, with stderr, so you can tell noise from real loss.

### KL divergence (Track A, full vocabulary)
On the same ~64 windows (~130k tokens) of WikiText-2 test, plus a second domain such as code or chat text for robustness:
- **Mean KL(P_bf16 || P_quant)** per token, in nats.
- **Top-1 agreement** (argmax flip rate) and **top-5 overlap**.
- **KL distribution**: median, p90, p99, max. Quantization damage is usually heavy-tailed, so the mean alone hides it.
- **KL vs token position**, to see whether error accumulates.
- **Noise floor**: KL(BF16 || FP16-cast-of-same-weights) and BF16 vs itself (should be exactly 0). This tells you what counts as "harmlessly small".

### Size and compression
- Disk size of the safetensors files, and bytes by dtype (the existing inspect code in `quantize_fp8.py` does this).
- **Compression %** = `1 - size_quant / size_bf16`, plus the ratio.
- Peak GPU memory after loading in vLLM.
- Expect roughly 45-48% reduction for the whole model, less than 50% because embeddings and `lm_head` stay BF16. For the 3B model with tied embeddings this overhead is small but visible, so report it.
- W8A16 and W8A8 should have **identical size** on disk, because only the activations differ. Verify that explicitly; if they differ, something is wrong.

### Speed (Track B, secondary)
Throughput and latency at batch 1/8/32/128. The compute story is different per recipe: W8A8 uses FP8 tensor cores (L4 is Ada, SM89, so it is supported); W8A16 saves memory bandwidth but dequantizes in the kernel.

---

## 4. Graphs to produce, and what each should reveal

| # | Graph | What to look for |
|---|---|---|
| 1 | **Weight histograms** for a few layers (q/k/v/o/gate/up/down) overlaid with the FP8 representable grid | Whether weights are well covered by E4M3's dynamic range |
| 2 | **Per-channel weight amax** (sorted, per layer) | Spread across channels. A large spread explains why per-channel beats per-tensor. |
| 3 | **Per-layer weight SQNR / MSE**, per-tensor vs per-channel, vs layer index | Which layers and projection types are hurt most |
| 4 | **Activation amax per layer** (per-token and per-tensor) | Outlier tokens and "massive activations", usually in `down_proj` inputs and early/late layers. This is the main reason W8A8 per-tensor can fall apart. |
| 5 | **Per-layer output error** (cosine/MSE of the quantized layer's output vs BF16) vs depth | Error accumulation and the most sensitive layers |
| 6 | **Loss vs recipe** (bar, with ΔPPL) | Overall ranking |
| 7 | **KL distribution** (histogram on log scale plus CDF) per recipe | Heavy tails |
| 8 | **KL vs token position** | Accumulating error |
| 9 | **Accuracy drop per task**, with error bars | Which tasks are sensitive |
| 10 | **Pareto: size (or speed) vs quality** | The trade-off and the best recipe |
| 11 | **Layer-ablation (sensitivity)**: quantize one layer at a time, measure ΔKL | Which layers to keep in BF16 for a mixed recipe |

Hypotheses to confirm or refute with the data:
- W8A16 will be nearly lossless, and per-channel will only marginally beat per-tensor there. FP8 has a wide dynamic range, so per-tensor weight scaling is already decent.
- W8A8 per-tensor will be clearly worse than W8A8 per-channel/per-token because of activation outliers.
- `down_proj` and specific layers will dominate the error budget.

---

## 5. Implementation steps

Work from `E:\quantt\fp_8` (rename the folder later if INT8 is added alongside).

### Phase 0: Fix the foundation
- [ ] `config.py`: set `MODEL_ID = "Qwen/Qwen2.5-3B"` (base), `MODEL_SHORT = "qwen2.5-3b"`, and rename `fp16` to `bf16`. Add a `RECIPES` table matching section 2.2.
- [ ] `modal_app.py` and `sanity_check.py`: change `dtype="float16"` to `"bfloat16"` everywhere.
- [ ] `quantize_fp8.py`: load with `torch.bfloat16`, and replace the hardcoded `168` with a count derived from the model config.
- [ ] Add `HF_TOKEN`-free download check: Qwen is ungated, so no secret is needed.
- [ ] Add a **no-training guard**: a small helper that runs every model under `torch.inference_mode()` and asserts `torch.is_grad_enabled()` is false and no parameter has `requires_grad=True`. Call it at the start of each Modal function.
- [ ] Run `modal run prepare_data.py` to generate and freeze the test data (section 2b). CPU only, no GPU cost.
- [ ] Rerun `modal run download_model.py` for the 3B model (CPU only, no GPU cost).
- [ ] Rerun the smoke tests (`test_quant`, `test_vllm`). Confirm compute capability is 8.9.
- [ ] Rerun the sanity check on `bf16`. Save `results/sanity_bf16.json`.
- [ ] Keep the old 0.5B results in a `results/_archive_0.5b/` folder so nothing is lost.

### Phase 1: Real checkpoints (Track B quantization)
- [ ] Extend `quantize_fp8.py` to take `--recipe` and build the scheme from a Track B checkpoint recipe table, using explicit `QuantizationScheme` objects (`FP8` weight-only for W8A16, `FP8_DYNAMIC` for `w8a8_pc`, and calibrated static per-tensor activation scales under a distinct `w8a8_pt_static` ID). Track A's `w8a8_pt` remains dynamic per-tensor fake quantization; do not compare it as if it were the same method as the static checkpoint.
- [ ] Produce the checkpoints `qwen2.5-3b-w8a16-pt`, `-w8a16-pc`, `-w8a8-pt`, `-w8a8-pc`.
- [ ] Save a `size_report.json` for each: bytes by dtype, compression %, number of scale tensors, and strategy for weights/activations.
- [ ] Run the sanity check (greedy token diff) for each recipe.
- [ ] Check that vLLM actually loads each one with its intended kernel path (read the vLLM log lines). A silent fallback would invalidate the speed results.

### Phase 2: Simulated harness (Track A), the core of the analysis
- [ ] `fakequant.py`: FP8 E4M3 quantize/dequantize helpers with per-tensor, per-channel and per-token scaling. Unit-test them on random tensors: round-trip error should be bounded, and a zero tensor must not divide by zero.
- [ ] `sim_model.py`: load the BF16 model in HF, then swap `nn.Linear` for a wrapper that applies the chosen recipe. Skip `lm_head`.
- [ ] **Validation gate**: check that the simulated weights match the real checkpoint's dequantized weights for `w8a16_pc` (they should be near-identical). Fix any mismatch before trusting any result.
- [ ] `eval_ppl_kl.py` (Modal, L4): for each recipe, compute loss/PPL and full-vocab KL against the BF16 logits on the shared windows. To avoid keeping 130k x 152k floats, compute BF16 log-probs in chunks and either compute KL on the fly or store only per-token KL.
- [ ] Save per-token KL arrays as `.npy` and aggregates as JSON in `results/`.
- [ ] Determinism: set seeds and use the same windows for every recipe. Run BF16 twice to confirm KL is 0.

### Phase 3: Per-layer diagnostics
- [ ] Forward hooks to capture weight stats (amax per channel, SQNR) and activation stats (amax per token/tensor, kurtosis) on a fixed calibration batch.
- [ ] Per-layer output error vs BF16.
- [ ] One-layer-at-a-time sensitivity sweep (36 layers x a couple of recipes; cheap at 3B).

### Phase 4: Benchmarks on real checkpoints
- [ ] `eval_lm.py`: lm-eval through the vLLM backend, for all recipes plus BF16, same tasks, same few-shot settings, same batch size. Save JSON per recipe.
- [ ] `bench_speed.py`: the `BENCH_*` grid from `config.py`. Record tokens/s, latency and peak memory.
- [ ] Cross-check: Track A perplexity vs Track B perplexity for each recipe.

### Phase 5: Analysis and report
- [ ] `make_plots.py`: reads `results/*.json|npy`, writes the graphs in section 4 to `results/plots/`. Colour and line style per recipe, kept consistent across plots.
- [ ] `make_summary.py`: one table in `results/summary/` with columns recipe, size MB, compression %, PPL, Δloss, mean KL, p99 KL, top-1 agreement, task accuracies, tokens/s.
- [ ] Write `docs/findings.md`: for each hypothesis in section 4, say confirmed or refuted and cite the plot.

### Phase 6 (later): INT8
Still PTQ only. Reuse the same harness: add `int8` recipes (W8A8 with SmoothQuant or GPTQ, INT8 per-channel/per-tensor) to `RECIPES`, and the `int_8/` folder can hold checkpoint scripts. The metrics and plots need no changes.

---

## 6. Risks and watch-outs
- **Silent dtype drift**: FP16 vs BF16 mismatches change results. Print the dtype inside every script.
- **Library pin drift**: `llmcompressor==0.15.0` and `vllm==0.34.0` pins are from the existing repo. I could not verify that those versions run in your Modal account beyond what the smoke tests prove, so rerun them before relying on them.
- **vLLM may not support every combination** on L4. If a recipe fails to load (for example W8A8 per-tensor dynamic activations), the failure itself is a finding. Record it, and still evaluate it in Track A.
- **Calibration leakage**: if the static recipe is calibrated on WikiText, do not evaluate KL or PPL on the same split. Calibrate on a different dataset (for example ultrachat or C4).
- **Cost control**: Track A runs a 3B model on 130k tokens per recipe, a few minutes each on an L4. The lm-eval runs are the expensive part, so start with 2 tasks and expand.
- **Small eval noise**: report stderr. Differences below it are not findings.
- **Modal Volume commits**: call `volume.commit()` after writing, and `volume.reload()` before reading in a different function.

---

## 7. Suggested order of work
1. Phase 0 (about 1 hour): fix the config and dtype, download the 3B model, rerun the smoke tests.
2. Phase 2 core (`fakequant.py`, KL/PPL for the 4 recipes): this gives you the first results and graphs fastest, without waiting on checkpoint quantization.
3. Phase 1 in parallel: real checkpoints and sizes.
4. Phases 3 and 4, then 5.
5. INT8 afterwards.

## 8. Assumptions I made (tell me if any are wrong)
- "lasoo" in your message means **loss**, so I included cross-entropy loss and perplexity as primary metrics.
- Primary model is **Qwen2.5-3B base**. Instruct is run as a secondary sanity check with the same code.
- "Generate data for testing" means build a frozen evaluation set (section 2b). If you meant synthetic data generated by an LLM, tell me.
- E4M3 is the FP8 format for both weights and activations (the standard choice for inference).
