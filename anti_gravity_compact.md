# Antigravity Compacted Session Summary

Generated: 2026-11-20 00:28 IST

## 1. Task Overview
- **Goal:** Complete an FP8 Post-Training Quantization (PTQ) research study on `Qwen/Qwen2.5-3B` (BF16 baseline) on Modal L4 GPUs (`e:\quantt`), comparing FP8 PTQ recipes across quality (loss, perplexity, full-vocab KL, lm-eval downstream accuracy, 25-prompt generation agreement), size/compression, speed/memory, and per-layer diagnostics, culminating in visual comparison graphs (`results/plots/`), summary tables (`results/summary/`), and `docs/findings.md`.
- **Hard Constraints:**
  - **PTQ ONLY:** No QAT, no fine-tuning, no `.backward()`, no LoRA, no distillation anywhere (`config.assert_inference_only()` called at start of every Modal function).
  - **Baseline dtype:** Strictly `bfloat16` (never `float16`).
  - **NEVER run `modal` commands from the agent:** Always give the exact `modal run ...` command(s) for the user to run in PowerShell (`cd E:\quantt\fp_8; $env:PYTHONUTF8 = "1"`) and paste the output logs. Local Python/file tasks CAN be run directly using `E:\quantt\venv\Scripts\python.exe`.
  - **Update `E:\quantt\project_state.json`** after each stage/execution, following its existing JSON format.
  - FP8 first; INT8 comes later using the same harness.

## 2. Progress

### Pre-existing Work
- **Phase 0 & Phase 1 (Track B checkpoints):** BF16 baseline downloaded to Modal Volume `quant-models` (`/vol/models/qwen2.5-3b-bf16`). Frozen test data built at `/vol/data/` (`ppl_wikitext` 64×2048 WikiText-2 test, `kl_chat` 64×2048 UltraChat test_sft, `calib_chat` 512×2048 UltraChat train_sft seed 42, `prompts.json` 25 prompts; manifest at `E:\quantt\results\data_manifest.json`). Four initial Track B checkpoints quantized (`w8a16_pt`, `w8a16_pc`, `w8a8_pc`, `w8a8_pt_static`) and 5-prompt vLLM sanity checks completed (`results/sanity_*.json`, `results/size_*.json`).
- **Phase 2 (Track A simulated eval):** `sim_eval.py` already run on 64 windows for `none`, `fp16cast`, `w8a16_pt`, `w8a16_pc`, `w8a8_pt`, `w8a8_pc` across `ppl_wikitext` and `kl_chat` (`results/sim/*.json` and `*.npy`).

### Completed This Session
- **Approved Implementation Plan:** `evaluation_benchmark_plan.md` (in Antigravity artifacts directory).
- **Step 0 (Complete, locally verified):**
  - Archived stale misnamed `results/size_w8a8_pt.json` → `results/_archive_misnamed/size_w8a8_pt.json`.
  - Updated `fp_8/config.py`: added `w8a8_pt_dyn` to `CHECKPOINT_RECIPES`, `w8a8_pt_static` to `RECIPES`, `ALL_FORMATS = ["bf16", *CHECKPOINT_RECIPES]`, removed `"wikitext"` from `EVAL_TASKS`, added `LM_EVAL_FEWSHOT = 0`, `LM_EVAL_BATCH = "auto"`, `EVAL_MMLU = False`.
  - Updated `fp_8/modal_app.py`: added `"vllm_common"` to `vllm_image`'s `.add_local_python_source(...)`.
  - Updated `fp_8/quantize_fp8.py`: added `w8a8_pt_dyn` branch (dynamic per-tensor FP8 activations, no calibration).
  - Created `fp_8/vllm_common.py`: shared `make_llm()` (pins `w8a16_fp8` to `"marlin"`, captures kernel/memory log lines, reports GPU compute capability & weight GiB) and `parse_fmts()`.
  - Created `fp_8/validate_sim.py`: Step 1 validation gate comparing Track A `qdq(W)` vs Track B checkpoint `w_fp8 * weight_scale` across all 252 Linear layers.
  - Created `fp_8/eval_ppl_vllm.py`: Step 2 (E1) Track B perplexity/loss on frozen windows via vLLM `prompt_logprobs=0`, with `--smoke` (4 WikiText windows, not saved) and parallel `.starmap` for `--fmts all`.
  - Updated `fp_8/sim_eval.py`: added `per_tensor_static` mode (loads `input_scale` tensors from `/vol/models/qwen2.5-3b-w8a8-pt-static` safetensors) and saves `window_loss_base` / `window_loss_quant` per window for bootstrap CIs.
- **Step 0b (Complete, run on Modal):**
  - `modal run quantize_fp8.py --recipe w8a8_pt_dyn` succeeded: `/vol/models/qwen2.5-3b-w8a8-pt-dyn`, 3397.42 MB (45.0% compression), 252 `weight_scale` tensors, 0 `input_scale` tensors, dynamic per-tensor FP8 activations; saved `results/size_w8a8_pt_dyn.json`.
- **Step 1 Validation Gate (Complete, run on Modal):**
  - `modal run validate_sim.py` succeeded and saved `results/validate/sim_vs_ckpt.json`.
  - `w8a16_pc`: `exact_fp8=0.9699`, `exact_bf16=0.7414`, `rmse_f32=4.00e-4`, `max_scale_rel_diff=3.32e-3`.
  - `w8a16_pt`: `exact_fp8=0.9691`, `exact_bf16=0.7287`, `rmse_f32=4.03e-4`, `max_scale_rel_diff=3.27e-3`.
  - Worst layer for both: `model.layers.21.mlp.down_proj` (`max_abs_diff=0.0691`).
- **Step 2 (E1 — Track B PPL, Complete on Modal):**
  - `modal run eval_ppl_vllm.py --fmts all` completed across all 6 formats on 64 WikiText + 64 chat windows (saved to `results/trackb_ppl/`).
  - **WikiText PPL:** `bf16` 7.782 → `w8a16_pc` 7.804 (+0.022) → `w8a16_pt` 7.809 (+0.027) → `w8a8_pc` 7.834 (+0.052) → `w8a8_pt_dyn` 7.855 (+0.073) → `w8a8_pt_static` 7.866 (+0.084).
  - **Chat PPL:** `bf16` 4.144 → `w8a16_pc` 4.157 (+0.013) → `w8a16_pt` 4.165 (+0.021) → `w8a8_pc` 4.180 (+0.036) → `w8a8_pt_dyn` 4.183 (+0.039) → `w8a8_pt_static` 4.193 (+0.049).
  - Exact monotonic precision ordering on both datasets and tight agreement with Track A (`bf16` 7.782 vs 7.778; `w8a16_pc` ΔPPL +0.022 in both tracks).
- **State file:** `project_state.json` is updated through Step 2.

## 3. Key Findings
- **Validation Gate (Step 1):** `validate_sim.py` reported `[FAIL]` only because `llm-compressor` stores `weight_scale` in `bfloat16` (1 ULP = $2^{-8} \approx 3.91 \times 10^{-3}$), whereas Track A computes `amax / 448` in `float32`. The observed `max_scale_rel_diff = 3.32e-3` is within 1 BF16 ULP, `exact_fp8 = 97.0%`, and `rmse_f32 = 4e-4`, confirming both tracks implement the same FP8 E4M3 quantization. Document in `docs/findings.md`.
- **Track A WikiText Results (baseline PPL 7.778):**
  - `fp16cast` (noise floor): ΔPPL +0.00002, mean KL 0.00067, top-1 98.0%
  - `w8a16_pt`: ΔPPL +0.035, mean KL 0.00564, top-1 96.3%
  - `w8a16_pc`: ΔPPL +0.022, mean KL 0.00581, top-1 96.2%
  - `w8a8_pc`: ΔPPL +0.046, mean KL 0.01050, top-1 94.7%
  - `w8a8_pt`: ΔPPL +0.065, mean KL 0.01144, top-1 94.6% (max KL on chat: 2.53 vs 1.05 for `w8a8_pc`)
  - Key insight: Quantizing activations (A8 vs A16) roughly doubles KL; per-tensor vs per-channel weights barely differs in W8A16; all FP8 recipes are 8–17× above the FP16 noise floor.
- **vLLM W8A16 Marlin Pin:** vLLM 0.34.0's default Humming backend fails NVRTC JIT compilation on L4 for W8A16 FP8; pinning `KernelConfig(linear_backend_per_quant={"w8a16_fp8": "marlin"})` in `vllm_common.py` resolves it.

## 4. Active Context
- **Workspace:** `E:\quantt` (run Modal commands from `E:\quantt\fp_8`; local Python via `E:\quantt\venv\Scripts\python.exe`).
- **All 6 Track B formats ready on Modal Volume:** `bf16`, `w8a16_pt`, `w8a16_pc`, `w8a8_pc`, `w8a8_pt_dyn`, `w8a8_pt_static`.
- **Pending user output:** Step 3 (`modal run sim_eval.py --recipes w8a8_pt_static`).
- **Files NOT yet created:** `fp_8/layer_diag.py`, `fp_8/bench_speed.py`, `fp_8/eval_lm.py`, `fp_8/stats.py`, `fp_8/make_plots.py`, `fp_8/make_summary.py`, `docs/findings.md`. `fp_8/sanity_check.py` not yet updated to 25 prompts.

## 5. Next Steps
1. **Finish Step 2 (E1 — Track B PPL):**
   - Review `eval_ppl_vllm.py --fmts bf16 --smoke` output (fix any vLLM 0.34 API mismatch; verify `bf16` PPL ≈ 7.78 on WikiText).
   - Run full sweep on Modal: `modal run eval_ppl_vllm.py --fmts all`.
2. **Step 3 (E2 — Static recipe in Track A sim):**
   - Run on Modal: `modal run sim_eval.py --recipes w8a8_pt_static` and cross-check PPL vs Track B `w8a8_pt_static`.
3. **Step 4 (E3 — Per-layer diagnostics):**
   - Create `fp_8/layer_diag.py` (`quant_image`, L4) with `--part weights|acts|layererr|sensitivity` on fixed 4 WikiText + 4 chat windows → `results/diag/{weights.json, weight_samples.npz, acts.json, layer_err.json, sensitivity.json}`.
   - Run all 4 parts on Modal.
4. **Step 5 (E5 — 25-prompt generation agreement):**
   - Archive old 5-prompt `results/sanity_*.json` to `results/_archive_sanity5/`.
   - Update `fp_8/sanity_check.py` to use `vllm_common.make_llm()`, all 25 prompts from `/vol/data/prompts.json`, `MAX_TOKENS = 128`, `--fmts all`, exact match / first divergence / matching prefix fraction.
   - Run on Modal.
5. **Step 6 (B1 — Speed & memory benchmark):**
   - Create `fp_8/bench_speed.py` (`vllm_image`) over `batch ∈ {1,8,32,128} × input ∈ {128,512}`, output 128, warmup 2, repeats 5, fixed-seed random token IDs, `ignore_eos=True` → `results/bench/<fmt>.json`.
   - Run `--fmts bf16 --smoke`, then `--fmts all` (sequential per format for fair timing) on Modal.
6. **Step 7 (E4 — Downstream accuracy via lm-eval):**
   - Create `fp_8/eval_lm.py` (`vllm_image`) wrapping `lm_eval.models.vllm_causallms.VLLM` with Marlin `KernelConfig`, 0-shot on `hellaswag`, `arc_easy`, `arc_challenge`, `winogrande`, `piqa` → `results/lm_eval/<fmt>.json` + `<fmt>__samples.npz` (for McNemar tests); support `--smoke` (`limit=50`, 2 tasks) and optional `--tasks mmlu --fewshot 5`.
   - Run `--fmts bf16 --smoke`, then `--fmts all` on Modal.
7. **Step 8 (R — Local stats, plots, summary, findings):**
   - Create & run locally: `fp_8/stats.py` (bootstrap CIs, McNemar + `--selftest`), `fp_8/make_plots.py` (13 plots PNG + SVG in `results/plots/`), `fp_8/make_summary.py` (`results/summary/summary.{md,csv}` + scientific consistency gates).
   - Write `docs/findings.md`, update Section 1 of `plan.md`, and finalize `project_state.json`.
