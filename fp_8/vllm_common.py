"""Shared vLLM engine builder and kernel/device diagnostics for Track B scripts."""
import logging
import re
import time

import config


class _LogCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.lines.append(self.format(record))
        except Exception:
            pass


_KERNEL_KEYWORDS = (
    "kernel", "Kernel", "Marlin", "marlin", "Cutlass", "cutlass",
    "FP8", "fp8", "scaled_mm", "Using", "Model loading took", "quantization",
    "Available KV cache", "GPU KV cache", "Maximum concurrency",
)


def make_llm(fmt: str, max_model_len: int = config.MAX_MODEL_LEN, **kwargs):
    """Load a checkpoint in vLLM with consistent dtype, seed, and W8A16 Marlin pin.

    Returns (llm, info_dict) where info_dict records load time, GPU capability,
    quantization config, captured kernel/memory log lines, and model weight GiB.
    """
    import torch
    from vllm import LLM
    from vllm.config import KernelConfig

    assert fmt in config.FORMATS, f"unknown format {fmt}; choose from {list(config.FORMATS)}"
    path = config.FORMATS[fmt]
    config.assert_inference_only()

    vllm_logger = logging.getLogger("vllm")
    cap = _LogCapture()
    vllm_logger.addHandler(cap)

    t0 = time.time()
    try:
        # vLLM 0.34 auto-selects Humming first for compressed-tensors FP8 W8A16,
        # which failed NVRTC JIT compilation on this L4. Pin only w8a16_fp8 to Marlin.
        kernel_config = kwargs.pop(
            "kernel_config",
            KernelConfig(linear_backend_per_quant={"w8a16_fp8": "marlin"}),
        )
        llm = LLM(
            model=path,
            dtype=config.DTYPE,
            max_model_len=max_model_len,
            gpu_memory_utilization=config.GPU_MEM_UTIL,
            seed=config.SEED,
            kernel_config=kernel_config,
            **kwargs,
        )
    finally:
        vllm_logger.removeHandler(cap)
    load_s = time.time() - t0

    quant = None
    try:
        quant = llm.llm_engine.model_config.quantization
    except Exception as e:
        quant = f"unavailable ({e})"

    cap_major, cap_minor = torch.cuda.get_device_capability(0)
    free_b, total_b = torch.cuda.mem_get_info(0)

    kernel_log = [line for line in cap.lines if any(k in line for k in _KERNEL_KEYWORDS)]
    weight_gib = None
    for line in kernel_log:
        m = re.search(r"Model loading took\s+([0-9.]+)\s+GiB", line)
        if m:
            weight_gib = float(m.group(1))
            break

    info = {
        "fmt": fmt,
        "path": path,
        "dtype": config.DTYPE,
        "quantization": quant,
        "load_seconds": round(load_s, 3),
        "gpu_name": torch.cuda.get_device_name(0),
        "compute_capability": f"{cap_major}.{cap_minor}",
        "cuda_mem_used_gib": round((total_b - free_b) / (1024 ** 3), 3),
        "cuda_mem_total_gib": round(total_b / (1024 ** 3), 3),
        "model_weight_gib": weight_gib,
        "kernel_log": kernel_log,
    }
    return llm, info


def parse_fmts(fmts_arg: str) -> list[str]:
    """Parse --fmts CLI argument ('all' or comma-separated format names)."""
    if fmts_arg.strip().lower() == "all":
        return list(config.ALL_FORMATS)
    names = [f.strip() for f in fmts_arg.split(",") if f.strip()]
    for n in names:
        assert n in config.FORMATS, f"unknown format '{n}'; choose from {list(config.FORMATS)}"
    return names
