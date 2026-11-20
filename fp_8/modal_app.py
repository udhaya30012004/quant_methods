"""Modal app: images, volume and a GPU smoke test.

Two separate images are used on purpose:
  * quant_image : llm-compressor (does the quantization)
  * vllm_image  : vLLM + lm-eval (benchmarks speed / accuracy / loads checkpoints)
llm-compressor and vLLM each pin their own torch/transformers versions, so keeping
them apart avoids dependency conflicts.

Smoke tests (run ONE at a time, from E:\\quantt\\fp_8):
    modal run modal_app.py::test_quant
    modal run modal_app.py::test_vllm
"""
import modal

import config

# ---------------------------------------------------------------- pinned versions
LLMCOMPRESSOR_VERSION = "0.15.0"
VLLM_VERSION = "0.34.0"
LM_EVAL_VERSION = "0.4.13"

app = modal.App(config.APP_NAME)

# Persistent storage for HF cache, checkpoints and result JSONs.
volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)
VOLUMES = {config.VOLUME_MOUNT: volume}

ENV = {
    "HF_HOME": config.REMOTE_HF_CACHE,          # model downloads cached on the Volume
    "HF_HUB_ENABLE_HF_TRANSFER": "0",
    "TOKENIZERS_PARALLELISM": "false",
}

quant_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(f"llmcompressor=={LLMCOMPRESSOR_VERSION}")
    .env(ENV)
    .add_local_python_source("config", "modal_app")
)

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(f"vllm=={VLLM_VERSION}", f"lm-eval=={LM_EVAL_VERSION}")
    .env(ENV)
    .add_local_python_source("config", "modal_app", "vllm_common")
)


def _gpu_info() -> None:
    import subprocess

    import torch

    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
    print("torch              :", torch.__version__)
    print("cuda available     :", torch.cuda.is_available())
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        print("gpu                :", torch.cuda.get_device_name(0))
        print("compute capability :", f"{major}.{minor}", "(>= 8.9 means native FP8)")
        print("fp8 supported      :", (major, minor) >= (8, 9))


@app.function(image=quant_image, gpu=config.GPU, volumes=VOLUMES, timeout=600)
def check_quant_env() -> None:
    import importlib.metadata as md

    _gpu_info()
    print("llmcompressor      :", md.version("llmcompressor"))
    print("transformers       :", md.version("transformers"))
    print("compressed-tensors :", md.version("compressed-tensors"))
    print("volume mount ok    :", __import__("os").path.isdir(config.VOLUME_MOUNT))


@app.function(image=vllm_image, gpu=config.GPU, volumes=VOLUMES, timeout=600)
def check_vllm_env() -> None:
    import importlib.metadata as md

    _gpu_info()
    print("vllm               :", md.version("vllm"))
    print("lm-eval            :", md.version("lm-eval"))
    print("transformers       :", md.version("transformers"))
    print("volume mount ok    :", __import__("os").path.isdir(config.VOLUME_MOUNT))


@app.local_entrypoint()
def test_quant() -> None:
    check_quant_env.remote()


@app.local_entrypoint()
def test_vllm() -> None:
    check_vllm_env.remote()
