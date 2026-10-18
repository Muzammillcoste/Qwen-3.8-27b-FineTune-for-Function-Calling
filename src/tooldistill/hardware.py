"""Hardware/environment preflight (spec H3, L #17). Read-only; allocates no model memory.

Fails safely: no GPU or too little free GPU memory raises a clear error and never produces any
score. Never prints tokens or secrets.
"""

from __future__ import annotations

import importlib.metadata as md
import platform
import shutil
import subprocess
from typing import Any

PACKAGES = ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "tokenizers",
            "huggingface_hub", "safetensors", "jsonschema", "PyYAML", "Jinja2")


class NoGPUError(RuntimeError):
    pass


class InsufficientGPUMemoryError(RuntimeError):
    pass


def _version(pkg: str) -> str | None:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def _nvidia_smi() -> str | None:
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=20)
        return r.stdout if r.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _ram_gb() -> float | None:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return round(int(line.split()[1]) / 1e6, 1)
    except OSError:
        return None
    return None


def preflight(out_dir: str = ".", require_gpu: bool = True, min_free_gpu_gb: float = 12.0,
              cuda=None) -> dict[str, Any]:
    """``cuda`` may be injected for tests (an object with torch.cuda's API)."""
    import torch

    cuda = cuda or torch.cuda
    rep: dict[str, Any] = {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {p: _version(p) for p in PACKAGES},
        "nvidia_smi": _nvidia_smi(), "ram_gb": _ram_gb(),
        "disk_free_gb": round(shutil.disk_usage(out_dir).free / 1e9, 1),
        "cuda_available": bool(cuda.is_available()),
    }
    if not rep["cuda_available"]:
        rep["status"] = "no_gpu"
        if require_gpu:
            raise NoGPUError("No CUDA GPU visible. Enable a GPU accelerator (T4) in the Kaggle "
                             "notebook settings. No evaluation or training was run.")
        return rep
    rep["device_count_visible"] = cuda.device_count()
    rep["gpu_name"] = cuda.get_device_name(0)
    rep["compute_capability"] = ".".join(map(str, cuda.get_device_capability(0)))
    free, total = cuda.mem_get_info(0)
    rep["gpu_total_gb"] = round(total / 1e9, 2)
    rep["gpu_free_gb"] = round(free / 1e9, 2)
    rep["bf16_supported"] = bool(cuda.is_bf16_supported())
    if rep["gpu_free_gb"] < min_free_gpu_gb:
        rep["status"] = "insufficient_gpu_memory"
        raise InsufficientGPUMemoryError(
            f"Only {rep['gpu_free_gb']} GB free on GPU 0 (< {min_free_gpu_gb} GB required). "
            "Restart the session or free memory. Model size/sequence length are NOT reduced "
            "automatically.")
    rep["status"] = "ok"
    return rep
