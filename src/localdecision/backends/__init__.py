"""Inference backends. ``load_backend`` picks MLX on Apple Silicon and PyTorch elsewhere."""

from __future__ import annotations

import importlib.util
import platform
from typing import Any

from .base import Backend, PrefixSession, Readout, SelfTest

__all__ = [
    "Backend",
    "PrefixSession",
    "Readout",
    "SelfTest",
    "default_backend",
    "default_model",
    "load_backend",
]


def default_backend() -> str:
    apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    if apple_silicon and importlib.util.find_spec("mlx_lm") is not None:
        return "mlx"
    return "torch"


def default_model(backend: str) -> str:
    if backend == "mlx":
        return "mlx-community/Qwen3-4B-Instruct-2507-8bit"
    return "Qwen/Qwen3-4B-Instruct-2507"


def load_backend(name: str = "auto", model: str | None = None, **options: Any) -> Backend:
    """Instantiate a backend. ``options`` are passed through (``batch_size``, ``max_context``...)."""
    options = {k: v for k, v in options.items() if v is not None}
    if name == "auto":
        name = default_backend()
    if name == "mlx":
        from .mlx_backend import MLXBackend

        options.pop("device", None)
        options.pop("dtype", None)
        return MLXBackend(model or default_model("mlx"), **options)
    if name == "torch":
        from .torch_backend import TorchBackend

        options.pop("bits", None)
        return TorchBackend(model or default_model("torch"), **options)
    if name == "mock":
        from .mock import MockBackend

        return MockBackend()
    raise ValueError(f"unknown backend {name!r}; expected auto, mlx, torch or mock")
