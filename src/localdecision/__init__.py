"""LocalDecision: typed, calibrated decisions from open-weight LLMs, fully local.

import localdecision as ld

engine = ld.load()  # MLX on Apple Silicon, PyTorch elsewhere
result = engine.system_one(
    "Help! My payouts have been failing for 3 days.",
    {
        "urgent": ld.Noul("Does this convey urgency?"),
        "team": ld.Choice("Which team should handle this?", ["billing", "technical", "sales"]),
    },
)
result.answers["team"].choice, result.answers["urgent"].noul
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .calibration import CalibrationProfile, bundled_profile
from .engine import Engine, Trace
from .schema import (
    Choice,
    ChoiceAnswer,
    ChoiceQuestion,
    Noul,
    NoulAnswer,
    NoulQuestion,
    Score,
    ScoreAnswer,
    ScoreQuestion,
    Settings,
    SystemOneRequest,
    SystemOneResponse,
)

__version__ = "0.1.0"

__all__ = [
    "CalibrationProfile",
    "Choice",
    "ChoiceAnswer",
    "ChoiceQuestion",
    "Engine",
    "Noul",
    "NoulAnswer",
    "NoulQuestion",
    "Score",
    "ScoreAnswer",
    "ScoreQuestion",
    "Settings",
    "SystemOneRequest",
    "SystemOneResponse",
    "Trace",
    "load",
]


def load(
    model: str | None = None,
    backend: str = "auto",
    calibration: str | Path | CalibrationProfile | None = "auto",
    **options: Any,
) -> Engine:
    """Load a model and return a ready ``Engine``.

    ``backend``: ``"auto"`` (MLX on Apple Silicon, else PyTorch), ``"mlx"``, ``"torch"`` or
    ``"mock"``. ``options`` go to the backend: ``batch_size``, ``max_context``,
    ``kv_budget_gb``, ``bits`` (MLX), ``device`` / ``dtype`` (PyTorch).

    ``calibration``: a profile path or object; ``"auto"`` (default) uses the generic profile
    shipped for this exact model and prompt version if there is one; ``None`` or ``"none"``
    returns raw model probabilities.
    """
    from .backends import load_backend

    engine = Engine(load_backend(backend, model, **options))
    if isinstance(calibration, CalibrationProfile):
        engine.calibration = calibration
    elif calibration == "auto":
        engine.calibration = bundled_profile(engine.fingerprint())
    elif calibration not in (None, "none"):
        engine.calibration = CalibrationProfile.load(calibration)
    return engine
