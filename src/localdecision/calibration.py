"""Post-hoc calibration fitted on your own labelled decisions.

Two independent tools, both fitted per question type (``noul``, ``choice``, ``score``):

* **Temperature scaling** rescales pooled log-probabilities by 1/T so that stated probabilities
  match observed accuracy (lower NLL / ECE). It never changes the argmax.
* **Split-conformal prediction sets.** From held-out nonconformity scores s = 1 − p(true label)
  we take the ⌈(n+1)(1−α)⌉-th smallest as q̂. At inference the set {c : p(c) ≥ 1 − q̂}
  contains the true answer with probability ≥ 1 − α (marginally, under exchangeability).
  A set of size one is a principled "safe to act automatically" signal.

A profile is bound to a fingerprint (backend, model, prompt version): changing any of them
invalidates the fit, and the engine warns about it.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .readout import log_softmax, logsumexp

FORMAT = "localdecision.calibration/v1"


@dataclass
class Sample:
    kind: str
    logp: np.ndarray  # pooled, uncalibrated log-probabilities
    label: int


@dataclass
class CalibrationProfile:
    temperatures: dict[str, float] = field(default_factory=dict)
    qhat: dict[str, float] = field(default_factory=dict)
    alpha: float | None = None
    fingerprint: dict[str, Any] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)

    def temperature(self, kind: str) -> float:
        return self.temperatures.get(kind, 1.0)

    def threshold(self, kind: str) -> float | None:
        return self.qhat.get(kind)

    def mismatches(self, fingerprint: dict[str, Any]) -> list[str]:
        return [k for k, v in self.fingerprint.items() if fingerprint.get(k) != v]

    def summary(self) -> dict[str, Any]:
        return {"temperatures": self.temperatures, "qhat": self.qhat, "alpha": self.alpha}

    def save(self, path: str | Path) -> None:
        payload = {"format": FORMAT, **asdict(self)}
        Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> CalibrationProfile:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.pop("format", None) != FORMAT:
            raise ValueError(f"{path} is not a {FORMAT} file")
        return cls(**data)


def nll(logps: Sequence[np.ndarray], labels: Sequence[int], temperature: float = 1.0) -> float:
    beta = 1.0 / temperature
    total = 0.0
    for lp, y in zip(logps, labels, strict=True):
        z = beta * np.asarray(lp)
        total += logsumexp(z) - z[y]
    return total / max(len(labels), 1)


def fit_temperature(
    logps: Sequence[np.ndarray], labels: Sequence[int], lo: float = 0.05, hi: float = 20.0
) -> float:
    """NLL-optimal temperature. NLL is convex in β = 1/T, so golden-section search on log β
    over [1/hi, 1/lo] finds the global optimum."""
    if not labels:
        return 1.0
    a, b = math.log(1.0 / hi), math.log(1.0 / lo)
    phi = (math.sqrt(5.0) - 1.0) / 2.0

    def f(u: float) -> float:
        return nll(logps, labels, 1.0 / math.exp(u))

    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(80):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = f(d)
    return float(1.0 / math.exp((a + b) / 2.0))


def fit_qhat(probs: Sequence[np.ndarray], labels: Sequence[int], alpha: float) -> float:
    """Split-conformal quantile of s = 1 − p(label) with the finite-sample correction."""
    scores = sorted(1.0 - float(p[y]) for p, y in zip(probs, labels, strict=True))
    n = len(scores)
    k = math.ceil((n + 1) * (1.0 - alpha))
    return 1.0 if n == 0 or k > n else scores[k - 1]


def fit_profile(
    samples: Sequence[Sample],
    alpha: float = 0.1,
    fingerprint: dict[str, Any] | None = None,
    seed: int = 0,
) -> CalibrationProfile:
    """Fit temperature and conformal threshold per question type.

    With at least 40 samples of a type, the samples are split in half: temperature on one
    half, conformal quantile on the other, so the coverage guarantee is not optimistic.
    """
    from .metrics import expected_calibration_error

    profile = CalibrationProfile(alpha=alpha, fingerprint=dict(fingerprint or {}))
    kinds = sorted({s.kind for s in samples})
    rng = random.Random(seed)
    for kind in kinds:
        group = [s for s in samples if s.kind == kind]
        rng.shuffle(group)
        if len(group) >= 40:
            half = len(group) // 2
            fit_set, conf_set = group[:half], group[half:]
        else:
            fit_set = conf_set = group
        t = fit_temperature([s.logp for s in fit_set], [s.label for s in fit_set])
        probs = [np.exp(log_softmax(np.asarray(s.logp) / t)) for s in conf_set]
        labels = [s.label for s in conf_set]
        profile.temperatures[kind] = round(t, 4)
        profile.qhat[kind] = round(fit_qhat(probs, labels, alpha), 6)
        raw = [np.exp(log_softmax(np.asarray(s.logp))) for s in conf_set]
        profile.report[kind] = {
            "samples": len(group),
            "split": len(group) >= 40,
            "nll_raw": round(nll([s.logp for s in conf_set], labels), 4),
            "nll_calibrated": round(nll([s.logp for s in conf_set], labels, t), 4),
            "ece_raw": round(expected_calibration_error(raw, labels), 4),
            "ece_calibrated": round(expected_calibration_error(probs, labels), 4),
        }
    return profile


PROFILE_DIR = Path(__file__).parent / "profiles"


def bundled_profile(fingerprint: dict[str, Any]) -> CalibrationProfile | None:
    """The generic profile shipped for this model, if its fingerprint matches exactly."""
    path = PROFILE_DIR / f"{fingerprint.get('model_id', '').replace('/', '--')}.json"
    if not path.is_file():
        return None
    profile = CalibrationProfile.load(path)
    return None if profile.mismatches(fingerprint) else profile
