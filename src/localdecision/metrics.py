"""Decision-quality metrics: accuracy, proper scoring rules, calibration, selective risk."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np


def expected_calibration_error(
    probs: Sequence[np.ndarray], labels: Sequence[int], bins: int = 15
) -> float:
    """Top-label ECE with equal-width confidence bins."""
    if not labels:
        return float("nan")
    conf = np.array([float(np.max(p)) for p in probs])
    correct = np.array(
        [int(np.argmax(p)) == y for p, y in zip(probs, labels, strict=True)], dtype=float
    )
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(conf[mask].mean() - correct[mask].mean())
    return float(ece)


def brier(p: np.ndarray, y: int) -> float:
    onehot = np.zeros_like(p)
    onehot[y] = 1.0
    return float(np.sum((p - onehot) ** 2))


def balanced_accuracy(preds: Sequence[int], labels: Sequence[int]) -> float:
    recalls = []
    for c in sorted(set(labels)):
        idx = [i for i, y in enumerate(labels) if y == c]
        recalls.append(sum(preds[i] == c for i in idx) / len(idx))
    return float(np.mean(recalls)) if recalls else float("nan")


def risk_coverage(confidence: Sequence[float], correct: Sequence[bool]) -> dict[str, float]:
    """AURC and accuracy when only the most confident fraction of decisions is automated."""
    order = np.argsort(-np.asarray(confidence), kind="stable")
    ok = np.asarray(correct, dtype=float)[order]
    n = len(ok)
    if n == 0:
        return {}
    cum_acc = np.cumsum(ok) / np.arange(1, n + 1)
    out = {"aurc": float(np.mean(1.0 - cum_acc))}
    for cov in (0.5, 0.8):
        k = max(1, math.ceil(cov * n))
        out[f"accuracy_at_{int(cov * 100)}pct"] = float(cum_acc[k - 1])
    return out


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-decision records (see ``evaluation.run``) into one report."""
    if not records:
        return {"decisions": 0}
    probs = [np.asarray(r["probs"], dtype=float) for r in records]
    labels = [int(r["label"]) for r in records]
    preds = [int(np.argmax(p)) for p in probs]
    correct = [p == y for p, y in zip(preds, labels, strict=True)]
    pmax = [float(np.max(p)) for p in probs]
    out: dict[str, Any] = {
        "decisions": len(records),
        "accuracy": float(np.mean(correct)),
        "balanced_accuracy": balanced_accuracy(preds, labels),
        "nll": float(
            np.mean(
                [-math.log(max(float(p[y]), 1e-12)) for p, y in zip(probs, labels, strict=True)]
            )
        ),
        "brier": float(np.mean([brier(p, y) for p, y in zip(probs, labels, strict=True)])),
        "ece": expected_calibration_error(probs, labels),
        "mean_top_probability": float(np.mean(pmax)),
        **risk_coverage(pmax, correct),
    }
    agreement = [float(r.get("agreement", 1.0)) for r in records]
    unstable = [a < 1.0 for a in agreement]
    out["unstable_share"] = float(np.mean(unstable))
    if any(unstable) and not all(unstable):
        out["accuracy_when_views_agree"] = float(
            np.mean([c for c, u in zip(correct, unstable, strict=True) if not u])
        )
        out["accuracy_when_views_disagree"] = float(
            np.mean([c for c, u in zip(correct, unstable, strict=True) if u])
        )
    if "format_mass" in records[0]:
        out["format_mass"] = float(np.mean([r["format_mass"] for r in records]))
    with_sets = [r for r in records if r.get("prediction_set") is not None]
    if with_sets:
        out["conformal_coverage"] = float(
            np.mean([r["label"] in r["prediction_set"] for r in with_sets])
        )
        out["conformal_mean_set_size"] = float(
            np.mean([len(r["prediction_set"]) for r in with_sets])
        )
        singletons = [r for r in with_sets if len(r["prediction_set"]) == 1]
        out["conformal_singleton_share"] = len(singletons) / len(with_sets)
        if singletons:
            # The "act automatically" rule: how often a one-element set is right.
            out["conformal_singleton_accuracy"] = float(
                np.mean([r["prediction_set"][0] == r["label"] for r in singletons])
            )
    scores = [r for r in records if r["kind"] == "score"]
    if scores:
        exp_level = [float(np.dot(np.arange(len(r["probs"])), r["probs"])) for r in scores]
        out["score_mae"] = float(
            np.mean([abs(e - r["label"]) for e, r in zip(exp_level, scores, strict=True)])
        )
        out["score_within_one"] = float(
            np.mean([abs(int(np.argmax(r["probs"])) - r["label"]) <= 1 for r in scores])
        )
    latencies = sorted({(r["row"], r["request_ms"]) for r in records if "request_ms" in r})
    if latencies:
        ms = np.array([m for _, m in latencies])
        out["request_ms_p50"] = float(np.percentile(ms, 50))
        out["request_ms_p95"] = float(np.percentile(ms, 95))
        per = [r["request_ms"] / r["questions_in_request"] for r in records if "request_ms" in r]
        out["ms_per_decision"] = float(np.mean(per))
    return out
