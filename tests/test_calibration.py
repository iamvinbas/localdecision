import numpy as np
import pytest

from localdecision.calibration import (
    CalibrationProfile,
    Sample,
    fit_profile,
    fit_qhat,
    fit_temperature,
)
from localdecision.metrics import brier, expected_calibration_error, risk_coverage, summarize
from localdecision.readout import log_softmax, softmax


def synthetic(n, k, true_temperature, seed):
    """Logits whose labels are drawn from softmax(z / T): a model overconfident by factor T."""
    rng = np.random.default_rng(seed)
    logps, labels = [], []
    for _ in range(n):
        z = rng.normal(scale=3.0, size=k)
        labels.append(int(rng.choice(k, p=softmax(z / true_temperature))))
        logps.append(log_softmax(z))
    return logps, labels


def test_temperature_is_recovered():
    logps, labels = synthetic(4000, 4, true_temperature=2.0, seed=0)
    assert fit_temperature(logps, labels) == pytest.approx(2.0, rel=0.1)


def test_conformal_sets_reach_the_target_coverage():
    logps, labels = synthetic(2000, 5, true_temperature=1.0, seed=1)
    probs = [np.exp(lp) for lp in logps]
    qhat = fit_qhat(probs[:1000], labels[:1000], alpha=0.1)
    covered = [p[y] >= 1 - qhat for p, y in zip(probs[1000:], labels[1000:], strict=True)]
    assert np.mean(covered) >= 0.87


def test_profile_fit_and_roundtrip(tmp_path):
    logps, labels = synthetic(200, 3, true_temperature=1.5, seed=2)
    profile = fit_profile(
        [Sample("choice", lp, y) for lp, y in zip(logps, labels, strict=True)],
        alpha=0.2,
        fingerprint={"model_id": "m"},
    )
    assert profile.temperatures["choice"] > 1.0
    assert profile.report["choice"]["nll_calibrated"] <= profile.report["choice"]["nll_raw"] + 1e-6
    path = tmp_path / "cal.json"
    profile.save(path)
    loaded = CalibrationProfile.load(path)
    assert loaded == profile
    assert loaded.mismatches({"model_id": "other"}) == ["model_id"]


def test_metrics():
    assert brier(np.array([1.0, 0.0]), 0) == 0.0
    assert brier(np.array([0.5, 0.5]), 0) == pytest.approx(0.5)
    probs = [np.array([0.8, 0.2])] * 10
    labels = [0] * 8 + [1] * 2  # 80% accurate at 80% confidence
    assert expected_calibration_error(probs, labels) == pytest.approx(0.0, abs=1e-12)
    rc = risk_coverage([0.9, 0.8, 0.7, 0.6], [True, True, False, False])
    assert rc["accuracy_at_50pct"] == 1.0
    records = [
        {
            "row": "r",
            "qid": "q",
            "kind": "score",
            "label": 1,
            "probs": [0.1, 0.8, 0.1],
            "agreement": 1.0,
            "request_ms": 10.0,
            "questions_in_request": 1,
        },
    ]
    s = summarize(records)
    assert s["accuracy"] == 1.0 and s["score_mae"] == pytest.approx(0.0)
