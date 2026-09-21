"""Integration tests on real weights. Skipped unless LOCALDECISION_TEST_MODEL is set, e.g.

LOCALDECISION_TEST_MODEL=mlx-community/Qwen3-0.6B-8bit pytest tests/test_model.py
"""

import os

import pytest

import localdecision as ld

MODEL = os.environ.get("LOCALDECISION_TEST_MODEL")
pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(not MODEL, reason="LOCALDECISION_TEST_MODEL not set"),
]

QUESTIONS = {
    "urgent": ld.Noul("Does this convey urgency?"),
    "team": ld.Choice(
        "Which team should handle this?",
        {"billing": "Payments, payouts, refunds", "technical": "Bugs, outages", "sales": None},
    ),
    "mood": ld.Score("How frustrated is the customer?", ["Calm", "Frustrated", "Very angry"]),
}
STATE = "Help! My payouts have been failing for 3 days and nobody answers my emails."


@pytest.fixture(scope="module")
def engine():
    backend = os.environ.get("LOCALDECISION_TEST_BACKEND", "auto")
    return ld.load(MODEL, backend)


def test_selftest_passes(engine):
    assert engine.backend.selftest.passed
    assert engine.prompter.split_ok and engine.prompter.head_ok


def test_shared_prefix_matches_direct_scoring(engine):
    backend = engine.backend
    shared = engine.system_one(STATE, QUESTIONS, debias="none")
    mode = backend.shared
    backend.shared = False
    engine._head = None
    try:
        direct = engine.system_one(STATE, QUESTIONS, debias="none")
    finally:
        backend.shared = mode
        engine._head = None
    for key in ("team", "mood"):
        for option, p in shared.answers[key].probabilities.items():
            assert p == pytest.approx(direct.answers[key].probabilities[option], abs=0.05)
    assert shared.answers["urgent"].noul == pytest.approx(direct.answers["urgent"].noul, abs=0.05)


def test_answers_are_sensible(engine):
    r = engine.system_one(STATE, QUESTIONS)
    assert r.answers["team"].choice == "billing"
    assert r.answers["urgent"].noul > 0.5
    assert r.usage.output_tokens == 0


def test_large_choice_tournament(engine):
    options = {f"city_{i}": None for i in range(60)}
    options["city_41"] = "Paris, the capital of France"
    r = engine.system_one(
        "The Eiffel Tower is in this city.",
        {"city": ld.Choice("Which city is described?", options)},
    )
    assert r.diagnostics["city"].stages == 2
    assert r.answers["city"].choice == "city_41"
