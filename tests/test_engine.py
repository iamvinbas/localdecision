import numpy as np
import pytest
from conftest import MergingTokenizer, NoTreeBackend

import localdecision as ld
from localdecision.backends.mock import MockBackend
from localdecision.calibration import CalibrationProfile
from localdecision.readout import softmax

STATE = "The invoice payment failed and the refund is late."
TEAM = ld.Choice(
    "Which team?",
    {"billing": "invoice payment refund", "technical": "bug crash", "sales": "pricing upgrade"},
)


def test_answers_are_typed_and_keep_the_callers_keys():
    engine = ld.Engine(MockBackend())
    r = engine.system_one(
        STATE,
        {
            "team": TEAM,
            "money": ld.Noul("Is money involved?", true="payment refund invoice", false="nothing"),
            "severity": ld.Score("How bad?", ["fine", "refund late", "payment failed refund late"]),
        },
    )
    assert list(r.answers) == ["team", "money", "severity"]
    assert r.answers["team"].choice == "billing"
    assert list(r.answers["team"].probabilities) == ["billing", "technical", "sales"]
    assert sum(r.answers["team"].probabilities.values()) == pytest.approx(1.0, abs=1e-5)
    assert r.answers["money"].noul > 0.5
    assert r.answers["severity"].score > 1.5
    assert r.answers["severity"].legend == {
        "0": "fine",
        "1": "refund late",
        "2": "payment failed refund late",
    }
    assert r.usage.output_tokens == 0
    t = r.timing
    assert t.cached_tokens > 0  # chat header + system prompt reused from the persistent cache
    assert r.usage.input_tokens == t.cached_tokens + t.prefix_tokens + t.suffix_tokens
    assert r.diagnostics["team"].views == 3  # auto -> cyclic for 3 options


def test_position_bias_is_removed_by_cyclic_views():
    unbiased = ld.Engine(MockBackend())
    biased = ld.Engine(MockBackend(position_bias=[2.0, 0.5, 0.0]))
    q = {"team": TEAM}
    reference = unbiased.system_one(STATE, q, debias="none").answers["team"].probabilities
    raw = biased.system_one(STATE, q, debias="none").answers["team"].probabilities
    fixed = biased.system_one(STATE, q, debias="cyclic").answers["team"].probabilities
    assert abs(raw["billing"] - reference["billing"]) > 0.05
    for key in reference:
        assert fixed[key] == pytest.approx(reference[key], abs=1e-6)


def test_large_choice_runs_as_an_exact_tournament():
    options = {
        f"opt{i}": ("invoice refund payment" if i == 137 else f"word{i}") for i in range(200)
    }
    r = ld.Engine(MockBackend()).system_one(STATE, {"big": ld.Choice("Pick one", options)})
    answer = r.answers["big"]
    assert answer.choice == "opt137"
    assert r.diagnostics["big"].stages == 2
    scores = np.array([3.0 if i == 137 else 0.0 for i in range(200)])
    expected = softmax(scores)
    got = np.array([answer.probabilities[f"opt{i}"] for i in range(200)])
    np.testing.assert_allclose(got, expected, atol=2e-6)


def test_prefix_tree_changes_cost_not_answers():
    q = {"team": TEAM, "money": ld.Noul("Is money involved?")}
    tree = ld.Engine(MockBackend()).system_one(STATE, q)
    flat = ld.Engine(NoTreeBackend()).system_one(STATE, q)
    for key in q:
        assert tree.answers[key] == flat.answers[key]
    assert tree.usage.input_tokens < flat.usage.input_tokens


def test_non_additive_tokenizer_falls_back_to_common_prefix():
    backend = MockBackend()
    backend.tokenizer = MergingTokenizer()
    engine = ld.Engine(backend)
    assert not engine.prompter.split_ok
    a = engine.system_one(STATE, {"team": TEAM}).answers["team"]
    b = ld.Engine(MockBackend()).system_one(STATE, {"team": TEAM}).answers["team"]
    assert a == b


def test_context_limit_is_enforced_not_truncated():
    engine = ld.Engine(MockBackend(max_context=200))
    with pytest.raises(ValueError, match="context limit"):
        engine.system_one("x" * 500, {"team": TEAM})


def test_calibration_profile_is_applied():
    profile = CalibrationProfile(temperatures={"choice": 4.0}, qhat={"choice": 0.9}, alpha=0.1)
    raw = ld.Engine(MockBackend()).system_one(STATE, {"team": TEAM})
    cal = ld.Engine(MockBackend(), profile).system_one(STATE, {"team": TEAM})
    assert cal.answers["team"].choice == raw.answers["team"].choice
    assert cal.answers["team"].confidence < raw.answers["team"].confidence
    assert cal.diagnostics["team"].temperature == 4.0
    assert cal.diagnostics["team"].prediction_set[0] == "billing"
    off = ld.Engine(MockBackend(), profile).system_one(STATE, {"team": TEAM}, calibrated=False)
    assert off.answers["team"] == raw.answers["team"]


def test_request_dicts_are_validated():
    engine = ld.Engine(MockBackend())
    with pytest.raises(ValueError):
        engine.run(
            {
                "state": "x",
                "questions": {
                    "q": {"type": "choice", "instructions": "?", "criteria": {"a": None}}
                },
            }
        )
