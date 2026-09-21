"""Adapters that turn public benchmarks into LocalDecision evaluation rows (needs ``datasets``).

Each adapter maps one dataset onto one primitive:

    boolq      noul    passage + yes/no question                    (google/boolq)
    sst5       score   5-level sentiment of a movie-review sentence  (SetFit/sst5)
    agnews     choice  4-way news topic                              (fancyzhx/ag_news)
    arc        choice  grade-school science, 3-5 answer options      (allenai/ai2_arc, Challenge)
    banking77  choice  77-way banking intent -> two-stage tournament (legacy-datasets/banking77)
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

Row = dict[str, Any]


def _load(name: str, config: str | None, split: str) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "public datasets need the 'datasets' extra: pip install 'localdecision[datasets]'"
        ) from exc
    return load_dataset(name, config, split=split)


def _sample(items: Any, limit: int | None, seed: int, skip: int) -> list[tuple[int, Any]]:
    """Deterministic shuffle, then the window [skip, skip + limit), with absolute positions."""
    indexed = list(enumerate(items))
    random.Random(seed).shuffle(indexed)
    window = indexed[skip:] if limit is None else indexed[skip : skip + limit]
    return window


def boolq(limit: int | None, seed: int, skip: int) -> list[Row]:
    rows = []
    for i, ex in _sample(_load("google/boolq", None, "validation"), limit, seed, skip):
        question = ex["question"].strip()
        rows.append(
            {
                "id": f"boolq-{i}",
                "state": ex["passage"],
                "questions": {
                    "answer": {
                        "type": "noul",
                        "instructions": question[0].upper() + question[1:] + "?",
                    }
                },
                "labels": {"answer": bool(ex["answer"])},
                "tags": ["boolq"],
            }
        )
    return rows


_SST5_LEVELS = ["Very negative", "Negative", "Neutral", "Positive", "Very positive"]


def sst5(limit: int | None, seed: int, skip: int) -> list[Row]:
    rows = []
    for i, ex in _sample(_load("SetFit/sst5", None, "validation"), limit, seed, skip):
        rows.append(
            {
                "id": f"sst5-{i}",
                "state": ex["text"],
                "questions": {
                    "sentiment": {
                        "type": "score",
                        "instructions": "How positive is the sentiment of this movie-review excerpt?",
                        "criteria": _SST5_LEVELS,
                    }
                },
                "labels": {"sentiment": int(ex["label"])},
                "tags": ["sst5"],
            }
        )
    return rows


_AGNEWS = {
    "World": "International news, politics, conflicts, diplomacy",
    "Sports": "Sports events, athletes, teams",
    "Business": "Companies, markets, economy, finance",
    "Sci/Tech": "Science, technology, computing, the internet",
}


def agnews(limit: int | None, seed: int, skip: int) -> list[Row]:
    ds = _load("fancyzhx/ag_news", None, "test")
    names = ds.features["label"].names
    rows = []
    for i, ex in _sample(ds, limit, seed, skip):
        rows.append(
            {
                "id": f"agnews-{i}",
                "state": ex["text"],
                "questions": {
                    "topic": {
                        "type": "choice",
                        "instructions": "What is the topic of this news article?",
                        "criteria": _AGNEWS,
                    }
                },
                "labels": {"topic": names[ex["label"]]},
                "tags": ["agnews"],
            }
        )
    return rows


def arc(limit: int | None, seed: int, skip: int) -> list[Row]:
    rows = []
    for i, ex in _sample(_load("allenai/ai2_arc", "ARC-Challenge", "test"), limit, seed, skip):
        texts, labels = ex["choices"]["text"], ex["choices"]["label"]
        if len(set(texts)) != len(texts):
            continue
        answer = texts[labels.index(ex["answerKey"])]
        rows.append(
            {
                "id": f"arc-{i}",
                "state": ex["question"],
                "questions": {
                    "answer": {
                        "type": "choice",
                        "instructions": "Which option correctly answers the science question in the state?",
                        "criteria": dict.fromkeys(texts),
                    }
                },
                "labels": {"answer": answer},
                "tags": ["arc"],
            }
        )
    return rows


def banking77(limit: int | None, seed: int, skip: int) -> list[Row]:
    ds = _load("legacy-datasets/banking77", None, "test")
    names = ds.features["label"].names
    criteria = dict.fromkeys(names)
    rows = []
    for i, ex in _sample(ds, limit, seed, skip):
        rows.append(
            {
                "id": f"banking77-{i}",
                "state": ex["text"],
                "questions": {
                    "intent": {
                        "type": "choice",
                        "instructions": "Which intent best describes what the bank customer wants?",
                        "criteria": criteria,
                    }
                },
                "labels": {"intent": names[ex["label"]]},
                "tags": ["banking77"],
            }
        )
    return rows


ADAPTERS: dict[str, Callable[[int | None, int, int], list[Row]]] = {
    "boolq": boolq,
    "sst5": sst5,
    "agnews": agnews,
    "arc": arc,
    "banking77": banking77,
}


def fetch(name: str, limit: int | None = 200, seed: int = 0, skip: int = 0) -> list[Row]:
    """Rows ``skip .. skip + limit`` of a seeded shuffle, so disjoint windows of the same
    dataset can serve as calibration and test sets."""
    if name not in ADAPTERS:
        raise ValueError(f"unknown dataset {name!r}; choose from {', '.join(ADAPTERS)}")
    return ADAPTERS[name](limit, seed, skip)
