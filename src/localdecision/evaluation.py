"""Evaluation harness: run labelled decisions through the engine and score the answers.

Dataset format (JSONL, one state per line; several questions may share a state):

    {"id": "ticket-7",
     "state": "...",                          # text or JSON
     "questions": {"team": {"type": "choice", "instructions": "...", "criteria": {...}}},
     "labels": {"team": "billing"},           # choice: option key · score: level index · noul: true/false
     "tags": ["routing"]}                     # optional, reported separately
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import Sample
from .engine import Engine
from .metrics import summarize
from .prompts import Spec
from .schema import Settings, SystemOneRequest


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if line.strip():
                row = json.loads(line)
                row.setdefault("id", f"{Path(path).stem}-{line_no}")
                rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def label_index(spec: Spec, label: Any) -> int:
    """Map a dataset label onto the index of the candidate it names."""
    if spec.kind == "noul":
        if isinstance(label, str):
            label = label.strip().lower() in {"true", "yes", "1"}
        return 0 if bool(label) else 1
    if spec.kind == "score":
        index = int(label)
        if not 0 <= index < spec.n:
            raise ValueError(f"score label {label!r} out of range for {spec.qid!r}")
        return index
    if label not in spec.keys:
        raise ValueError(f"label {label!r} is not an option of {spec.qid!r}")
    return spec.keys.index(label)


def run(
    engine: Engine,
    rows: Sequence[dict[str, Any]],
    *,
    debias: str = "auto",
    calibrated: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """One record per labelled question. Latency is measured per request (state)."""
    settings = Settings(debias=debias, calibrated=calibrated)
    records: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        request = SystemOneRequest(
            state=row["state"], questions=row["questions"], settings=settings
        )
        t0 = time.perf_counter()
        _, traces = engine.run_with_traces(request)
        ms = (time.perf_counter() - t0) * 1000
        labels = row.get("labels", {})
        for tr in traces:
            if tr.spec.qid not in labels:
                continue
            records.append(
                {
                    "row": row["id"],
                    "qid": tr.spec.qid,
                    "kind": tr.spec.kind,
                    "tags": row.get("tags", []),
                    "keys": list(tr.spec.keys),
                    "label": label_index(tr.spec, labels[tr.spec.qid]),
                    "probs": [round(float(x), 6) for x in tr.probs],
                    "logp_raw": [round(float(x), 6) for x in tr.logp_raw],
                    "agreement": tr.agreement,
                    "views": tr.views,
                    "stages": tr.stages,
                    "format_mass": round(tr.format_mass, 6),
                    "prediction_set": tr.prediction_set,
                    "request_ms": round(ms, 2),
                    "questions_in_request": len(traces),
                }
            )
        if progress:
            progress(i + 1, len(rows))
    return records


def report(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Overall summary plus breakdowns by question type and by tag."""
    out: dict[str, Any] = {"overall": summarize(records), "by_kind": {}, "by_tag": {}}
    for kind in sorted({r["kind"] for r in records}):
        out["by_kind"][kind] = summarize([r for r in records if r["kind"] == kind])
    for tag in sorted({t for r in records for t in r.get("tags", [])}):
        out["by_tag"][tag] = summarize([r for r in records if tag in r.get("tags", [])])
    return out


def samples(records: Sequence[dict[str, Any]]) -> list[Sample]:
    """Calibration samples (uncalibrated pooled log-probabilities + labels)."""
    return [
        Sample(r["kind"], np.asarray(r["logp_raw"], dtype=float), int(r["label"])) for r in records
    ]


def recalibrate(records: Sequence[dict[str, Any]], profile: Any) -> list[dict[str, Any]]:
    """Apply a calibration profile to recorded raw log-probabilities (offline, no model)."""
    from .readout import apply_temperature, conformal_set

    out = []
    for r in records:
        logp = apply_temperature(
            np.asarray(r["logp_raw"], dtype=float), profile.temperature(r["kind"])
        )
        probs = np.exp(logp)
        qhat = profile.threshold(r["kind"])
        out.append(
            {
                **r,
                "probs": [round(float(x), 6) for x in probs],
                "prediction_set": None if qhat is None else conformal_set(probs, qhat),
            }
        )
    return out


# ---------------------------------------------------------------------- pretty printing

_COLUMNS = [
    ("decisions", "n", "{:.0f}"),
    ("accuracy", "acc", "{:.3f}"),
    ("balanced_accuracy", "bal.acc", "{:.3f}"),
    ("nll", "nll", "{:.3f}"),
    ("brier", "brier", "{:.3f}"),
    ("ece", "ece", "{:.3f}"),
    ("accuracy_at_80pct", "acc@80%", "{:.3f}"),
    ("unstable_share", "unstable", "{:.3f}"),
    ("ms_per_decision", "ms/dec", "{:.0f}"),
]


_CONFORMAL = [
    ("conformal_coverage", "coverage", "{:.3f}"),
    ("conformal_singleton_share", "singleton", "{:.3f}"),
]


def format_table(rows: dict[str, dict[str, Any]], extended: bool = False) -> str:
    columns = _COLUMNS + (_CONFORMAL if extended else [])
    header = f"{'':<26}" + "".join(f"{label:>10}" for _, label, _ in columns)
    lines = [header, "-" * len(header)]
    for name, summary in rows.items():
        cells = []
        for key, _, fmt in columns:
            value = summary.get(key)
            cells.append(f"{'—' if value is None else fmt.format(value):>10}")
        lines.append(f"{name[:26]:<26}" + "".join(cells))
    return "\n".join(lines)
