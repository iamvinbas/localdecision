"""Turn the outputs of run_public.sh into the markdown tables used in the README.

python benchmarks/make_tables.py results/public-Qwen3-1.7B-8bit
python benchmarks/make_tables.py results/public-Qwen3-0.6B-8bit results/public-Qwen3-1.7B-8bit ...

One folder prints the quality and calibration tables for that model. Several folders print one
row per model and dataset instead, to compare model sizes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from localdecision import evaluation as ev
from localdecision.calibration import CalibrationProfile
from localdecision.metrics import summarize

DATASETS = [
    ("boolq", "BoolQ", "noul"),
    ("sst5", "SST-5", "score (5)"),
    ("agnews", "AG News", "choice (4)"),
    ("arc", "ARC-Challenge", "choice (3–5)"),
    ("banking77", "Banking77", "choice (77)"),
]


def _load(root: Path, key: str) -> dict[str, Any] | None:
    """Summaries of one dataset: 1 view, debiased, and debiased + its own calibration."""
    folder = root / f"{key}-test"
    if not (folder / "summary.json").exists():
        return None
    auto_records = ev.load_jsonl(folder / "records-auto.jsonl")
    out = {
        "none": summarize(ev.load_jsonl(folder / "records-none.jsonl")),
        "auto": summarize(auto_records),
        "profile": None,
        "calibrated": None,
    }
    profile_path = root / f"{key}-calib" / "calibration.json"
    if profile_path.exists():
        profile = CalibrationProfile.load(profile_path)
        out["profile"] = profile
        out["calibrated"] = summarize(ev.recalibrate(auto_records, profile))
    return out


def _automated(cal: dict[str, Any]) -> str:
    """Share of decisions with a one-element set, and how many of those were right."""
    n = cal["decisions"]
    k = round(cal["conformal_singleton_share"] * n)
    if k == 0:
        return "0% | —"
    right = round(cal["conformal_singleton_accuracy"] * k)
    return f"{k / n:.0%} | {right / k:.3f} ({right}/{k})"


def _agreement(auto: dict[str, Any]) -> str:
    """Accuracy when the debiased views agree, and when they do not (a label-free signal)."""
    agree = auto.get("accuracy_when_views_agree")
    disagree = auto.get("accuracy_when_views_disagree")
    if agree is None or disagree is None:
        return "—"
    return f"{agree:.2f} / {disagree:.2f}"


def one_model(root: Path) -> None:
    quality, calib = [], []
    alpha = None
    for key, name, primitive in DATASETS:
        data = _load(root, key)
        if data is None:
            continue
        none, auto, cal, profile = data["none"], data["auto"], data["calibrated"], data["profile"]
        quality.append(
            f"| {name} | {primitive} | {auto['decisions']} | {none['accuracy']:.3f} "
            f"| {auto['accuracy']:.3f} | {auto['unstable_share']:.2f} | {_agreement(auto)} "
            f"| {none['ms_per_decision']:.0f} | {auto['ms_per_decision']:.0f} |"
        )
        if cal is None:
            continue
        alpha = profile.alpha
        kind = next(iter(profile.temperatures))
        calib.append(
            f"| {name} | {profile.temperatures[kind]:.1f} "
            f"| {auto['nll']:.2f} → {cal['nll']:.2f} | {auto['ece']:.3f} → {cal['ece']:.3f} "
            f"| {cal['conformal_coverage']:.3f} | {cal['conformal_mean_set_size']:.1f} "
            f"| {_automated(cal)} |"
        )
    print(
        "| Set | Primitive | n | Accuracy, 1 view | Accuracy, debiased | Views disagree "
        "| Accuracy, views agree / disagree | ms / decision, 1 view | ms / decision, debiased |"
    )
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    print("\n".join(quality))
    if alpha is None:
        print("\n(no <dataset>-calib/calibration.json: calibration table skipped)")
        return
    print(f"\nConformal alpha = {alpha} (target coverage {1 - alpha:.0%})\n")
    print(
        "| Set | Temperature | NLL raw → calibrated | ECE raw → calibrated | Set coverage "
        "| Mean set size | Singleton sets (automated) | Accuracy when automated |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    print("\n".join(calib))


def _model_name(root: Path) -> str:
    for key, _, _ in DATASETS:
        summary = root / f"{key}-test" / "summary.json"
        if summary.exists():
            return json.loads(summary.read_text(encoding="utf-8"))["engine"]["model"]
    return root.name


def compare(roots: list[Path]) -> None:
    """Rows are datasets, columns are models; cells show debiased accuracy and latency."""
    names = [_model_name(root) for root in roots]
    print("| Set | " + " | ".join(names) + " |")
    print("| --- |" + " ---: |" * len(roots))
    # Per model: calibrated ECE of each set, and decisions / automated / automated and right.
    eces: list[list[float]] = [[] for _ in roots]
    totals = [[0, 0, 0] for _ in roots]
    for key, name, _ in DATASETS:
        cells = []
        for i, root in enumerate(roots):
            data = _load(root, key)
            if data is None:
                cells.append("—")
                continue
            auto, cal = data["auto"], data["calibrated"]
            cells.append(f"{auto['accuracy']:.3f} · {auto['ms_per_decision']:.0f} ms")
            if cal is not None:
                eces[i].append(cal["ece"])
                automated = round(cal["conformal_singleton_share"] * cal["decisions"])
                right = round(cal.get("conformal_singleton_accuracy", 0.0) * automated)
                totals[i][0] += cal["decisions"]
                totals[i][1] += automated
                totals[i][2] += right
        print(f"| {name} | " + " | ".join(cells) + " |")
    ece_cells = [f"{sum(e) / len(e):.3f}" if e else "—" for e in eces]
    print("| ECE after calibration, mean of the sets | " + " | ".join(ece_cells) + " |")
    auto_cells = [
        "—" if not n else f"{a / n:.0%} · {r / a:.3f} right" if a else "0%" for n, a, r in totals
    ]
    print("| Automated (singleton sets), all sets pooled | " + " | ".join(auto_cells) + " |")


if __name__ == "__main__":
    folders = [Path(a) for a in sys.argv[1:]]
    if len(folders) == 1:
        one_model(folders[0])
    else:
        compare(folders)
