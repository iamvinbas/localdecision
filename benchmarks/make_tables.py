"""Turn the outputs of run_public.sh into the markdown tables used in the README.

python benchmarks/make_tables.py results/public-Qwen3-1.7B-8bit
"""

from __future__ import annotations

import sys
from pathlib import Path

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


def main(root: Path) -> None:
    profile_path = root / "calibration.json"
    profile = CalibrationProfile.load(profile_path) if profile_path.exists() else None
    quality, calib = [], []
    for key, name, primitive in DATASETS:
        folder = root / f"{key}-test"
        if not folder.exists():
            continue
        none = summarize(ev.load_jsonl(folder / "records-none.jsonl"))
        auto_records = ev.load_jsonl(folder / "records-auto.jsonl")
        auto = summarize(auto_records)
        quality.append(
            f"| {name} | {primitive} | {auto['decisions']} | {none['accuracy']:.3f} | {auto['accuracy']:.3f} "
            f"| {auto['unstable_share']:.3f} | {none['ms_per_decision']:.0f} | {auto['ms_per_decision']:.0f} |"
        )
        if profile is None:
            continue
        cal = summarize(ev.recalibrate(auto_records, profile))
        calib.append(
            f"| {name} | {auto['nll']:.3f} → {cal['nll']:.3f} | {auto['ece']:.3f} → {cal['ece']:.3f} "
            f"| {auto['brier']:.3f} → {cal['brier']:.3f} | {cal['conformal_coverage']:.3f} "
            f"| {cal['conformal_singleton_share']:.3f} | {cal['accuracy_at_80pct']:.3f} |"
        )
    print(
        "| Dataset | Primitive | n | Accuracy, 1 view | Accuracy, debiased | Views disagree | ms/decision, 1 view | ms/decision, debiased |"
    )
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    print("\n".join(quality))
    if profile is None:
        print("\n(no calibration.json in this folder: calibration table skipped)")
        return
    print()
    print(f"Temperatures: {profile.temperatures} · conformal alpha = {profile.alpha}")
    print()
    print(
        "| Dataset | NLL raw → calibrated | ECE raw → calibrated | Brier raw → calibrated | Set coverage | Singleton sets | Accuracy on top 80% |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    print("\n".join(calib))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
