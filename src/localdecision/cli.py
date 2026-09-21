"""Command line: ``localdecision serve | ask | eval | fetch | calibrate | report | selftest``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


def _model_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("model")
    g.add_argument("--backend", default="auto", choices=["auto", "mlx", "torch", "mock"])
    g.add_argument("--model", default=None, help="Hugging Face repo id or local path")
    g.add_argument(
        "--batch-size", type=int, default=None, help="max views per forward pass (default 8)"
    )
    g.add_argument(
        "--max-context", type=int, default=None, help="token limit per view (default 32768)"
    )
    g.add_argument(
        "--kv-budget-gb", type=float, default=None, help="memory for replicated prefix caches"
    )
    g.add_argument("--bits", type=int, default=None, choices=[4, 8], help="MLX: quantize in memory")
    g.add_argument("--device", default=None, help="torch: cuda, mps or cpu")
    g.add_argument("--dtype", default=None, help="torch: bfloat16, float16 or float32")
    g.add_argument(
        "--calibration",
        default="auto",
        help="profile JSON; 'auto' = generic profile shipped for the model (if any); 'none' = raw",
    )


def _engine(args: argparse.Namespace):
    from . import load

    t0 = time.perf_counter()
    options = {
        "batch_size": args.batch_size,
        "max_context": args.max_context,
        "kv_budget_gb": args.kv_budget_gb,
        "bits": args.bits,
        "device": args.device,
        "dtype": args.dtype,
    }
    if args.backend == "mock":
        options = {}
    engine = load(args.model, args.backend, args.calibration, **options)
    info = engine.describe()
    selftest = info.get("selftest", {})
    _err(
        f"loaded {info['model']} on {info['backend']} in {time.perf_counter() - t0:.1f}s"
        + (
            f" · self-test {'passed' if selftest.get('passed') else 'FAILED'} ({selftest.get('mode')})"
            if selftest
            else ""
        )
    )
    if engine.calibration is not None:
        stale = engine.calibration.mismatches(engine.fingerprint())
        if stale:
            _err(f"warning: calibration profile was fitted with a different {', '.join(stale)}")
        else:
            _err(f"calibration: temperatures {engine.calibration.temperatures}")
    return engine


def _err(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------- commands


def cmd_serve(args: argparse.Namespace) -> None:
    from .server import serve

    serve(_engine(args), args.host, args.port)


def cmd_ask(args: argparse.Namespace) -> None:
    text = (
        sys.stdin.read() if args.request == "-" else Path(args.request).read_text(encoding="utf-8")
    )
    engine = _engine(args)
    response = engine.run(json.loads(text))
    print(response.model_dump_json(indent=2, exclude_none=True))


def cmd_eval(args: argparse.Namespace) -> None:
    from . import evaluation as ev

    rows = ev.load_jsonl(args.data)[: args.limit]
    engine = _engine(args)
    ev.run(engine, rows[:1], debias="none")  # warm-up: compile kernels outside the timings
    out_dir = Path(args.output) if args.output else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    table: dict[str, Any] = {}
    reports: dict[str, Any] = {}
    for mode in args.debias.split(","):
        t0 = time.perf_counter()

        def progress(done: int, total: int, mode: str = mode, t0: float = t0) -> None:
            if done % 25 == 0 or done == total:
                _err(f"  [{mode}] {done}/{total} states · {time.perf_counter() - t0:.0f}s")

        records = ev.run(engine, rows, debias=mode, progress=progress)
        rep = ev.report(records)
        reports[mode] = rep
        table[f"{mode} · all"] = rep["overall"]
        for kind, summary in rep["by_kind"].items():
            table[f"{mode} · {kind}"] = summary
        if out_dir:
            ev.write_jsonl(out_dir / f"records-{mode}.jsonl", records)
    print(ev.format_table(table))
    if args.by_tag:
        for mode, rep in reports.items():
            print(f"\n[{mode}] by tag")
            print(ev.format_table(rep["by_tag"]))
    if out_dir:
        meta = {
            "data": str(args.data),
            "rows": len(rows),
            "engine": engine.describe(),
            "fingerprint": engine.fingerprint(),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "reports": reports,
        }
        (out_dir / "summary.json").write_text(
            json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8"
        )
        _err(f"wrote {out_dir}/summary.json and records-*.jsonl")


def cmd_fetch(args: argparse.Namespace) -> None:
    from . import evaluation as ev
    from .public_data import fetch

    rows = fetch(args.name, limit=args.limit, seed=args.seed, skip=args.skip)
    ev.write_jsonl(args.output, rows)
    _err(f"wrote {len(rows)} rows to {args.output}")


def cmd_calibrate(args: argparse.Namespace) -> None:
    from . import evaluation as ev
    from .calibration import fit_profile

    records = [r for path in args.records for r in ev.load_jsonl(path)]
    fingerprint: dict[str, Any] = {}
    summary = Path(args.records[0]).with_name("summary.json")
    if summary.exists():
        fingerprint = json.loads(summary.read_text(encoding="utf-8")).get("fingerprint", {})
    profile = fit_profile(ev.samples(records), alpha=args.alpha, fingerprint=fingerprint)
    profile.save(args.output)
    print(
        json.dumps(
            {"temperatures": profile.temperatures, "qhat": profile.qhat, "report": profile.report},
            indent=2,
        )
    )
    _err(f"wrote {args.output}")


def cmd_report(args: argparse.Namespace) -> None:
    """Re-score saved records, optionally under a calibration profile (no model needed)."""
    from . import evaluation as ev
    from .calibration import CalibrationProfile

    profile = CalibrationProfile.load(args.calibration) if args.calibration else None
    table: dict[str, Any] = {}
    for path in args.records:
        records = ev.load_jsonl(path)
        if profile is not None:
            records = ev.recalibrate(records, profile)
        rep = ev.report(records)
        name = Path(path).parent.name or Path(path).stem
        table[f"{name} · all"] = rep["overall"]
        if len(rep["by_kind"]) > 1:
            for kind, summary in rep["by_kind"].items():
                table[f"{name} · {kind}"] = summary
        if args.json:
            print(json.dumps(rep, indent=2))
    print(ev.format_table(table, extended=profile is not None))


def cmd_selftest(args: argparse.Namespace) -> None:
    engine = _engine(args)
    print(json.dumps(engine.describe(), indent=2, default=str))


# ---------------------------------------------------------------------- entry point


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="localdecision", description="Typed, calibrated decisions from a local LLM."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("serve", help="run the HTTP API")
    _model_args(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("ask", help="answer one request (JSON file or - for stdin)")
    _model_args(p)
    p.add_argument("request")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("eval", help="score a labelled JSONL dataset")
    _model_args(p)
    p.add_argument("data")
    p.add_argument("--debias", default="none,auto", help="comma-separated modes to compare")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output", default=None, help="directory for records and summary")
    p.add_argument("--by-tag", action="store_true", help="also print a table per tag")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("fetch", help="export a public benchmark as evaluation JSONL")
    p.add_argument("name", choices=["boolq", "sst5", "agnews", "arc", "banking77"])
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--limit", type=int, default=200)
    p.add_argument(
        "--skip", type=int, default=0, help="skip the first N rows of the seeded shuffle"
    )
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("calibrate", help="fit temperature + conformal thresholds from eval records")
    p.add_argument("records", nargs="+", help="records-*.jsonl written by `eval --output`")
    p.add_argument("-o", "--output", required=True)
    p.add_argument(
        "--alpha", type=float, default=0.1, help="conformal miscoverage (0.1 -> 90%% coverage)"
    )
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("report", help="summarize saved eval records (optionally recalibrated)")
    p.add_argument("records", nargs="+")
    p.add_argument("--calibration", default=None, help="apply this profile to the raw log-probs")
    p.add_argument("--json", action="store_true", help="also print the full report as JSON")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("selftest", help="load a model and check the shared-prefix path")
    _model_args(p)
    p.set_defaults(func=cmd_selftest)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
