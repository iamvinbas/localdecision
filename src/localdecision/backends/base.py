"""Backend contract.

A backend does one thing: prefill a token prefix once, then read the next-token logits of a
few target tokens at the end of many short suffixes that all continue that prefix. It never
samples or decodes. Engines, prompts and math live elsewhere, so a backend is ~200 lines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Readout:
    logits: np.ndarray  # logits of the requested target tokens, float64
    lse: float  # log-sum-exp over the full vocabulary at the same position


@dataclass
class SessionStats:
    prefill_ms: float = 0.0
    readout_ms: float = 0.0
    prefix_tokens: int = 0
    suffix_tokens: int = 0
    probes: int = 0
    batches: int = 0


class PrefixSession(ABC):
    """A prefilled prefix. ``score`` may be called several times (e.g. tournament rounds)."""

    def __init__(self, prefix_len: int) -> None:
        self.stats = SessionStats(prefix_tokens=prefix_len)

    @abstractmethod
    def score(
        self, suffixes: Sequence[Sequence[int]], targets: Sequence[Sequence[int]]
    ) -> list[Readout]:
        """For each suffix, logits of its target tokens at the suffix's last position."""

    def extend(self, tokens: Sequence[int]) -> PrefixSession:
        """A child session whose prefix is this prefix + ``tokens`` (computed once).

        Lets several views of one question share their question header on top of the
        shared state: a two-level prefix tree."""
        raise NotImplementedError

    def close(self) -> None:  # noqa: B027 - optional hook
        pass

    def __enter__(self) -> PrefixSession:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@dataclass
class SelfTest:
    """Result of comparing the shared-prefix path with a plain full forward pass."""

    passed: bool
    max_abs_diff: float
    mode: str  # "shared-prefix" or "direct"
    detail: str = ""


class Backend(ABC):
    name: str = "base"

    def __init__(self, model_id: str, tokenizer: Any, max_context: int) -> None:
        self.model_id = model_id
        self.tokenizer = tokenizer
        self.max_context = max_context
        self.selftest: SelfTest | None = None
        self.extra_info: dict[str, Any] = {}

    @property
    def model_name(self) -> str:
        return self.model_id.rstrip("/").split("/")[-1]

    @abstractmethod
    def open(self, prefix: Sequence[int]) -> PrefixSession:
        """Prefill ``prefix`` and return a session to score suffixes against it."""

    # ------------------------------------------------------------------ self-test hooks

    def _configure(self, shared: bool, split: bool) -> bool:
        """Select a scoring path. ``shared``: reuse the prefix cache; ``split``: run only the
        final hidden states through the LM head. Return False if the path is unavailable."""
        return shared and split

    def _reference(self, ids: list[int], targets: list[int]) -> Readout:
        """Target logits and log-sum-exp from one plain full forward pass, no cache."""
        raise NotImplementedError

    def run_selftest(self, tolerance: float = 5e-2) -> SelfTest:
        """Check the fast path against a plain forward pass; fall back until one agrees.

        Paths are tried from fastest to safest: shared prefix with head-only readout, shared
        prefix with full logits, then direct scoring (prefix recomputed per suffix).
        """
        tok = self.tokenizer
        text = tok.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": "Is the sky usually blue on a clear day? Reply A for yes or B for no.",
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = list(tok.encode(text, add_special_tokens=False))
        prefix, tails = ids[:-9], [ids[-9:], ids[-9:-4]]
        target = list(tok.encode("A B C D", add_special_tokens=False))[:4]
        reference = [self._reference(prefix + t, target) for t in tails]
        notes: list[str] = []
        attempts = [("shared-prefix", True, True), ("shared-prefix", True, False)]
        attempts += [("direct", False, True), ("direct", False, False)]
        for mode, shared, split in attempts:
            label = mode if split else f"{mode}, full logits"
            if not self._configure(shared, split):
                continue
            try:
                with self.open(prefix) as session:
                    got = session.score(tails, [target, target])
            except Exception as exc:  # e.g. a cache type that cannot be replicated
                notes.append(f"{label}: {type(exc).__name__}: {exc}")
                continue
            worst = max(
                float(np.max(np.abs((g.logits - g.lse) - (r.logits - r.lse))))
                for g, r in zip(got, reference, strict=True)
            )
            if worst < tolerance:
                self.selftest = SelfTest(True, round(worst, 6), label, "; ".join(notes))
                return self.selftest
            notes.append(f"{label}: max |Δ log p| = {worst:.4f}")
        self._configure(False, False)
        self.selftest = SelfTest(False, float("inf"), "direct, full logits", "; ".join(notes))
        return self.selftest

    def describe(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "backend": self.name,
            "model_id": self.model_id,
            "model": self.model_name,
            "max_context": self.max_context,
        }
        if self.selftest is not None:
            info["selftest"] = {
                "passed": self.selftest.passed,
                "max_abs_diff": self.selftest.max_abs_diff,
                "mode": self.selftest.mode,
                "detail": self.selftest.detail,
            }
        info.update(self.extra_info)
        return info


def batches_by_length(lengths: Sequence[int], max_batch: int) -> list[list[int]]:
    """Indices grouped into batches of similar length (longest first) to minimize padding."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    return [order[i : i + max_batch] for i in range(0, len(order), max_batch)]


def pad_targets(targets: Sequence[Sequence[int]]) -> tuple[np.ndarray, list[int]]:
    width = max(len(t) for t in targets)
    out = np.zeros((len(targets), width), dtype=np.int64)
    for row, t in enumerate(targets):
        out[row, : len(t)] = t
    return out, [len(t) for t in targets]
