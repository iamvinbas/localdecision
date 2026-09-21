"""Deterministic, model-free backend used by the test-suite and for dry runs of the pipeline.

It turns the token ids back into text (character-level tokenizer), finds the state and the
options shown in the prompt, and gives every option a logit from a pluggable ``scorer`` plus
an optional per-position bias. That position bias is multiplicative in probability space,
which is exactly the kind of bias the cyclic views are designed to cancel.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence

import numpy as np

from ..readout import logsumexp
from .base import Backend, PrefixSession, Readout

Scorer = Callable[[str, str, list[str]], list[float]]

_OPTION = re.compile(r"^([A-Z])\. (.*)$", re.M)
_QUESTION = re.compile(r"Question: (.*?)\n\n", re.S)
_WORD = re.compile(r"[a-z0-9]+")


class CharTokenizer:
    """One token per character; a minimal chat template."""

    chat_template = "mock"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **_):
        text = "".join(f"<|{m['role']}|>\n{m['content']}<|end|>\n" for m in messages)
        return text + ("<|assistant|>\n" if add_generation_prompt else "")


def overlap_scorer(state: str, question: str, options: list[str]) -> list[float]:
    """Logit = number of distinct option words that also appear in the state."""
    words = set(_WORD.findall(state.lower()))
    return [float(len(set(_WORD.findall(o.lower())) & words)) for o in options]


class MockBackend(Backend):
    name = "mock"

    def __init__(
        self,
        scorer: Scorer = overlap_scorer,
        position_bias: Sequence[float] | None = None,
        format_mass: float = 0.95,
        max_context: int = 1_000_000,
    ) -> None:
        super().__init__("mock/overlap", CharTokenizer(), max_context)
        self.scorer = scorer
        self.position_bias = list(position_bias or [])
        self.log_format_mass = float(np.log(format_mass))
        self.calls = 0

    def open(self, prefix: Sequence[int]) -> PrefixSession:
        return _MockSession(self, list(prefix))


class _MockSession(PrefixSession):
    def __init__(self, backend: MockBackend, prefix: list[int]) -> None:
        super().__init__(len(prefix))
        self.backend = backend
        self.prefix = prefix

    def extend(self, tokens):
        child = _MockSession(self.backend, self.prefix + list(tokens))
        child.stats.prefix_tokens = len(tokens)
        return child

    def score(self, suffixes, targets):
        t0 = time.perf_counter()
        tok = self.backend.tokenizer
        out: list[Readout] = []
        for suffix, target in zip(suffixes, targets, strict=True):
            self.backend.calls += 1
            text = tok.decode(self.prefix + list(suffix))
            head, _, rest = text.partition("</state>")
            state = head.split("<state>", 1)[-1]
            question = m.group(1) if (m := _QUESTION.search(rest)) else ""
            shown = dict(_OPTION.findall(rest))
            letters = [chr(t) for t in target]
            scores = self.backend.scorer(state, question, [shown[x] for x in letters])
            bias = self.backend.position_bias
            logits = np.array(
                [s + (bias[i] if i < len(bias) else 0.0) for i, s in enumerate(scores)],
                dtype=np.float64,
            )
            out.append(Readout(logits, logsumexp(logits) - self.backend.log_format_mass))
            self.stats.suffix_tokens += len(suffix)
        self.stats.probes += len(out)
        self.stats.batches += 1
        self.stats.readout_ms += (time.perf_counter() - t0) * 1000
        return out
