"""Pure math that turns per-view answer-letter logits into one distribution per question.

No model, no I/O: everything here is deterministic numpy and unit-tested in isolation.

Notation: a question has ``n`` candidates in canonical order. A *view* shows a subset of them
in some order (``order[pos] = candidate index``) and yields one logit per displayed letter.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + float(np.log(np.sum(np.exp(x - m))))


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x - logsumexp(x)


def softmax(x: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax(x))


def cyclic_orders(n: int, max_views: int) -> list[list[int]]:
    """Cyclic shifts of ``range(n)``; evenly spaced when there are more than ``max_views``."""
    base = list(range(n))
    shifts = (
        range(n)
        if n <= max_views
        else sorted({round(k * n / max_views) % n for k in range(max_views)})
    )
    return [base[s:] + base[:s] for s in shifts]


def view_orders(n: int, mode: str, max_views: int = 8) -> list[list[int]]:
    """Option orderings to read for one question.

    ``cyclic`` puts every candidate in every position exactly once. If the model's position
    preference is multiplicative, p_view(c) ∝ p(c) · π(pos_view(c)), the geometric mean over
    all cyclic views multiplies every candidate by the same constant (∏ π)^(1/n), so the
    positional prior cancels exactly after renormalization. ``swap`` (original + reversed) is
    the two-view approximation, and is exact for two options.
    """
    base = list(range(n))
    if mode == "none" or n < 2:
        return [base]
    if mode == "swap":
        return [base, base[::-1]]
    if mode == "cyclic":
        return cyclic_orders(n, max_views)
    raise ValueError(f"unknown debias mode: {mode!r}")


@dataclass(frozen=True)
class Pooled:
    logp: np.ndarray  # pooled log-probabilities over the candidates (normalized)
    agreement: float  # share of views whose argmax equals the pooled argmax
    view_argmax: tuple[int, ...]


def pool_views(n: int, views: Sequence[tuple[Sequence[int], np.ndarray]]) -> Pooled:
    """Geometric-mean pooling of views that each cover all ``n`` candidates.

    ``views`` holds ``(order, display_logits)`` pairs. Each view is normalized over its own
    letters first, mapped back to canonical candidate order, then averaged in log space.
    """
    if not views:
        raise ValueError("no views to pool")
    table = np.empty((len(views), n), dtype=np.float64)
    for row, (order, logits) in enumerate(views):
        if len(order) != n or len(logits) != n:
            raise ValueError("pool_views expects views over every candidate")
        table[row, np.asarray(order)] = log_softmax(logits)
    mean = table.mean(axis=0)
    logp = mean - logsumexp(mean)
    top = int(np.argmax(logp))
    argmaxes = tuple(int(i) for i in table.argmax(axis=1))
    agreement = sum(a == top for a in argmaxes) / len(argmaxes)
    return Pooled(logp, agreement, argmaxes)


def chunk_indices(n: int, max_per_chunk: int) -> list[list[int]]:
    """Split ``range(n)`` into the fewest balanced, contiguous chunks of at most ``max_per_chunk``."""
    m = -(-n // max_per_chunk)
    bounds = np.linspace(0, n, m + 1).round().astype(int)
    return [list(range(bounds[i], bounds[i + 1])) for i in range(m)]


def select_finalists(
    chunks: Sequence[Sequence[int]],
    chunk_logp: Sequence[np.ndarray],
    budget: int,
    per_chunk_cap: int = 4,
) -> list[int]:
    """Top-k candidates of every chunk, with k chosen so the final round fits ``budget`` letters."""
    k = max(1, min(per_chunk_cap, budget // len(chunks)))
    finalists: list[int] = []
    for members, logp in zip(chunks, chunk_logp, strict=True):
        best = np.argsort(-np.asarray(logp), kind="stable")[: min(k, len(members))]
        finalists.extend(members[int(j)] for j in sorted(best))
    return finalists


def luce_chain(
    n: int,
    chunks: Sequence[Sequence[int]],
    chunk_logp: Sequence[np.ndarray],
    finalists: Sequence[int],
    final_logp: np.ndarray,
) -> np.ndarray:
    """Merge a two-stage tournament into one distribution over all ``n`` candidates.

    Under Luce's choice axiom every candidate has a weight w(c) and P(c | S) = w(c) / Σ_S w.
    Stage 2 fixes the relative weights of the finalists; stage 1 fixes the ratios inside each
    chunk. A non-finalist is therefore placed relative to its chunk's best finalist a:
    log w(c) = log w(a) + log q(c) - log q(a). For a model that obeys the axiom this reproduces
    the single-stage softmax over all ``n`` candidates exactly.
    """
    position = {c: i for i, c in enumerate(finalists)}
    logw = np.full(n, np.nan)
    logw[np.asarray(finalists)] = final_logp
    for members, q in zip(chunks, chunk_logp, strict=True):
        local = [j for j, c in enumerate(members) if c in position]
        if not local:
            raise ValueError("every chunk needs at least one finalist")
        anchor = max(local, key=lambda j: q[j])
        base = logw[members[anchor]] - q[anchor]
        for j, c in enumerate(members):
            if c not in position:
                logw[c] = base + q[j]
    return logw - logsumexp(logw)


def apply_temperature(logp: np.ndarray, temperature: float) -> np.ndarray:
    if temperature == 1.0:
        return logp
    return log_softmax(logp / temperature)


def choice_confidence(p: np.ndarray) -> float:
    """(n · p_max − 1) / (n − 1): 1 when all mass is on one option, 0 when uniform."""
    n = len(p)
    if n < 2:
        return 1.0
    return float(np.clip((n * float(np.max(p)) - 1.0) / (n - 1.0), 0.0, 1.0))


def margin(p: np.ndarray) -> float:
    if len(p) < 2:
        return 1.0
    top2 = np.sort(p)[-2:]
    return float(top2[1] - top2[0])


def entropy_confidence(p: np.ndarray) -> float:
    n = len(p)
    if n < 2:
        return 1.0
    q = p[p > 0]
    entropy = float(-(q * np.log(q)).sum())
    return float(np.clip(1.0 - entropy / np.log(n), 0.0, 1.0))


def expected_level(p: np.ndarray) -> float:
    return float(np.dot(np.arange(len(p)), p))


def conformal_set(p: np.ndarray, qhat: float) -> list[int]:
    """Split-conformal (LAC) set: every candidate with p ≥ 1 − q̂, most likely first, never empty."""
    members = sorted((int(i) for i in np.flatnonzero(p >= 1.0 - qhat)), key=lambda i: -p[i])
    return members or [int(np.argmax(p))]
