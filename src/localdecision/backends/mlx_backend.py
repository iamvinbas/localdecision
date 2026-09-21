"""Apple Silicon backend on MLX / mlx-lm.

Shared-prefix scoring:

1. The prefix is prefilled once into the model's native cache (in chunks of ``prefill_step``).
2. For a micro-batch of ``b`` suffixes, the cache is replicated ``b`` times along the batch axis
   and the right-padded suffixes run in one forward pass. Right padding is safe for any causal
   model: a real token never attends to the padding that follows it.
3. Only the final hidden state of each suffix goes through the LM head, and only the logits of
   the answer letters (plus the full-vocabulary log-sum-exp) are copied back to the host.

The batch size adapts to the prefix: replicating a long prefix costs memory, so ``b`` is capped
by ``kv_budget_gb``. At load time a self-test compares this path with a plain full forward pass
and falls back to direct (no reuse) scoring if the two disagree.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from .base import Backend, PrefixSession, Readout, batches_by_length, pad_targets

DEFAULT_MODEL = "mlx-community/Qwen3-1.7B-8bit"


class MLXBackend(Backend):
    name = "mlx"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        *,
        batch_size: int = 8,
        kv_budget_gb: float = 2.0,
        max_context: int = 32768,
        prefill_step: int = 1024,
        bits: int | None = None,
        selftest: bool = True,
    ) -> None:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm import load

        self.mx = mx
        model, wrapper = load(model_id)
        tokenizer = getattr(wrapper, "_tokenizer", wrapper)
        super().__init__(model_id, tokenizer, max_context)
        if bits is not None and not _is_quantized(model):
            nn.quantize(model, group_size=64, bits=bits)
        self.model = model
        self.batch_size = batch_size
        self.kv_budget = kv_budget_gb * 1024**3
        self.prefill_step = prefill_step
        self.pad_id = _pad_id(tokenizer)
        self._body, self._lm_head = _split_model(model)
        self._inner, self._head = self._body, self._lm_head
        self.shared = True
        self.extra_info = {
            "device": str(mx.default_device()),
            "quantized": _is_quantized(model),
            "batch_size": batch_size,
            "kv_budget_gb": kv_budget_gb,
        }
        if selftest:
            self.run_selftest()

    # ------------------------------------------------------------------ primitives

    def open(self, prefix: Sequence[int]) -> PrefixSession:
        return _MLXSession(self, list(prefix))

    def _forward(self, tokens: Any, cache: Any) -> Any:
        """Hidden states when the model splits cleanly into body + head, else full logits."""
        if self._inner is not None:
            return self._inner(tokens, cache=cache)
        return self.model(tokens, cache=cache)

    def _readout(
        self, tokens: Any, cache: Any, last: list[int], targets: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        mx = self.mx
        out = self._forward(tokens, cache)
        rows = mx.arange(len(last))
        picked = out[rows, mx.array(last)]
        logits = (self._head(picked) if self._inner is not None else picked).astype(mx.float32)
        lse = mx.logsumexp(logits, axis=-1)
        sel = mx.take_along_axis(logits, mx.array(targets), axis=-1)
        mx.eval(sel, lse)
        return np.array(sel, dtype=np.float64), np.array(lse, dtype=np.float64)

    # ------------------------------------------------------------------ self-test hooks

    def _configure(self, shared: bool, split: bool) -> bool:
        if split and self._body is None:
            return False
        self.shared = shared
        self._inner, self._head = (self._body, self._lm_head) if split else (None, None)
        return True

    def _reference(self, ids: list[int], targets: list[int]) -> Readout:
        mx = self.mx
        logits = self.model(mx.array([ids]))[0, -1].astype(mx.float32)
        lse = mx.logsumexp(logits)
        sel = logits[mx.array(targets)]
        mx.eval(sel, lse)
        return Readout(np.array(sel, dtype=np.float64), float(lse.item()))


class _MLXSession(PrefixSession):
    def __init__(self, backend: MLXBackend, prefix: list[int]) -> None:
        super().__init__(len(prefix))
        self.b = backend
        self.prefix = prefix
        self.cache: list[Any] | None = None
        self.kv_bytes = 0
        if backend.shared and prefix:
            self._prefill()
        else:
            self.stats.prefix_tokens = 0  # direct mode: the prefix is recomputed per view

    def _prefill(self, tokens: list[int] | None = None) -> None:
        """Run ``tokens`` (default: the whole prefix) into the cache, in chunks."""
        from mlx_lm.models.cache import make_prompt_cache

        mx, b = self.b.mx, self.b
        t0 = time.perf_counter()
        if self.cache is None:
            self.cache = make_prompt_cache(b.model)
        tokens = self.prefix if tokens is None else tokens
        for i in range(0, len(tokens), b.prefill_step):
            b._forward(mx.array([tokens[i : i + b.prefill_step]]), self.cache)
            mx.eval(_cache_arrays(self.cache))
        self.kv_bytes = sum(a.nbytes for a in _cache_arrays(self.cache))
        self.stats.prefill_ms += (time.perf_counter() - t0) * 1000

    def extend(self, tokens: Sequence[int]) -> _MLXSession:
        tokens = list(tokens)
        child = _MLXSession(self.b, [])
        child.prefix = self.prefix + tokens
        child.stats.prefix_tokens = len(tokens) if self.cache is not None else 0
        if self.cache is not None:
            child.cache = _clone_cache(self.cache, 1)
            child._prefill(tokens)
        return child

    def _max_batch(self, longest_suffix: int) -> int:
        b = self.b
        if self.cache is None:
            return b.batch_size
        per_row = self.kv_bytes * (1 + longest_suffix / max(len(self.prefix), 1))
        return int(max(1, min(b.batch_size, b.kv_budget // max(per_row, 1))))

    def score(
        self, suffixes: Sequence[Sequence[int]], targets: Sequence[Sequence[int]]
    ) -> list[Readout]:
        mx, b = self.b.mx, self.b
        t0 = time.perf_counter()
        out: list[Readout | None] = [None] * len(suffixes)
        lengths = [len(s) for s in suffixes]
        for batch in batches_by_length(lengths, self._max_batch(max(lengths))):
            if self.cache is not None:
                rows = [list(suffixes[i]) for i in batch]
                cache = _clone_cache(self.cache, len(batch))
            else:
                rows = [self.prefix + list(suffixes[i]) for i in batch]
                cache = None
                self.stats.prefix_tokens += len(self.prefix) * len(batch)
            width = max(len(r) for r in rows)
            tokens = mx.array([r + [b.pad_id] * (width - len(r)) for r in rows])
            tgt, counts = pad_targets([targets[i] for i in batch])
            sel, lse = b._readout(tokens, cache, [len(r) - 1 for r in rows], tgt)
            for j, i in enumerate(batch):
                out[i] = Readout(sel[j, : counts[j]], float(lse[j]))
            self.stats.batches += 1
        self.stats.probes += len(suffixes)
        self.stats.suffix_tokens += sum(lengths)
        self.stats.readout_ms += (time.perf_counter() - t0) * 1000
        return out  # type: ignore[return-value]

    def close(self) -> None:
        if self.kv_bytes > 256 * 1024**2:
            self.b.mx.clear_cache()
        self.cache = None


# ---------------------------------------------------------------------- helpers


def _split_model(model: Any) -> tuple[Any, Any]:
    """(body, head) such that head(body(x)) == model(x); (None, None) if not recognised."""
    inner = getattr(model, "model", None)
    if inner is None or not callable(inner):
        return None, None
    head = getattr(model, "lm_head", None)
    if head is None:
        embed = getattr(inner, "embed_tokens", None)
        head = getattr(embed, "as_linear", None)
    return (inner, head) if head is not None else (None, None)


def _is_quantized(model: Any) -> bool:
    import mlx.nn as nn

    return any(
        isinstance(m, nn.QuantizedLinear | nn.QuantizedEmbedding) for _, m in model.named_modules()
    )


def _pad_id(tokenizer: Any) -> int:
    for attr in ("pad_token_id", "eos_token_id"):
        value = getattr(tokenizer, attr, None)
        if isinstance(value, int):
            return value
    return 0


def _cache_arrays(cache: Any) -> list[Any]:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    return [v for _, v in tree_flatten([c.state for c in cache]) if isinstance(v, mx.array)]


def _clone_cache(cache: list[Any], n: int) -> list[Any]:
    return [_clone_one(c, n) for c in cache]


def _clone_one(c: Any, n: int) -> Any:
    """Copy one layer cache with every array repeated ``n`` times along the batch axis."""
    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache, CacheList, KVCache, RotatingKVCache

    def rep(a: Any) -> Any:
        return None if a is None else mx.repeat(a, n, axis=0)

    if type(c) is KVCache:
        new = KVCache()
        keys, values = c.state
        new.keys, new.values, new.offset = rep(keys), rep(values), c.offset
        return new
    if isinstance(c, RotatingKVCache):
        new = copy.copy(c)
        new.keys, new.values = rep(c.keys), rep(c.values)
        return new
    if isinstance(c, ArraysCache):
        new = copy.copy(c)
        new.cache = [rep(a) for a in c.cache]
        return new
    if isinstance(c, CacheList):
        new = copy.copy(c)
        new.caches = tuple(_clone_one(x, n) for x in c.caches)
        return new
    raise TypeError(f"cannot replicate cache type {type(c).__name__}")
