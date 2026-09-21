"""PyTorch / Hugging Face Transformers backend (CUDA, Apple MPS or CPU).

Same algorithm as the MLX backend: prefill the prefix once into a ``DynamicCache``, replicate
it along the batch axis for each micro-batch of right-padded suffixes, and read only the answer
letters' logits at each suffix's last real position.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from .base import Backend, PrefixSession, Readout, batches_by_length, pad_targets

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


class TorchBackend(Backend):
    name = "torch"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        *,
        device: str = "auto",
        dtype: str = "auto",
        batch_size: int = 8,
        kv_budget_gb: float = 2.0,
        max_context: int = 32768,
        prefill_step: int = 1024,
        selftest: bool = True,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = _pick_device(torch, device)
        torch_dtype = _pick_dtype(torch, dtype, self.device)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        super().__init__(model_id, tokenizer, max_context)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype)
        self.model.to(self.device).eval()
        self.batch_size = batch_size
        self.kv_budget = kv_budget_gb * 1024**3
        self.prefill_step = prefill_step
        self.pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else (tokenizer.eos_token_id or 0)
        )
        self._body = getattr(self.model, self.model.base_model_prefix, None)
        self._lm_head = self.model.get_output_embeddings()
        if self._body is None or self._lm_head is None:
            self._body = self._lm_head = None
        self._inner, self._head = self._body, self._lm_head
        self.shared = True
        self.extra_info = {
            "device": str(self.device),
            "dtype": str(torch_dtype).replace("torch.", ""),
            "batch_size": batch_size,
            "kv_budget_gb": kv_budget_gb,
        }
        if selftest:
            self.run_selftest()

    def open(self, prefix: Sequence[int]) -> PrefixSession:
        return _TorchSession(self, list(prefix))

    def _forward(self, input_ids: Any, cache: Any) -> tuple[Any, Any]:
        use_cache = cache is not None
        if self._inner is not None:
            out = self._inner(input_ids=input_ids, past_key_values=cache, use_cache=use_cache)
            return out.last_hidden_state, out.past_key_values
        out = self.model(input_ids=input_ids, past_key_values=cache, use_cache=use_cache)
        return out.logits, out.past_key_values

    def _readout(
        self, tokens: Any, cache: Any, last: list[int], targets: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        with torch.inference_mode():
            out, _ = self._forward(tokens, cache)
            rows = torch.arange(len(last), device=self.device)
            picked = out[rows, torch.tensor(last, device=self.device)]
            logits = (self._head(picked) if self._inner is not None else picked).float()
            lse = torch.logsumexp(logits, dim=-1)
            sel = torch.gather(logits, 1, torch.from_numpy(targets).to(self.device))
        return sel.cpu().double().numpy(), lse.cpu().double().numpy()

    # ------------------------------------------------------------------ self-test hooks

    def _configure(self, shared: bool, split: bool) -> bool:
        if split and self._body is None:
            return False
        self.shared = shared
        self._inner, self._head = (self._body, self._lm_head) if split else (None, None)
        return True

    def _reference(self, ids: list[int], targets: list[int]) -> Readout:
        torch = self.torch
        with torch.inference_mode():
            logits = (
                self.model(input_ids=torch.tensor([ids], device=self.device)).logits[0, -1].float()
            )
            lse = torch.logsumexp(logits, dim=-1)
            sel = logits[torch.tensor(targets, device=self.device)]
        return Readout(sel.cpu().double().numpy(), float(lse.item()))


class _TorchSession(PrefixSession):
    def __init__(self, backend: TorchBackend, prefix: list[int]) -> None:
        super().__init__(len(prefix))
        self.b = backend
        self.prefix = prefix
        self.cache: Any = None
        self.kv_bytes = 0
        if backend.shared and prefix:
            self._prefill()
        else:
            self.stats.prefix_tokens = 0  # direct mode: the prefix is recomputed per view

    def extend(self, tokens: Sequence[int]) -> _TorchSession:
        tokens = list(tokens)
        child = _TorchSession(self.b, [])
        child.prefix = self.prefix + tokens
        child.stats.prefix_tokens = len(tokens) if self.cache is not None else 0
        if self.cache is not None:
            child.cache = copy.deepcopy(self.cache)
            child._prefill(tokens)
        return child

    def _prefill(self, tokens: list[int] | None = None) -> None:
        """Run ``tokens`` (default: the whole prefix) into the cache, in chunks."""
        torch, b = self.b.torch, self.b
        t0 = time.perf_counter()
        cache = self.cache
        tokens = self.prefix if tokens is None else tokens
        with torch.inference_mode():
            for i in range(0, len(tokens), b.prefill_step):
                chunk = torch.tensor([tokens[i : i + b.prefill_step]], device=b.device)
                if b._inner is not None:
                    cache = b._inner(
                        input_ids=chunk, past_key_values=cache, use_cache=True
                    ).past_key_values
                else:
                    cache = b.model(
                        input_ids=chunk, past_key_values=cache, use_cache=True
                    ).past_key_values
        _sync(torch, b.device)
        self.cache = cache
        self.kv_bytes = _cache_bytes(torch, cache)
        self.stats.prefill_ms += (time.perf_counter() - t0) * 1000

    def _max_batch(self, longest_suffix: int) -> int:
        b = self.b
        if self.cache is None:
            return b.batch_size
        per_row = self.kv_bytes * (1 + longest_suffix / max(len(self.prefix), 1))
        return int(max(1, min(b.batch_size, b.kv_budget // max(per_row, 1))))

    def score(
        self, suffixes: Sequence[Sequence[int]], targets: Sequence[Sequence[int]]
    ) -> list[Readout]:
        torch, b = self.b.torch, self.b
        t0 = time.perf_counter()
        out: list[Readout | None] = [None] * len(suffixes)
        lengths = [len(s) for s in suffixes]
        for batch in batches_by_length(lengths, self._max_batch(max(lengths))):
            if self.cache is not None:
                rows = [list(suffixes[i]) for i in batch]
                cache = copy.deepcopy(self.cache)
                if len(batch) > 1:
                    cache.batch_repeat_interleave(len(batch))
            else:
                rows = [self.prefix + list(suffixes[i]) for i in batch]
                cache = None
                self.stats.prefix_tokens += len(self.prefix) * len(batch)
            width = max(len(r) for r in rows)
            tokens = torch.tensor(
                [r + [b.pad_id] * (width - len(r)) for r in rows], device=b.device
            )
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
        self.cache = None


def _pick_device(torch: Any, device: str) -> Any:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _pick_dtype(torch: Any, dtype: str, device: Any) -> Any:
    if dtype != "auto":
        return getattr(torch, dtype)
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def _sync(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _cache_bytes(torch: Any, cache: Any) -> int:
    total = 0
    for layer in getattr(cache, "layers", []):
        for value in vars(layer).values():
            if isinstance(value, torch.Tensor):
                total += value.numel() * value.element_size()
    return total
