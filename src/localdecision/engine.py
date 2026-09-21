"""The decision engine: one state + typed questions in, typed answers with probabilities out.

Pipeline for one request:

1. **Plan.** Every question becomes a candidate list (``Spec``) and a set of *views*: the same
   multiple-choice question with the options in different orders (see ``readout.view_orders``).
   Choices with more options than letters become a two-stage tournament.
2. **Prefill once.** The chat header, system prompt and state form a prefix shared by every
   view of every question. The backend encodes it a single time.
3. **Read, don't generate.** Each view's short suffix is run against the cached prefix and only
   the logits of its answer letters are read. No token is ever sampled.
4. **Pool and calibrate.** Views are pooled in log space (cancelling position bias), tournaments
   are merged with Luce chaining, a calibration temperature is applied, and the result is
   rendered as a typed answer plus diagnostics.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import readout as R
from .backends.base import Backend, PrefixSession, Readout, SessionStats
from .calibration import CalibrationProfile
from .prompts import PROMPT_VERSION, Prompter, Spec, build_spec, render_question, render_state
from .schema import (
    ChoiceAnswer,
    Diagnostics,
    NoulAnswer,
    ScoreAnswer,
    Settings,
    SystemOneRequest,
    SystemOneResponse,
    Timing,
    Usage,
)

MIN_SHARED_HEADER = 8  # tokens; below this a child session costs more than it saves


@dataclass
class View:
    spec: int  # index of the question in the request
    stage: int  # 1, or 2 for the final round of a tournament
    group: int  # chunk index for stage-1 tournament views, -1 otherwise
    order: list[int]  # candidate indices displayed as A, B, C, ...
    suffix: list[int] = field(default_factory=list)
    readout: Readout | None = None


@dataclass
class _Tournament:
    chunks: list[list[int]]
    chunk_logp: list[np.ndarray] = field(default_factory=list)
    finalists: list[int] = field(default_factory=list)


@dataclass
class Trace:
    """Everything known about one answered question; the evaluation harness consumes it."""

    spec: Spec
    logp_raw: np.ndarray  # pooled log-probabilities before calibration
    probs: np.ndarray  # final probabilities (after calibration)
    agreement: float
    views: int
    stages: int
    format_mass: float
    temperature: float
    prediction_set: list[int] | None


class Engine:
    """Thread-safe front door. One engine wraps one loaded model."""

    def __init__(
        self,
        backend: Backend,
        calibration: CalibrationProfile | None = None,
        *,
        prompter: Prompter | None = None,
        max_options_per_view: int = 26,
        chunk_size: int = 20,
    ) -> None:
        self.backend = backend
        self.prompter = prompter or Prompter(backend.tokenizer)
        self.calibration = calibration
        self.max_per_view = min(max_options_per_view, self.prompter.max_letters)
        self.chunk_size = min(chunk_size, self.max_per_view)
        self._lock = threading.Lock()
        self._head: tuple[list[int], PrefixSession] | None = None

    def head_session(self) -> tuple[list[int], PrefixSession] | None:
        """Chat header + system prompt, prefilled once and reused by every request."""
        if not self.prompter.head_ok:
            return None
        if self._head is None:
            ids = self.prompter.encode(self.prompter.head_text)
            self._head = (ids, self.backend.open(ids))
        return self._head

    # ------------------------------------------------------------------ public API

    @property
    def model_name(self) -> str:
        return self.backend.model_name

    def fingerprint(self) -> dict[str, Any]:
        return {
            "backend": self.backend.name,
            "model_id": self.backend.model_id,
            "prompt_version": PROMPT_VERSION,
            "letters": self.prompter.max_letters,
        }

    def describe(self) -> dict[str, Any]:
        info = self.backend.describe()
        info.update(
            prompt_version=PROMPT_VERSION,
            answer_letters=self.prompter.max_letters,
            shared_prefix_tokenization=self.prompter.split_ok,
            calibration=None if self.calibration is None else self.calibration.summary(),
        )
        return info

    def system_one(
        self, state: Any, questions: dict[str, Any], **settings: Any
    ) -> SystemOneResponse:
        """Convenience call: ``engine.system_one(state, {"q": Noul("...")}, debias="swap")``."""
        request = SystemOneRequest(
            state=state, questions=questions, settings=Settings(**settings) if settings else None
        )
        return self.run(request)

    def run(self, request: SystemOneRequest | dict[str, Any]) -> SystemOneResponse:
        return self.run_with_traces(request)[0]

    def run_with_traces(
        self, request: SystemOneRequest | dict[str, Any]
    ) -> tuple[SystemOneResponse, list[Trace]]:
        if isinstance(request, dict):
            request = SystemOneRequest.model_validate(request)
        settings = request.settings or Settings()
        with self._lock:
            return self._run(request, settings)

    # ------------------------------------------------------------------ planning

    def _debias_mode(self, kind: str, n: int, settings: Settings) -> str:
        mode = settings.debias
        if n < 2:
            return "none"
        if mode == "auto":
            return "cyclic" if kind == "choice" and n <= 4 else "swap"
        if mode == "cyclic" and kind == "score":
            return "swap"  # keep an ordinal scale monotone on screen
        return mode

    def _plan(
        self, specs: list[Spec], settings: Settings
    ) -> tuple[list[View], dict[int, _Tournament]]:
        views: list[View] = []
        tournaments: dict[int, _Tournament] = {}
        for si, spec in enumerate(specs):
            if spec.n > self.max_per_view:
                t = _Tournament(R.chunk_indices(spec.n, self.chunk_size))
                tournaments[si] = t
                chunk_mode = "none" if settings.debias == "none" else "swap"
                for g, members in enumerate(t.chunks):
                    for order in R.view_orders(len(members), chunk_mode):
                        views.append(View(si, 1, g, [members[i] for i in order]))
            else:
                mode = self._debias_mode(spec.kind, spec.n, settings)
                for order in R.view_orders(spec.n, mode, settings.max_views):
                    views.append(View(si, 1, -1, order))
        return views, tournaments

    # ------------------------------------------------------------------ execution

    def _run(
        self, request: SystemOneRequest, settings: Settings
    ) -> tuple[SystemOneResponse, list[Trace]]:
        t0 = time.perf_counter()
        specs = [build_spec(qid, q) for qid, q in request.questions.items()]
        views, tournaments = self._plan(specs, settings)

        runner = _SessionRunner(self, request.state)
        try:
            runner.score(specs, views)
            final_views: list[View] = []
            for si, t in tournaments.items():
                self._finish_stage_one(t, [v for v in views if v.spec == si])
                mode = self._debias_mode("choice", len(t.finalists), settings)
                for order in R.view_orders(len(t.finalists), mode, settings.max_views):
                    final_views.append(View(si, 2, -1, [t.finalists[i] for i in order]))
            if final_views:
                runner.score(specs, final_views)
        finally:
            runner.close()

        all_views = views + final_views
        traces = [
            self._aggregate(
                spec, [v for v in all_views if v.spec == si], tournaments.get(si), settings
            )
            for si, spec in enumerate(specs)
        ]
        stats = runner.stats
        answers = {tr.spec.qid: _answer(tr.spec, tr.probs) for tr in traces}
        diagnostics = timing = None
        if settings.diagnostics:
            diagnostics = {tr.spec.qid: _diagnostics(tr) for tr in traces}
            timing = Timing(
                total_ms=round((time.perf_counter() - t0) * 1000, 2),
                prefill_ms=round(stats.prefill_ms, 2),
                readout_ms=round(stats.readout_ms, 2),
                cached_tokens=runner.cached_tokens,
                prefix_tokens=stats.prefix_tokens,
                suffix_tokens=stats.suffix_tokens,
                probes=stats.probes,
                batches=stats.batches,
            )
        response = SystemOneResponse(
            model=self.model_name,
            answers=answers,
            usage=Usage(
                input_tokens=runner.cached_tokens + stats.prefix_tokens + stats.suffix_tokens,
                output_tokens=0,
            ),
            diagnostics=diagnostics,
            timing=timing,
        )
        return response, traces

    def _finish_stage_one(self, t: _Tournament, views: list[View]) -> None:
        """Pool each chunk's views and pick the finalists for the last round."""
        for g, members in enumerate(t.chunks):
            local = {c: j for j, c in enumerate(members)}
            pooled = R.pool_views(
                len(members),
                [([local[c] for c in v.order], v.readout.logits) for v in views if v.group == g],
            )
            t.chunk_logp.append(pooled.logp)
        t.finalists = R.select_finalists(t.chunks, t.chunk_logp, self.max_per_view)

    def _aggregate(
        self, spec: Spec, views: list[View], tournament: _Tournament | None, settings: Settings
    ) -> Trace:
        if tournament is None:
            pooled = R.pool_views(spec.n, [(v.order, v.readout.logits) for v in views])
            logp_raw, stages = pooled.logp, 1
        else:
            fin = tournament.finalists
            local = {c: j for j, c in enumerate(fin)}
            pooled = R.pool_views(
                len(fin),
                [([local[c] for c in v.order], v.readout.logits) for v in views if v.stage == 2],
            )
            logp_raw = R.luce_chain(
                spec.n, tournament.chunks, tournament.chunk_logp, fin, pooled.logp
            )
            stages = 2

        format_mass = float(
            np.mean([np.exp(R.logsumexp(v.readout.logits) - v.readout.lse) for v in views])
        )
        use_cal = self.calibration is not None and settings.calibrated
        temperature = self.calibration.temperature(spec.kind) if use_cal else 1.0
        probs = np.exp(R.apply_temperature(logp_raw, temperature))
        qhat = self.calibration.threshold(spec.kind) if use_cal else None
        pset = R.conformal_set(probs, qhat) if qhat is not None else None
        return Trace(
            spec=spec,
            logp_raw=logp_raw,
            probs=probs,
            agreement=pooled.agreement,
            views=len(views),
            stages=stages,
            format_mass=format_mass,
            temperature=temperature,
            prediction_set=pset,
        )


class _SessionRunner:
    """Owns the prefilled prefix for one request and tokenizes view suffixes against it."""

    def __init__(self, engine: Engine, state: Any) -> None:
        self.engine = engine
        self.state = state
        self.prompter = engine.prompter
        self.backend = engine.backend
        self.sessions: list[PrefixSession] = []
        self.session: PrefixSession | None = None
        self.prefix: list[int] = []
        self.cached_tokens = 0  # prompt tokens served from the persistent header cache

    def score(self, specs: Sequence[Spec], views: Sequence[View]) -> None:
        p = self.prompter
        blocks = [render_question(specs[v.spec], v.order) for v in views]
        if p.split_ok:
            if self.session is None and not self._open_on_head():
                self._open(p.encode(p.prefix_text(self.state)))
            suffixes = [p.encode(p.suffix_text(b)) for b in blocks]
        else:
            fulls = [p.encode(p.full_text(self.state, b)) for b in blocks]
            if self.session is None or any(f[: len(self.prefix)] != self.prefix for f in fulls):
                self._open(fulls[0][: _common_prefix_len(fulls)])
            suffixes = [f[len(self.prefix) :] for f in fulls]

        limit = self.backend.max_context
        for v, suffix in zip(views, suffixes, strict=True):
            total = len(self.prefix) + len(suffix)
            if total > limit:
                raise ValueError(
                    f"question {specs[v.spec].qid!r} needs {total} tokens, above the context "
                    f"limit of {limit}; shorten the state or raise --max-context"
                )
            v.suffix = suffix

        # Second level of the prefix tree: views of one question share the question header
        # (and every option line that happens to coincide), so compute it once per question.
        flat: list[View] = []
        by_question: dict[int, list[View]] = {}
        for v in views:
            by_question.setdefault(v.spec, []).append(v)
        for group in by_question.values():
            shared = _common_prefix_len([v.suffix for v in group]) if len(group) > 1 else 0
            if shared < MIN_SHARED_HEADER:
                flat.extend(group)
                continue
            try:
                child = self.session.extend(group[0].suffix[:shared])
            except NotImplementedError:
                flat.extend(group)
                continue
            self.sessions.append(child)
            try:
                self._read(child, group, shared)
            finally:
                child.close()
        if flat:
            self._read(self.session, flat, 0)

    def _read(self, session: PrefixSession, views: Sequence[View], skip: int) -> None:
        targets = [self.prompter.letter_ids[: len(v.order)] for v in views]
        readouts = session.score([v.suffix[skip:] for v in views], targets)
        for v, r in zip(views, readouts, strict=True):
            v.readout = r

    def _open_on_head(self) -> bool:
        head = self.engine.head_session()
        if head is None:
            return False
        head_ids, head_session = head
        state_ids = self.prompter.encode(render_state(self.state))
        try:
            self.session = head_session.extend(state_ids)
        except NotImplementedError:
            return False
        self.prefix = head_ids + state_ids
        self.cached_tokens = len(head_ids)
        self.sessions.append(self.session)
        return True

    def _open(self, prefix: list[int]) -> None:
        if self.session is not None:
            self.session.close()
        self.prefix = prefix
        self.session = self.backend.open(prefix)
        self.sessions.append(self.session)

    @property
    def stats(self) -> SessionStats:
        total = SessionStats()
        for s in self.sessions:
            for name in vars(total):
                setattr(total, name, getattr(total, name) + getattr(s.stats, name))
        return total

    def close(self) -> None:
        if self.session is not None:
            self.session.close()


def _common_prefix_len(seqs: Sequence[Sequence[int]]) -> int:
    """Longest common token prefix, leaving at least one token in every suffix."""
    limit = min(len(s) for s in seqs) - 1
    first = np.asarray(seqs[0][:limit])
    n = limit
    for s in seqs[1:]:
        diff = np.flatnonzero(np.asarray(s[:limit]) != first)
        if diff.size:
            n = min(n, int(diff[0]))
    return max(n, 0)


def _round(x: float) -> float:
    return round(float(x), 6)


def _answer(spec: Spec, p: np.ndarray) -> NoulAnswer | ChoiceAnswer | ScoreAnswer:
    if spec.kind == "noul":
        return NoulAnswer(noul=_round(p[0]))
    probabilities = {k: _round(v) for k, v in zip(spec.keys, p, strict=True)}
    confidence = _round(R.choice_confidence(p))
    if spec.kind == "choice":
        return ChoiceAnswer(
            choice=spec.keys[int(np.argmax(p))], probabilities=probabilities, confidence=confidence
        )
    return ScoreAnswer(
        score=_round(R.expected_level(p)),
        legend=dict(spec.legend or {}),
        probabilities=probabilities,
        confidence=confidence,
    )


def _diagnostics(tr: Trace) -> Diagnostics:
    return Diagnostics(
        views=tr.views,
        agreement=_round(tr.agreement),
        margin=_round(R.margin(tr.probs)),
        entropy_confidence=_round(R.entropy_confidence(tr.probs)),
        format_mass=_round(tr.format_mass),
        temperature=_round(tr.temperature),
        stages=tr.stages,
        prediction_set=None
        if tr.prediction_set is None
        else [tr.spec.keys[i] for i in tr.prediction_set],
    )
