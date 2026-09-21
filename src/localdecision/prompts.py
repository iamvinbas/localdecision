"""Prompt construction.

Every question is turned into a multiple-choice prompt whose answer is a single letter token.
The prompt is laid out so that everything that depends on the state comes first:

    [chat header][system prompt][<state> ... </state>] | [question + options][assistant header]
    \\___________________ shared prefix ______________/   \\_________ per-view suffix ________/

The prefix is tokenized and prefilled once per request; each view (question x option order)
only adds its short suffix. ``Prompter`` verifies at start-up that this split is exact for the
loaded tokenizer (no BPE merge across the boundary) and that every answer letter is one token.
"""

from __future__ import annotations

import json
import string
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .schema import ChoiceQuestion, NoulQuestion, ScoreQuestion, Structured

PROMPT_VERSION = "ld-prompt-v1"

SYSTEM_PROMPT = (
    "You are a decision engine. You read a state and answer one multiple-choice question "
    "about it, judging only from what the state and the question say. "
    "Reply with the letter of the single best option."
)

LETTERS = string.ascii_uppercase
_SENTINEL = "⁣LOCALDECISION-SPLIT⁣"


def render_value(value: Structured, *, compact: bool = False) -> str:
    """Text as-is; JSON structure as JSON with the caller's key order preserved."""
    if isinstance(value, str):
        return value.strip()
    if compact:
        return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
    return json.dumps(value, ensure_ascii=False, indent=2)


def render_state(state: Structured) -> str:
    return f"<state>\n{render_value(state)}\n</state>\n\n"


@dataclass(frozen=True)
class Spec:
    """A question normalized into an ordered list of candidates."""

    qid: str
    kind: str  # "noul" | "choice" | "score"
    instructions: str
    keys: tuple[str, ...]  # answer keys in canonical order
    lines: tuple[str, ...]  # display text of each candidate (without its letter)
    legend: dict[str, str] | None = None

    @property
    def n(self) -> int:
        return len(self.keys)


def build_spec(qid: str, question: NoulQuestion | ChoiceQuestion | ScoreQuestion) -> Spec:
    instructions = render_value(question.instructions)
    if isinstance(question, NoulQuestion):
        crit = question.criteria
        yes = (
            "Yes"
            if crit is None or crit.true is None
            else f"Yes: {render_value(crit.true, compact=True)}"
        )
        no = (
            "No"
            if crit is None or crit.false is None
            else f"No: {render_value(crit.false, compact=True)}"
        )
        return Spec(qid, "noul", instructions, ("true", "false"), (yes, no))
    if isinstance(question, ChoiceQuestion):
        keys = tuple(question.criteria)
        lines = tuple(
            key if desc is None else f"{key}: {render_value(desc, compact=True)}"
            for key, desc in question.criteria.items()
        )
        return Spec(qid, "choice", instructions, keys, lines)
    if isinstance(question, ScoreQuestion):
        levels = [render_value(level, compact=True) for level in question.criteria]
        keys = tuple(str(i) for i in range(len(levels)))
        lines = tuple(f"Level {i}: {text}" for i, text in enumerate(levels))
        return Spec(qid, "score", instructions, keys, lines, dict(zip(keys, levels, strict=True)))
    raise TypeError(f"unsupported question type: {type(question).__name__}")


def render_question(spec: Spec, order: Sequence[int]) -> str:
    """Question block for one view: candidates ``order`` shown as A, B, C, ..."""
    if spec.kind == "score":
        top = spec.n - 1
        if order[0] <= order[-1]:
            header = (
                f"Options (an ordered scale from level 0, the lowest, to level {top}, the highest):"
            )
        else:
            header = f"Options (an ordered scale listed from level {top}, the highest, down to level 0, the lowest):"
    else:
        header = "Options:"
    body = "\n".join(f"{LETTERS[pos]}. {spec.lines[i]}" for pos, i in enumerate(order))
    return (
        f"Question: {spec.instructions}\n\n{header}\n{body}\n\n"
        "Answer with the letter of the best option."
    )


class Prompter:
    """Binds the prompt layout to a tokenizer's chat template and verifies it."""

    def __init__(self, tokenizer: Any, system_prompt: str = SYSTEM_PROMPT) -> None:
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError(
                "the tokenizer has no chat template; LocalDecision needs an instruction-tuned model"
            )
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self._system_role = self._supports_system_role()
        self._head, self._tail = self._split_template()
        self.template_ok = self._check_template()
        self.letter_ids = self._letter_tokens()
        self.split_ok = self.template_ok and self._check_boundary()
        self.head_ok = self.split_ok and self._check_head()

    # ----------------------------------------------------------------- rendering

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def chat(self, user: str) -> str:
        """Full prompt text for one user message, ending where the answer letter goes."""
        if self._system_role:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user},
            ]
        else:
            messages = [{"role": "user", "content": f"{self.system_prompt}\n\n{user}"}]
        return self._apply(messages)

    @property
    def head_text(self) -> str:
        """Chat header + system prompt: identical for every request."""
        return self._head

    def prefix_text(self, state: Structured) -> str:
        return self._head + render_state(state)

    def suffix_text(self, question_block: str) -> str:
        return question_block + self._tail

    def full_text(self, state: Structured, question_block: str) -> str:
        return self.chat(render_state(state) + question_block)

    @property
    def max_letters(self) -> int:
        return len(self.letter_ids)

    # ----------------------------------------------------------------- checks

    def _apply(self, messages: list[dict[str, str]]) -> str:
        tok = self.tokenizer
        try:
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _supports_system_role(self) -> bool:
        try:
            text = self._apply(
                [{"role": "system", "content": "S-probe"}, {"role": "user", "content": "U-probe"}]
            )
        except Exception:
            return False
        return "S-probe" in text

    def _split_template(self) -> tuple[str, str]:
        text = self.chat(_SENTINEL)
        if text.count(_SENTINEL) != 1:
            raise ValueError("could not locate the user message inside the chat template")
        head, tail = text.split(_SENTINEL)
        return head, tail

    def _check_template(self) -> bool:
        sample = render_state({"k": "v\nw"}) + "Question: q?\n\nOptions:\nA. x\nB. y"
        return self.chat(sample) == self._head + sample + self._tail

    def _letter_tokens(self) -> list[int]:
        spec = Spec("probe", "choice", "Which option?", ("a", "b"), ("a", "b"))
        base_text = self.full_text("Probe state.", render_question(spec, [0, 1]))
        base = self.encode(base_text)
        ids: list[int] = []
        for letter in LETTERS:
            ext = self.encode(base_text + letter)
            if len(ext) != len(base) + 1 or ext[: len(base)] != base or ext[-1] in ids:
                break
            ids.append(ext[-1])
        if len(ids) < 2:
            raise ValueError("answer letters are not single tokens after the chat template")
        return ids

    def _check_head(self) -> bool:
        head = self.encode(self._head)
        return all(
            head + self.encode(render_state(state)) == self.encode(self.prefix_text(state))
            for state in ("A plain state.", {"items": [1, 2], "note": "x"}, "  leading spaces")
        )

    def _check_boundary(self) -> bool:
        spec = Spec("probe", "choice", "Is it ok?", ("yes", "no"), ("yes", "no"))
        for state in ("A plain state.", {"items": [1, 2], "note": "x"}):
            prefix = self.prefix_text(state)
            suffix = self.suffix_text(render_question(spec, [1, 0]))
            if self.encode(prefix) + self.encode(suffix) != self.encode(prefix + suffix):
                return False
        return True
