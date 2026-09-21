"""Wire format of the decision API.

The request/response shape follows the System One contract (``POST /v1/systemone``):
a ``state`` plus a map of typed ``questions`` in, one typed ``answer`` per question out.
Everything LocalDecision adds on top (``settings``, ``diagnostics``, ``timing``) is optional
and lives in separate top-level fields, so strict clients can ignore it.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Structured = str | dict[str, Any] | list[Any]
"""Instructions, criteria and state may be plain text or JSON structure."""

MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 10
MAX_QUESTIONS = 512


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, dict | list):
        return len(value) == 0
    return False


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- questions


class NoulCriteria(_Strict):
    true: Structured | None = None
    false: Structured | None = None


class NoulQuestion(_Strict):
    """Yes/no question. The answer is the probability that the answer is *yes*."""

    type: Literal["noul"] = "noul"
    instructions: Structured
    criteria: NoulCriteria | None = None

    def __init__(self, instructions: Structured | None = None, /, **data: Any) -> None:
        if instructions is not None:
            data["instructions"] = instructions
        if "true" in data or "false" in data:
            data["criteria"] = {"true": data.pop("true", None), "false": data.pop("false", None)}
        super().__init__(**data)

    @field_validator("instructions")
    @classmethod
    def _instructions(cls, v: Structured) -> Structured:
        if _is_blank(v):
            raise ValueError("instructions must not be empty")
        return v


class ChoiceQuestion(_Strict):
    """Pick exactly one option from a set. ``criteria`` maps option -> description (or null)."""

    type: Literal["choice"] = "choice"
    instructions: Structured
    criteria: dict[str, Structured | None]

    def __init__(
        self,
        instructions: Structured | None = None,
        criteria: dict[str, Structured | None] | list[str] | None = None,
        /,
        **data: Any,
    ) -> None:
        if instructions is not None:
            data["instructions"] = instructions
        if criteria is not None:
            data["criteria"] = dict.fromkeys(criteria) if isinstance(criteria, list) else criteria
        super().__init__(**data)

    @field_validator("instructions")
    @classmethod
    def _instructions(cls, v: Structured) -> Structured:
        if _is_blank(v):
            raise ValueError("instructions must not be empty")
        return v

    @field_validator("criteria")
    @classmethod
    def _criteria(cls, v: dict[str, Structured | None]) -> dict[str, Structured | None]:
        if len(v) < 2:
            raise ValueError("a choice needs at least 2 options")
        if len(v) > MAX_CHOICE_OPTIONS:
            raise ValueError(f"a choice accepts at most {MAX_CHOICE_OPTIONS} options")
        if any(not key.strip() for key in v):
            raise ValueError("option names must not be empty")
        return v


class ScoreQuestion(_Strict):
    """Rate the state on an ordered rubric. ``criteria`` lists the levels from low to high."""

    type: Literal["score"] = "score"
    instructions: Structured
    criteria: list[Structured]

    def __init__(
        self,
        instructions: Structured | None = None,
        criteria: list[Structured] | None = None,
        /,
        **data: Any,
    ) -> None:
        if instructions is not None:
            data["instructions"] = instructions
        if criteria is not None:
            data["criteria"] = criteria
        super().__init__(**data)

    @field_validator("instructions")
    @classmethod
    def _instructions(cls, v: Structured) -> Structured:
        if _is_blank(v):
            raise ValueError("instructions must not be empty")
        return v

    @field_validator("criteria")
    @classmethod
    def _criteria(cls, v: list[Structured]) -> list[Structured]:
        if len(v) < 2:
            raise ValueError("a score needs at least 2 levels")
        if len(v) > MAX_SCORE_LEVELS:
            raise ValueError(f"a score accepts at most {MAX_SCORE_LEVELS} levels")
        if any(_is_blank(level) for level in v):
            raise ValueError("score levels must not be empty")
        return v


Question = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]

# Short aliases for application code: ``Choice("Which team?", {...})``.
Noul = NoulQuestion
Choice = ChoiceQuestion
Score = ScoreQuestion


# --------------------------------------------------------------------------- request

DebiasMode = Literal["auto", "none", "swap", "cyclic"]


class Settings(_Strict):
    """LocalDecision-specific knobs. All optional; defaults are what the server uses."""

    debias: DebiasMode = Field(
        "auto",
        description=(
            "How many option orderings to read and pool. 'none': one view. 'swap': original + "
            "reversed. 'cyclic': every cyclic shift (up to max_views). 'auto': cyclic for <= 4 "
            "options, swap otherwise."
        ),
    )
    max_views: int = Field(8, ge=1, le=26, description="Cap on views per question for 'cyclic'.")
    calibrated: bool = Field(True, description="Apply the loaded calibration profile, if any.")
    diagnostics: bool = Field(True, description="Return per-question diagnostics.")


class SystemOneRequest(_Strict):
    state: Structured
    model: str = "localdecision-latest"
    questions: dict[str, Question]
    settings: Settings | None = None

    @field_validator("state")
    @classmethod
    def _state(cls, v: Structured) -> Structured:
        if _is_blank(v):
            raise ValueError("state must not be empty")
        return v

    @model_validator(mode="after")
    def _questions(self) -> SystemOneRequest:
        if not self.questions:
            raise ValueError("at least one question is required")
        if len(self.questions) > MAX_QUESTIONS:
            raise ValueError(f"at most {MAX_QUESTIONS} questions per request")
        return self


# --------------------------------------------------------------------------- response


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


class Diagnostics(BaseModel):
    """How the answer was produced and how much to trust it."""

    views: int = Field(description="Option orderings read and pooled (after stage 1 + 2).")
    agreement: float = Field(description="Share of views whose top option equals the answer.")
    margin: float = Field(description="Top probability minus runner-up.")
    entropy_confidence: float = Field(description="1 - H(p) / log(n).")
    format_mass: float = Field(
        description="Mean share of the full next-token distribution on the answer letters."
    )
    temperature: float = Field(description="Calibration temperature applied (1.0 = raw).")
    stages: int = Field(1, description="2 when a large choice was solved as a tournament.")
    prediction_set: list[str] | None = Field(
        None, description="Conformal prediction set, when a calibration profile provides one."
    )


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int = 0


class Timing(BaseModel):
    """Where the time went. Token counts are tokens actually run through the model."""

    total_ms: float
    prefill_ms: float = Field(description="State (and question headers) prefill.")
    readout_ms: float = Field(description="Batched view suffixes + answer-letter readout.")
    cached_tokens: int = Field(
        0, description="Prompt tokens reused from the persistent header cache."
    )
    prefix_tokens: int = Field(
        description="Shared tokens computed once (state + question headers)."
    )
    suffix_tokens: int = Field(description="Per-view tokens (options + assistant header).")
    probes: int = Field(description="Views read.")
    batches: int = Field(description="Forward passes over view suffixes.")


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage
    diagnostics: dict[str, Diagnostics] | None = None
    timing: Timing | None = None

    @property
    def nouls(self) -> dict[str, NoulAnswer]:
        return {k: a for k, a in self.answers.items() if isinstance(a, NoulAnswer)}

    @property
    def choices(self) -> dict[str, ChoiceAnswer]:
        return {k: a for k, a in self.answers.items() if isinstance(a, ChoiceAnswer)}

    @property
    def scores(self) -> dict[str, ScoreAnswer]:
        return {k: a for k, a in self.answers.items() if isinstance(a, ScoreAnswer)}


class ModelCard(BaseModel):
    name: str
    description: str
    release_date: str


class ModelList(BaseModel):
    models: list[ModelCard]
