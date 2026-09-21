<div align="center">

# LocalDecision

**Typed, calibrated decisions from open-weight LLMs.<br/>One forward pass, zero generated tokens, entirely on your machine.**

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Backends](https://img.shields.io/badge/backends-MLX%20%C2%B7%20PyTorch-purple)](#install)
[![Output tokens](https://img.shields.io/badge/output%20tokens-0-black)](#how-it-works)

</div>

> **New here?** [howToWorkApp.md](howToWorkApp.md) walks you from an empty machine to your first
> decisions, step by step.

Most automation needs a **decision**, not a paragraph: *which queue*, *is this urgent*, *how
severe*. Asking a chat model for that means generating text, hoping it is valid JSON, and
parsing it back into an `if` statement, with no honest measure of how sure the model was.

LocalDecision turns any instruction-tuned open model into a **decision function**. You send a
*state* and typed *questions*; you get typed *answers* with a full probability distribution,
read directly from the model's next-token logits. Nothing is sampled, so an answer can never be
malformed, and every answer comes with the numbers your code needs to decide whether to act.

```python
import localdecision as ld

engine = ld.load()  # Qwen3-1.7B on MLX (Apple Silicon) or PyTorch (CUDA / MPS / CPU)

result = engine.system_one(
    "Help! My payouts have been failing for 3 days and our suppliers are waiting.",
    {
        "urgent": ld.Noul("Does this convey urgency?"),
        "team": ld.Choice("Which team should handle this?", {
            "billing": "Payments, payouts, refunds",
            "technical": "Bugs, outages",
            "sales": None,
        }),
        "anger": ld.Score("How frustrated is the customer?", ["Calm", "Frustrated", "Very angry"]),
    },
)

result.answers["team"].choice          # 'billing'
result.answers["team"].probabilities   # {'billing': 0.99.., 'technical': 0.00.., 'sales': 0.0..}
result.answers["urgent"].noul          # 0.99..  (probability of "yes")
result.answers["anger"].score          # 1.0.. (expected level, can land between levels)
result.diagnostics["team"].agreement   # 1.0   (all option orderings agree)
```

## Highlights

- **Three typed primitives.** `noul` (yes/no → P(yes)), `choice` (2–255 options → argmax +
  distribution + confidence), `score` (ordered rubric → expected level + distribution).
- **Zero generation.** One forward pass per view; only the answer letters' logits are read.
  Output is typed by construction: no JSON repair, no refusals, no hallucinated labels.
- **Prefix tree execution.** The system prompt is cached once per process, the state once per
  request, each question header once per question; option views run as right-padded batches on
  replicated KV caches. Questions stay independent: adding one never changes another's answer.
- **Position-debiased views.** Options are shown in cyclic orders and pooled by geometric mean,
  which cancels a multiplicative position preference *exactly*. View disagreement is exposed as
  a label-free instability signal.
- **Up to 255 options.** Large choices run as a tournament whose rounds are merged with Luce's
  choice axiom, reproducing the single-shot distribution exactly for a consistent model.
- **Calibration you can trust.** Per-primitive temperature scaling plus split-conformal
  prediction sets with a finite-sample coverage guarantee: a singleton set is a principled
  "act automatically" rule.
- **Drop-in HTTP API.** `POST /v1/systemone` speaks the same wire format as TypeSafe's hosted
  System One API; the official `typesafe-sdk` works against a local server by changing only its
  base URL (verified with `typesafe-sdk` 0.7.0).
- **Two backends, verified at load.** MLX for Apple Silicon, PyTorch for CUDA/MPS/CPU. A
  self-test checks the fast path against a plain forward pass and falls back automatically.
- **Evaluation built in.** Accuracy, NLL, Brier, ECE, risk–coverage, conformal coverage, latency;
  adapters for public datasets; a hand-written smoke set.

## How it works

```mermaid
flowchart LR
    H["chat header + system prompt<br/><i>cached once per process</i>"] --> S["state<br/><i>once per request</i>"]
    S --> Q1["question 1 header"]
    S --> Q2["question 2 header"]
    Q1 --> V1["options A·B·C"] & V2["options B·C·A"] & V3["options C·A·B"]
    Q2 --> W1["Yes·No"] & W2["No·Yes"]
    V1 & V2 & V3 --> P1["letter logits → geometric pooling<br/>(position bias cancels)"]
    W1 & W2 --> P2["letter logits → pooling"]
    P1 & P2 --> C["calibration<br/>temperature · conformal set"] --> A["typed answers<br/>+ diagnostics"]
```

1. **Every question becomes a multiple-choice prompt** whose answer is one letter token. The
   tokenizer is checked at load time: each letter must be a single token in that position, and
   prefix + suffix tokenization must be exactly additive, so cached prefixes are bit-for-bit the
   tokens a full prompt would produce.
2. **The prompt is executed as a tree.** Shared nodes are computed once; leaves (option views)
   are batched against replicated KV caches.
3. **Only the answer letters are read.** The final hidden state of each view goes through the
   LM head; a softmax over the letters on screen gives that view's distribution.
4. **Views are pooled, tournaments merged, calibration applied**, and the result is rendered as
   a typed answer with diagnostics (`agreement`, `margin`, `entropy_confidence`, `format_mass`,
   `prediction_set`) and timings.

The math, the proofs of exactness and the references are in [docs/ALGORITHM.md](docs/ALGORITHM.md).

## Install

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/) (or pip).

```bash
git clone https://github.com/iamvinbas/localdecision
cd localdecision

# Apple Silicon (Metal)
uv sync --extra mlx --extra server

# NVIDIA GPU, Apple MPS or CPU
uv sync --extra torch --extra server
```

The first run downloads the default model from Hugging Face: `mlx-community/Qwen3-1.7B-8bit`
(~1.8 GB) on MLX, `Qwen/Qwen3-1.7B` (~4 GB) on PyTorch. It fits an ordinary laptop with 8 GB of
RAM. Any instruction-tuned causal LM with a chat template works: pass `--model <repo or path>`;
`mlx-community/Qwen3-0.6B-8bit` (~0.6 GB) is even lighter. A decision-tuned 1.7B model is the
next step on the [roadmap](#roadmap); its training set builder is ready in [training/](training/README.md).

```bash
uv run localdecision selftest          # loads the model and prints the self-test result
```

## Usage

### Python

```python
import localdecision as ld

engine = ld.load("mlx-community/Qwen3-1.7B-8bit", calibration="calibration.json")
r = engine.system_one(state, questions, debias="auto")   # settings are optional
```

`state` can be a string, a JSON object or an array. `instructions` and option descriptions can
also be structured JSON (e.g. `{"policy": "...", "question": "Is this eligible under `policy`?"}`).

### HTTP

```bash
uv run localdecision serve --port 8080            # LOCALDECISION_API_KEY=... enables Bearer auth

curl -s localhost:8080/v1/systemone -H 'Content-Type: application/json' -d @examples/ticket.json
```

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/systemone` | state + questions → answers (+ `diagnostics`, `timing`) |
| `GET /v1/models` | served model and the `localdecision-latest` alias |
| `GET /health` | backend, device, self-test, calibration |
| `GET /docs` | interactive OpenAPI documentation |

Request and answer shapes:

```jsonc
// request
{
  "state": "text | object | array",
  "model": "localdecision-latest",               // any name is accepted; the response reports the real one
  "questions": {
    "is_urgent":  {"type": "noul",   "instructions": "Does this convey urgency?",
                   "criteria": {"true": "Time-sensitive", "false": "Routine"}},   // optional
    "department": {"type": "choice", "instructions": "Which team?",
                   "criteria": {"billing": "Payments", "technical": "Bugs", "sales": null}},
    "frustration":{"type": "score",  "instructions": "How frustrated?",
                   "criteria": ["Calm", "Frustrated", "Very angry"]}             // low -> high
  },
  "settings": {"debias": "auto", "calibrated": true, "diagnostics": true}        // optional extension
}

// response for examples/ticket.json (Qwen3-1.7B 8-bit, raw probabilities, MacBook Pro M3 Pro)
{
  "model": "Qwen3-1.7B-8bit",
  "answers": {
    "is_urgent":   {"type": "noul", "noul": 1.0},
    "department":  {"type": "choice", "choice": "billing",
                    "probabilities": {"billing": 0.999952, "technical": 4.8e-05, "account": 0.0, "sales": 0.0},
                    "confidence": 0.999936},
    "frustration": {"type": "score", "score": 1.0,
                    "legend": {"0": "Calm, just stating facts", "1": "Frustrated but civil", "2": "Very angry, threatening to leave"},
                    "probabilities": {"0": 0.0, "1": 0.999999, "2": 1e-06}, "confidence": 0.999998}
  },
  "usage": {"input_tokens": 595, "output_tokens": 0},
  "diagnostics": {                                  // one entry per question; one shown
    "department": {"views": 4, "agreement": 0.75, "margin": 0.999903, "entropy_confidence": 0.999619,
                   "format_mass": 0.980255, "temperature": 1.0, "stages": 1}
  },
  "timing": {"total_ms": 580.67, "prefill_ms": 134.86, "readout_ms": 394.37, "cached_tokens": 48,
             "prefix_tokens": 101, "suffix_tokens": 446, "probes": 8, "batches": 3}
}
```

`settings.debias`: `none` (1 view), `swap` (original + reversed), `cyclic` (every cyclic shift,
capped by `max_views`), `auto` (cyclic for ≤ 4 options, swap otherwise). Invalid requests (fewer
than 2 options, more than 255 options or 10 levels, empty state, over the context limit) return
`422` with the reason; nothing is ever truncated silently.

**Existing System One clients** work unchanged:

```bash
TYPESAFE_BASE_URL=http://127.0.0.1:8080 TYPESAFE_API_KEY=local python your_app.py
```

### Command line

| Command | What it does |
| --- | --- |
| `localdecision serve` | run the HTTP API |
| `localdecision ask request.json` | answer one request from a file (or `-` for stdin) |
| `localdecision eval data.jsonl --debias none,auto` | score a labelled dataset, compare modes |
| `localdecision fetch boolq -o data/boolq.jsonl` | export a public benchmark (`boolq`, `sst5`, `agnews`, `arc`, `banking77`) |
| `localdecision calibrate records.jsonl -o calibration.json` | fit temperature + conformal thresholds |
| `localdecision report records.jsonl --calibration calibration.json` | re-score saved records offline |
| `localdecision selftest` | load a model and check the shared-prefix path |

Common flags: `--backend auto|mlx|torch`, `--model`, `--batch-size`, `--max-context`,
`--kv-budget-gb`, `--bits 4|8` (MLX), `--device`, `--dtype` (PyTorch), `--calibration`.

## Results

Measured on a **MacBook Pro M3 Pro (18 GB)** with `mlx-community/Qwen3-1.7B-8bit` on MLX 0.32,
prompt `ld-prompt-v1`, **zero-shot**: this is the starting point that the decision-tuned model
([training/](training/README.md)) has to beat. Per-decision records are in [`results/`](results)
and can be re-scored with `localdecision report`. Latency is wall-clock per decision, warm model.

| Set | Primitive | n | Accuracy, 1 view | Accuracy, debiased | ms / decision, 1 view | ms / decision, debiased |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Smoke set ([hand-written](benchmarks/README.md)) | mixed | 83 | 0.867 | 0.867 | 90 | 178 |
| BoolQ | noul | 200 | 0.725 | 0.755 | 163 | 212 |
| AG News | choice (4) | 200 | 0.875 | 0.875 | 134 | 309 |
| SST-5 | score (5 levels) | 200 | 0.330 | 0.350 | 114 | 190 |

What the baseline shows:

- **Fast enough for real-time use**: about 0.1–0.2 s per decision with one view on a laptop.
- **Debiased views** add up to 3 points (BoolQ, SST-5) at roughly twice the cost.
- **Raw probabilities are badly over-confident** (NLL 6.4 on BoolQ and 10.3 on SST-5 with one
  view): the model says ~100% even when it is wrong. `diagnostics.agreement` still exposes many
  of those cases (see the example above: 100% "billing", but one ordering out of four disagrees).
- **Fine-grained scales are hard** for a small model (SST-5).

Calibration ([below](#calibrating-on-your-own-data)) fixes the meaning of the probabilities
today; decision-tuning the model itself is the next step on the [roadmap](#roadmap).

## Calibrating on your own data

Raw probabilities from instruction-tuned models are sharply over-confident. Before you put
thresholds on them, fit a profile on labelled decisions from your domain:

```bash
# 1. label a few hundred real decisions in the eval JSONL format (see benchmarks/README.md)
localdecision eval my-calibration.jsonl --debias auto --output runs/cal
# 2. fit temperature + conformal thresholds (alpha = tolerated miss rate of the sets)
localdecision calibrate runs/cal/records-auto.jsonl -o calibration.json --alpha 0.05
# 3. check on a separate labelled set, then serve with the profile
localdecision eval my-test.jsonl --calibration calibration.json
localdecision serve --calibration calibration.json
```

With a profile loaded, every answer carries `diagnostics.prediction_set`. A reasonable policy:
act automatically when the set has one element, ask a human (or a bigger model) otherwise, and
choose `alpha` from the cost of a wrong automatic action. The profile stores a fingerprint of
the model, backend and prompt version; a mismatch is reported at start-up.

## Design notes

- **Keep code in control.** Ask narrow, atomic questions and combine them in code
  ([examples/support_router.py](examples/support_router.py)); change a weight in Python rather
  than a prompt.
- **Numbers belong in code.** Small models compare numbers and dates unreliably (see the
  `numeric` tag above). Compute them and ask the model the semantic part.
- **One resident model.** Requests are serialized on one inference thread; parallelism lives
  inside a request, where it is free.

## Limitations

- Quality is bounded by the model. A 1.7B model is fast and fine for common-sense judgments,
  weaker at fine-grained scales, arithmetic, dates and multi-hop reasoning.
- Uncalibrated probabilities are over-confident; `confidence` is a property of the
  distribution's shape, not a probability of being right, until you calibrate.
- Cyclic views remove *positional* bias, not bias towards particular option *content*.
- English prompts work best; other languages work but are less accurate.
- Localhost by default; no rate limiting. Put it behind your own gateway before exposing it.

## Roadmap

- **Decision-tuned small model.** Fine-tune a 1–2B model (LoRA) on the LocalDecision prompt,
  with a loss on the answer letters only (cross-entropy + Brier) and options shuffled at every
  step, so it is calibrated and position-invariant by design. The training set is ready:
  `training/build_dataset.py` builds 116k decisions from permissively licensed sources, with
  AG News, SST-5, Banking77 and the smoke set kept out for testing. Training fits a free cloud
  GPU (Kaggle / Colab); the trainer is not written yet.
- Cross-request prefix cache (radix tree) for repeated states, e.g. game loops and agents.
- Multi-token option labels to lift the 26-letters-per-view limit without a tournament.
- Contextual (content-free) prior correction as an optional second debiasing step.
- vLLM / llama.cpp backends; streaming of answers as each question completes.
- A small playground page served by `localdecision serve`.

## Project layout

```
src/localdecision/
  schema.py          wire format (pydantic): questions, answers, settings, diagnostics
  prompts.py         prompt layout, tokenizer checks (single-token letters, additive split)
  engine.py          planning, prefix tree execution, tournaments, answers
  readout.py         pure math: pooling, Luce chaining, confidence statistics
  calibration.py     temperature scaling, split-conformal thresholds
  metrics.py         accuracy, NLL, Brier, ECE, risk-coverage
  evaluation.py      eval harness and reports
  public_data.py     public benchmark adapters
  server.py, cli.py  FastAPI server and command line
  backends/          mlx_backend.py, torch_backend.py, mock.py (deterministic, for tests)
docs/ALGORITHM.md    the method in detail, with references
benchmarks/          smoke set, public benchmark runner, prefix-reuse benchmark
examples/            request JSON, confidence-gated router, stdlib HTTP client
tests/               unit tests (no weights needed) + optional real-model tests
```

## Development

```bash
uv sync --extra mlx --extra server --extra dev
uv run pytest -q                        # unit tests, no model weights needed
LOCALDECISION_TEST_MODEL=mlx-community/Qwen3-0.6B-8bit uv run pytest tests/test_model.py
uv run ruff check src tests && uv run ruff format --check src tests
```

## License

[MIT](LICENSE) © 2026 Vincenzo Basile. Model weights are downloaded from their publishers and
keep their own licenses (Qwen3: Apache-2.0).

<sub>TypeSafe and Jev are trademarks of their respective owners. LocalDecision is an independent
project, not affiliated with or endorsed by them; the `/v1/systemone` compatibility refers only to
the public HTTP request/response format.</sub>
