# How to work with LocalDecision — a guide from zero

This guide assumes you have never used the project (or a terminal much) before. Follow the
steps in order; each one ends with a way to check that it worked. The technical background is in
[README.md](README.md) and [docs/ALGORITHM.md](docs/ALGORITHM.md), but you do not need it to start.

- [1. What the app does](#1-what-the-app-does)
- [2. What you need](#2-what-you-need)
- [3. Install the tools](#3-install-the-tools)
- [4. Get the code and install it](#4-get-the-code-and-install-it)
- [5. First run](#5-first-run)
- [6. Ask your first questions](#6-ask-your-first-questions)
- [7. Read the answer](#7-read-the-answer)
- [8. Write your own request](#8-write-your-own-request)
- [9. Use it as a web service](#9-use-it-as-a-web-service)
- [10. Use it from Python](#10-use-it-from-python)
- [11. Make the probabilities trustworthy (calibration)](#11-make-the-probabilities-trustworthy-calibration)
- [12. Measure the quality on your own data](#12-measure-the-quality-on-your-own-data)
- [13. Speed vs. accuracy](#13-speed-vs-accuracy)
- [14. Troubleshooting](#14-troubleshooting)
- [15. Glossary](#15-glossary)
- [16. Cheat sheet](#16-cheat-sheet)

---

## 1. What the app does

You give it:

1. a **state**: some text or data, e.g. a customer email, a JSON record, a news article;
2. one or more **questions** about that state, each with a fixed set of possible answers.

It gives back, for every question, **which answer** and **how likely each answer is**, for
example *"billing: 91%, technical: 5%, account: 2%, sales: 2%"*.

There are three kinds of questions:

| Kind | Use it for | Example | You get |
| --- | --- | --- | --- |
| `noul` | yes / no | "Is the customer asking for a refund?" | probability of *yes* (0 to 1) |
| `choice` | pick one of 2–255 options | "Which team should handle this?" | the chosen option + a probability per option |
| `score` | a level on an ordered scale (2–10 levels) | "How angry is the customer? calm / annoyed / furious" | an average level (e.g. 1.3) + a probability per level |

Everything runs **on your computer** with a free open model (Qwen3). Nothing is sent to the
internet except the one-time model download.

---

## 2. What you need

| Your computer | Works? | Notes |
| --- | --- | --- |
| Mac with Apple Silicon (M1, M2, M3, M4…), 16 GB RAM or more | ✅ recommended | Uses Apple's MLX; this is the tested setup |
| Mac with Apple Silicon, 8 GB RAM | ✅ with the small model | Use `mlx-community/Qwen3-0.6B-8bit` (see [section 13](#13-speed-vs-accuracy)) |
| Linux or Windows with an NVIDIA GPU | ⚠️ experimental | PyTorch backend; tested on macOS CPU/MPS so far, not yet on CUDA |
| Any computer, CPU only | ⚠️ slow | Works with the small model, a few seconds per question |
| Intel Mac | ⚠️ CPU only | Same as above |

Disk space: about **5 GB** for the default model (4.3 GB) plus the code and libraries,
or about **1 GB** with the small model.

To check your Mac: Apple menu → *About This Mac*. "Chip: Apple M…" means Apple Silicon.

---

## 3. Install the tools

You need three programs: a **terminal**, **git** (downloads the code) and **uv** (installs
Python and the libraries). You also need the **GitHub CLI** while the repository is private.

### macOS

Open the **Terminal** app (Spotlight: `⌘ + space`, type *Terminal*). Then copy each line,
paste it and press Enter:

```bash
xcode-select --install                                   # installs git (a window may pop up; accept)
curl -LsSf https://astral.sh/uv/install.sh | sh          # installs uv
```

Close the Terminal window and open a new one, so it picks up `uv`. For the GitHub CLI, install
[Homebrew](https://brew.sh) if you do not have it, then:

```bash
brew install gh
```

### Windows

Open **PowerShell** (Start menu → type *PowerShell*) and run:

```powershell
winget install --id Git.Git -e
winget install --id GitHub.cli -e
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close PowerShell and open it again.

### Linux

```bash
sudo apt install git          # or your distribution's package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
# GitHub CLI: https://github.com/cli/cli/blob/trunk/docs/install_linux.md
```

**Check:** these three commands should each print a version number:

```bash
git --version
uv --version
gh --version
```

---

## 4. Get the code and install it

### 4.1 Download the code

While the repository is **private**, you need a GitHub account that has been given access.
Log in once (it opens the browser):

```bash
gh auth login
```

Then download the project and enter its folder:

```bash
gh repo clone iamvinbas/localdecision
cd localdecision
```

(Once the repository is public, `git clone https://github.com/iamvinbas/localdecision` works
without logging in.)

### 4.2 Install the libraries

Pick **one** line, depending on your computer:

```bash
uv sync --extra mlx --extra server       # Mac with Apple Silicon
uv sync --extra torch --extra server     # Linux / Windows / Intel Mac
```

This creates a private folder `.venv` with Python and everything the app needs. It does not
touch the rest of your system. It takes a minute or two.

> **Windows + NVIDIA:** the PyTorch package installed by default on Windows is CPU-only. For
> the GPU, install a CUDA build afterwards, following [pytorch.org](https://pytorch.org/get-started/locally/),
> e.g. `uv pip install torch --index-url https://download.pytorch.org/whl/cu128`.

### 4.3 Activate the environment

Every time you open a new terminal to use the app, first go to the folder and activate it:

```bash
cd localdecision
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows (PowerShell)
```

Your prompt now starts with `(localdecision)`. From here on, the `localdecision` command is
available.

> **Important:** do not use a bare `uv run ...` to start commands. It re-syncs the environment
> without the `mlx`/`torch` extra and removes it. Either activate as above (recommended) or
> write `uv run --no-sync localdecision ...`.

**Check:**

```bash
localdecision --help
pytest -q          # runs the automatic tests, no model needed; ends with "passed"
```

---

## 5. First run

```bash
localdecision selftest
```

The **first time**, this downloads the model from Hugging Face (about 4.3 GB; it can take a
while). The files are stored in `~/.cache/huggingface` and reused afterwards. Then it checks that
the fast computation path gives the same numbers as the slow reference path, and prints a JSON
report. Look for:

```
loaded Qwen3-4B-Instruct-2507-8bit on mlx in 4.7s · self-test passed (shared-prefix)
```

`self-test passed` means everything works. If it says `FAILED`, the app still works but falls
back to a slower mode; see [troubleshooting](#14-troubleshooting).

---

## 6. Ask your first questions

The project ships an example request, [`examples/ticket.json`](examples/ticket.json): a customer
email and three questions (is it urgent? which team? how frustrated?).

```bash
localdecision ask examples/ticket.json
```

After a few seconds (the model loads every time you run `ask`) you get a JSON answer. To get
probabilities that mean something (see [section 11](#11-make-the-probabilities-trustworthy-calibration)),
add the calibration profile that comes with the repository:

```bash
localdecision ask examples/ticket.json --calibration results/public-Qwen3-4B-Instruct-2507-8bit/calibration.json
```

Try also the routing demo: five support tickets, each routed to a team or to a human.

```bash
python examples/support_router.py
```

---

## 7. Read the answer

Shortened answer for `examples/ticket.json`, with calibration:

```json
{
  "answers": {
    "is_urgent":   {"type": "noul", "noul": 0.917},
    "department":  {"type": "choice", "choice": "billing",
                    "probabilities": {"billing": 0.915, "technical": 0.045, "account": 0.020, "sales": 0.019},
                    "confidence": 0.887},
    "frustration": {"type": "score", "score": 1.31,
                    "probabilities": {"0": 0.127, "1": 0.436, "2": 0.436}, "confidence": 0.155}
  },
  "usage": {"input_tokens": 563, "output_tokens": 0},
  "diagnostics": {"department": {"views": 4, "agreement": 1.0, "prediction_set": ["billing"]}},
  "timing": {"total_ms": 1395.6}
}
```

How to read it:

| Field | Meaning |
| --- | --- |
| `noul` | Probability that the answer is *yes*. 0.917 → very likely urgent. |
| `choice` | The most likely option. |
| `probabilities` | The chance of every option (they add up to 1). |
| `score` | The average level. 1.31 = between "Frustrated but civil" (1) and "Very angry" (2). |
| `confidence` | 1 = all probability on one answer, 0 = completely unsure. |
| `diagnostics.views` | How many times the question was asked with the options in a different order. |
| `diagnostics.agreement` | Share of those orderings that gave the same answer. **Below 1.0 = the model is unsure.** |
| `diagnostics.prediction_set` | Answers that are still plausible (only with calibration). **One element = safe to act automatically.** |
| `usage.output_tokens` | Always 0: the model never writes text, it only scores the options. |
| `timing.total_ms` | Time spent on the request, in milliseconds. |

In the example the frustration question has 0.436 / 0.436 on levels 1 and 2: the model cannot
decide between "frustrated" and "very angry". That is useful information, not a bug: your code
can treat it as "not sure".

---

## 8. Write your own request

Create a file, e.g. `my-request.json`, next to `README.md`:

```json
{
  "state": "Hi, the blender I ordered last week arrived with a cracked jar. Can you send a new jar? I don't need a refund.",
  "questions": {
    "wants_refund": {
      "type": "noul",
      "instructions": "Does the customer ask for a refund?"
    },
    "next_step": {
      "type": "choice",
      "instructions": "What should the support agent do next?",
      "criteria": {
        "refund": "Give the money back",
        "replacement": "Ship a replacement part or item",
        "track_order": "Share tracking information",
        "not_stated": "The message does not say what the customer wants"
      }
    },
    "politeness": {
      "type": "score",
      "instructions": "How polite is the message?",
      "criteria": ["Rude", "Neutral", "Very polite"]
    }
  }
}
```

```bash
localdecision ask my-request.json
```

Rules:

- `state` can be text (`"..."`) or structured JSON (`{"order": {"id": 12, "status": "delivered"}}`).
- `noul`: only `instructions`. Optional: `"criteria": {"true": "what yes means", "false": "what no means"}`.
- `choice`: `criteria` maps each option name to a short description (or `null`). 2 to 255 options.
- `score`: `criteria` is a list of levels **from lowest to highest**. 2 to 10 levels.
- Question names (`wants_refund`, …) are yours; answers come back under the same names.

Tips for good answers:

1. **One thing per question.** Instead of "Is this a good candidate?", ask separately about
   experience, skills and location, and combine the answers in your code.
2. **Describe the options.** `"billing": "Payments, invoices, refunds"` beats a bare `"billing"`.
3. **Give an escape option** like `"not_stated"` when the state might not contain the answer.
4. **Write questions in English.** States can be in other languages (Italian works), but
   English questions work best.
5. **Do math in code.** Small models compare numbers and dates badly. Compute "days since
   delivery > 30" yourself and ask the model the part that needs judgment.

JSON must be valid: double quotes, no trailing commas. If you get an error, paste the file into
any online JSON validator.

---

## 9. Use it as a web service

Start the server (it keeps the model loaded, so answers are fast):

```bash
localdecision serve
```

Wait for `Uvicorn running on http://127.0.0.1:8080`. Leave this terminal open; stop the server
with `Ctrl + C`.

**From the browser:** open <http://127.0.0.1:8080/docs>, click `POST /v1/systemone` →
*Try it out*, paste a request (e.g. the content of `examples/ticket.json`) → *Execute*.

**From another terminal:**

```bash
curl -s http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d @examples/ticket.json
```

**From any program:** send an HTTP `POST` with the request JSON to `/v1/systemone`.
[`examples/http_client.py`](examples/http_client.py) does it with plain Python.

Useful options:

```bash
localdecision serve --port 9000                                  # another port
localdecision serve --calibration results/public-Qwen3-4B-Instruct-2507-8bit/calibration.json
LOCALDECISION_API_KEY=my-secret localdecision serve              # require "Authorization: Bearer my-secret"
```

Clients written for TypeSafe's hosted System One API work against this server by pointing
their base URL to it, e.g. `TYPESAFE_BASE_URL=http://127.0.0.1:8080 TYPESAFE_API_KEY=local`.

The server listens only on your own computer (`127.0.0.1`). Do not expose it to the internet
without a proper gateway in front.

---

## 10. Use it from Python

With the environment activated, start `python` (or use a script / Jupyter notebook):

```python
import localdecision as ld

engine = ld.load(calibration="results/public-Qwen3-4B-Instruct-2507-8bit/calibration.json")

result = engine.system_one(
    "The package arrived broken, I want my money back.",
    {
        "action": ld.Choice("What should support do?", {
            "refund": "Give the money back",
            "replacement": "Send a new item",
            "track_order": "Share tracking information",
        }),
        "angry": ld.Noul("Is the customer angry?"),
        "severity": ld.Score("How serious is the problem?", ["Minor", "Moderate", "Severe"]),
    },
)

print(result.answers["action"].choice)        # 'refund'
print(result.answers["action"].probabilities)
print(result.answers["angry"].noul)
print(result.diagnostics["action"].agreement)
```

Load the model **once** (`ld.load(...)` takes a few seconds) and reuse `engine` for all your
requests. A full example that routes tickets with confidence thresholds is
[`examples/support_router.py`](examples/support_router.py).

---

## 11. Make the probabilities trustworthy (calibration)

Out of the box the model is **over-confident**: it often says 100% even when it is wrong.
Calibration fixes what the numbers *mean*; it never changes which answer is chosen.

**Ready-made profile.** The repository includes
`results/public-Qwen3-4B-Instruct-2507-8bit/calibration.json`, fitted on public datasets for the
default model. Pass it with `--calibration ...` (CLI) or `calibration=...` (Python). It only fits
that exact model; with another model you get a warning.

**Your own profile (best).** Probabilities are most trustworthy when calibrated on decisions
from *your* domain:

1. Collect a few hundred real examples with the correct answer (format in
   [section 12](#12-measure-the-quality-on-your-own-data)).
2. Run them and fit the profile:

   ```bash
   localdecision eval my-calibration.jsonl --debias auto --output runs/cal
   localdecision calibrate runs/cal/records-auto.jsonl -o my-calibration-profile.json --alpha 0.05
   ```

   `--alpha 0.05` means: "the `prediction_set` may miss the correct answer at most 5% of the time".
3. Use it: `localdecision serve --calibration my-calibration-profile.json`.

A simple rule for automation: **act automatically when `prediction_set` has one element; ask a
person otherwise.**

---

## 12. Measure the quality on your own data

Try the built-in test set first (83 hand-written decisions, about one minute):

```bash
localdecision eval benchmarks/smoke-v1.jsonl --by-tag
```

The table shows, per question type and per topic: `acc` (share of correct answers), `nll` and
`brier` (lower is better: how good the probabilities are), `ece` (how far confidence is from
reality), `unstable` (how often the orderings disagreed) and `ms/dec` (milliseconds per decision).

To test on your data, write a `.jsonl` file: **one JSON object per line**, each with a state,
the questions and the correct answers under `labels`:

```json
{"id": "t1", "state": "My invoice is wrong, I was charged twice.", "questions": {"team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": null, "technical": null}}}, "labels": {"team": "billing"}}
{"id": "t2", "state": "The app crashes on startup.", "questions": {"team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": null, "technical": null}}}, "labels": {"team": "technical"}}
```

Labels: the option name for `choice`, the level number (0, 1, 2…) for `score`, `true`/`false`
for `noul`.

```bash
localdecision eval my-data.jsonl --output runs/my-data
localdecision report runs/my-data/records-auto.jsonl       # show the summary again later
```

Public datasets can be downloaded in the same format:

```bash
localdecision fetch agnews --limit 100 -o data/agnews.jsonl
localdecision eval data/agnews.jsonl
```

---

## 13. Speed vs. accuracy

| Knob | Faster | More accurate |
| --- | --- | --- |
| Model | `--model mlx-community/Qwen3-0.6B-8bit` (~0.6 GB, several times faster) | default 4B model |
| Views | `"settings": {"debias": "none"}` in the request (one ordering) | `"auto"` (default: several orderings) |
| Questions | fewer, shorter questions | more, well-described questions |
| State | only the relevant part of your data | — |

Measured on an M3 Pro with the 4B model: about 0.2–0.4 s per decision with one ordering, about
0.4–0.7 s with the default debiased orderings. Asking several questions about the **same** state
in **one** request is much cheaper than separate requests, because the state is processed once.

The small model is much less accurate: use it to try things or on low-memory machines, and
measure it on your data before relying on it.

---

## 14. Troubleshooting

| Problem | Fix |
| --- | --- |
| `command not found: localdecision` | Activate the environment: `source .venv/bin/activate` (Windows: `.venv\Scripts\activate`) from inside the project folder. |
| `No module named 'mlx'` / `'torch'` after `uv run` | A bare `uv run` removed the extra. Run `uv sync --extra mlx --extra server` again, then activate instead of using `uv run`. |
| The model download is slow or fails | Check the connection and retry; downloads resume. Optional: create a free Hugging Face token and `export HF_TOKEN=...` for higher rate limits. |
| Out of memory / the computer freezes | Use the small model; close other apps; lower `--kv-budget-gb 1` and `--batch-size 4`. |
| The first answer is slow | Normal: the model loads and the GPU warms up. Use `serve` to keep it loaded. |
| `422` error with a message | The request is invalid (e.g. a choice with 1 option, a score with 11 levels, empty state, text too long). The message says what to fix. Nothing is ever cut silently. |
| `401` error | The server was started with `LOCALDECISION_API_KEY`; send `Authorization: Bearer <key>`. |
| `address already in use` | Another program uses port 8080: `localdecision serve --port 9000`. |
| `self-test FAILED` | The fast path gave different numbers for this model; the app switched to a slower safe mode automatically. Results are still correct. |
| `warning: calibration profile was fitted with a different ...` | The profile belongs to another model or prompt version: fit a new one ([section 11](#11-make-the-probabilities-trustworthy-calibration)). |
| Answers look wrong | Rewrite the question in English, describe each option, split complex questions, add a `not_stated` option, move numbers/dates into code. |
| Free disk space used by models | Delete the model folder, e.g. `rm -rf ~/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-Instruct-2507-8bit`. It is downloaded again when needed. |

---

## 15. Glossary

- **State** — the input you want decisions about (text or JSON).
- **Question / primitive** — `noul` (yes/no), `choice` (pick one), `score` (level on a scale).
- **Model** — the open AI model doing the reading (Qwen3). Downloaded once, runs locally.
- **Token** — a piece of a word; models read text as tokens. Cost and speed grow with tokens.
- **Logits** — the model's raw scores for the next token; the app reads only the scores of the
  answer letters (A, B, C…) instead of letting the model write.
- **View** — the same question shown with the options in a different order. Several views are
  combined to cancel the model's preference for some positions.
- **Agreement** — how many views picked the same answer. Low agreement = unsure.
- **Calibration** — adjusting probabilities so that "80%" really means right 8 times out of 10.
- **Prediction set** — the answers still plausible after calibration; one element = confident.
- **Backend** — the engine that runs the model: MLX (Apple Silicon) or PyTorch (others).

---

## 16. Cheat sheet

```bash
# every new terminal
cd localdecision && source .venv/bin/activate

localdecision selftest                               # check the installation
localdecision ask examples/ticket.json               # one request from a file
echo '{"state":"...","questions":{...}}' | localdecision ask -      # one request from text
localdecision serve                                  # web service on http://127.0.0.1:8080 (docs at /docs)
localdecision eval benchmarks/smoke-v1.jsonl         # measure quality
localdecision fetch boolq -o data/boolq.jsonl        # download a public dataset
localdecision calibrate runs/x/records-auto.jsonl -o profile.json   # fit a calibration profile
localdecision report runs/x/records-auto.jsonl --calibration profile.json   # re-score saved results
python examples/support_router.py                    # routing demo

# flags that work on ask / serve / eval / selftest
--model mlx-community/Qwen3-0.6B-8bit     # small, fast model
--calibration path/to/profile.json        # calibrated probabilities ('none' = raw)
--backend torch --device cpu              # force PyTorch (e.g. no Apple Silicon)
```
