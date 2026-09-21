# Training a small decision model

> **Status:** the training-set builder is ready and tested; the trainer is not written yet.
> Training needs a few GPU hours, so it is meant for a free cloud GPU (Kaggle, Colab) rather
> than a laptop.

Goal: specialize a small open model (1–2B parameters, default base **Qwen3-1.7B**) so that it
makes typed decisions far better than the same model zero-shot, with probabilities that are
calibrated out of the box and no position bias, small enough for an ordinary home PC.

The model is trained on exactly what it does at inference time: the LocalDecision prompt
(state, question, lettered options) and the probability of the answer letter. It never learns
to write text.

## 1. Build the data

```bash
python training/build_dataset.py --out data/train            # ~120k decisions, a few minutes
python training/build_dataset.py --out data/train-smoke --per-source 0.01   # quick check
```

Output (in `data/`, not committed):

| File | Content |
| --- | --- |
| `decisions-v1.train.jsonl` | training decisions, eval-JSONL format (state, questions, labels, tags) |
| `decisions-v1.val.jsonl` | 2% held out to monitor training |
| `decisions-v1.stats.json` | counts per question type and task family, sources and licenses |

Every row is validated with the same schema and label mapping the engine uses, so a row that
would be rejected at inference time never reaches training. Option order is **not** fixed in the
data: the trainer shuffles it every time an example is seen.

## 2. Sources

Only permissively licensed data is used, so trained weights can be redistributed.

| Source | Family | Primitives | License |
| --- | --- | --- | --- |
| [MultiNLI](https://huggingface.co/datasets/nyu-mll/multi_nli) | claim vs. evidence | noul, choice (supported / contradicted / not stated), score | CC-BY-3.0 / CC-BY-SA-3.0 / MIT (per genre) |
| [SNLI](https://huggingface.co/datasets/stanfordnlp/snli) | claim vs. evidence | same | CC-BY-SA-4.0 |
| [WANLI](https://huggingface.co/datasets/alisawuffles/WANLI) | claim vs. evidence (hard cases) | same | CC-BY-4.0 |
| [BoolQ](https://huggingface.co/datasets/google/boolq) (train) | reading comprehension | noul | CC-BY-SA-3.0 |
| [ARC](https://huggingface.co/datasets/allenai/ai2_arc) Easy + Challenge (train) | science questions | choice | CC-BY-SA-4.0 |
| [CommonsenseQA](https://huggingface.co/datasets/tau/commonsense_qa) (train) | common sense | choice (5) | MIT |
| [DBpedia-14](https://huggingface.co/datasets/fancyzhx/dbpedia_14) | topic | choice (3–14 options, 12% "none of these") | CC-BY-SA-3.0 |
| [Amazon polarity](https://huggingface.co/datasets/fancyzhx/amazon_polarity) | review sentiment | noul, choice | Apache-2.0 |
| [Civil Comments](https://huggingface.co/datasets/google/civil_comments) | moderation | score (4 toxicity levels), noul | CC0-1.0 |
| synthetic JSON records | structured state | field lookup with "not stated", equality, flags, numeric thresholds, bands | MIT (this repo) |

Excluded on purpose: datasets with non-commercial or unknown licenses (e.g. ANLI, RACE, SciQ,
Yelp, AG News, SST-5).

## 3. What stays out of training

| Set | Why |
| --- | --- |
| BoolQ validation, ARC-Challenge test | in-domain test (their *train* splits are used) |
| **AG News** | topic classification on news: trained only on encyclopedia topics (DBpedia) |
| **SST-5** | 5-level sentiment: never trained on graded sentiment |
| **Banking77** | intent classification: family never seen |
| **benchmarks/smoke-v1.jsonl** | hand-written, mixed |

The last four measure whether the model learned to *decide*, not just to solve the training
datasets.

## 4. Next: the trainer (planned)

- Base model: Qwen3-1.7B (Apache-2.0), LoRA on the attention and MLP projections.
- Each step renders an example with the LocalDecision `Prompter` and a fresh random option
  order, so the model sees exactly the inference-time prompt and cannot learn position shortcuts.
- Loss on the answer letters only: cross-entropy over the displayed letters plus a Brier term,
  so the probabilities come out calibrated.
- Early stopping on `decisions-v1.val.jsonl`; evaluation with `localdecision eval` on the sets
  listed above, against the zero-shot baseline in the main README.
- Output: merged weights quantized to 8 bit, published with a model card that credits the base
  model as its license requires.
