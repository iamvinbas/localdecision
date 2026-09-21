# Benchmarks

Two kinds of data, used for different purposes.

## `smoke-v1.jsonl` — hand-written decision set

44 states and 83 labelled decisions (35 choice, 33 noul, 15 score) written for this project.
They cover the situations a decision layer meets in practice:

| Tag | What it probes |
| --- | --- |
| `support` | ticket routing, urgency, customer frustration |
| `ecommerce`, `json` | next best action and facts read from structured (JSON) state |
| `facts` | yes/no facts about a record |
| `missing-evidence` | the right answer is an explicit "not stated" option |
| `negation` | "I do **not** want a refund" |
| `italian` | non-English input with English questions |
| `cv` | ordinal fit of a candidate to a job (score) |
| `moderation` | allow / review / remove, threat detection |
| `smart-home` | intent classification |
| `rules` | applying a policy passed as structured instructions |
| `numeric` | threshold comparisons on numbers (a known weak spot) |

It is small and its labels are the author's: treat it as a **smoke test** that catches
regressions and shows failure modes, not as a benchmark to compare models on. A few labels are
deliberately borderline (e.g. whether a polite message about a double charge is "calm").

```bash
localdecision eval benchmarks/smoke-v1.jsonl --debias none,auto --by-tag --output results/local-smoke
```

## Public datasets

`localdecision fetch` converts public datasets into the same JSONL format, one primitive each:

| Name | Primitive | Source |
| --- | --- | --- |
| `boolq` | noul | [google/boolq](https://huggingface.co/datasets/google/boolq) validation — passage + yes/no question |
| `sst5` | score (5 levels) | [SetFit/sst5](https://huggingface.co/datasets/SetFit/sst5) validation — review sentiment |
| `agnews` | choice (4) | [fancyzhx/ag_news](https://huggingface.co/datasets/fancyzhx/ag_news) test — news topic |
| `arc` | choice (3–5) | [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) ARC-Challenge test |
| `banking77` | choice (77, tournament) | [legacy-datasets/banking77](https://huggingface.co/datasets/legacy-datasets/banking77) test — intent |

Protocol (`run_public.sh`): each dataset is shuffled with seed 0. Rows `[0, N)` are the **test**
window; rows `[N, 2N)` are a disjoint **calibration** window used only to fit the temperature and
conformal thresholds. Every test number is computed once on the test window; nothing is tuned on it.

```bash
bash benchmarks/run_public.sh                                   # N=200, default model
MODEL=mlx-community/Qwen3-0.6B-8bit N=100 bash benchmarks/run_public.sh
```

Outputs per dataset: `records-<mode>.jsonl` (one line per decision, with raw pooled
log-probabilities, so any metric can be recomputed offline with `localdecision report`) and
`summary.json` (engine fingerprint, self-test result, full report).
