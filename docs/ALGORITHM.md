# How LocalDecision decides

This document describes the decision algorithm end to end: what is computed, why it is
correct, and what it costs. Code references point to `src/localdecision/`.

## 1. The contract

A request is a **state** *s* (text or JSON) and a set of **questions**. Each question *q* has a
closed answer set *C = {c₁ … cₙ}* defined by the caller:

| Primitive | Answer set | Returned value |
| --- | --- | --- |
| `noul` | {yes, no} | P(yes) |
| `choice` | caller's options (2–255) | argmax, full distribution, confidence |
| `score` | ordered levels 0 … n−1 (2–10) | E[level], full distribution, confidence |

The output is a probability distribution *p(c | s, q)* over *C*. Nothing else can come out of
the model: the answer is typed **by construction**, not by parsing.

## 2. Reading a decision instead of generating it

Each question is rendered as a multiple-choice prompt whose options are labelled with single
letters `A, B, C, …`. The prompt ends exactly where the assistant's first token would go.

For one rendering (a *view* *v*) the model produces next-token logits *z*. We keep only the
logits of the letters on screen and normalize over them:

```
p_v(c) = exp(z[letter_v(c)]) / Σ_{c'} exp(z[letter_v(c')])
```

One forward pass, zero sampled tokens. Two facts are verified when a model is loaded
(`prompts.Prompter`):

* every letter is **one token** in that position (`encode(prompt + "A") == encode(prompt) + [id_A]`);
* the prompt splits into prefix and suffix **additively** (`encode(p) + encode(s) == encode(p + s)`),
  so cached prefixes are exactly the tokens a full prompt would have produced.

The share of the full vocabulary distribution that lands on the allowed letters is reported as
`format_mass`: a cheap signal that the model understood the question format.

## 3. A prefix tree over the prompt

Every view of every question in a request shares most of its tokens. The prompt is laid out so
the shared parts come first, and the backend computes each shared node once:

```
[chat header + system prompt]            level 0: cached once per process
  └─ [<state> … </state>]                level 1: once per request
       ├─ [Question 1 + first options]   level 2: once per question
       │    ├─ view 1.1 (option order a) leaves: batched together
       │    └─ view 1.2 (option order b)
       └─ [Question 2 …]
            └─ …
```

With header *H*, state *S*, question header *hq* and per-view tails *o(q,v)*, the tokens run
through the model are

```
cost = |S| + Σ_q ( |hq| + Σ_v |o(q,v)| )        (+ |H| once per process)
```

instead of `Σ_q Σ_v (|H| + |S| + |hq| + |o(q,v)|)` for independent calls. The saving grows with
the size of the state and the number of questions and views (`engine._SessionRunner`).

**Batching leaves.** A node's key/value cache is replicated *b* times along the batch axis and
*b* right-padded suffixes run in one forward pass. Right padding is exact for any causal model:
a real token never attends to padding that comes after it. The readout gathers the hidden state
at each suffix's last real token and multiplies only those rows by the LM head. The batch size
adapts to the prefix size so replicated caches stay within `kv_budget_gb`.

**Self-test.** At load time the backend scores a probe prompt through the fast path and through
a plain full forward pass, and compares the log-probabilities of a few tokens
(`backends.base.Backend.run_selftest`). If they disagree beyond 0.05 nats the engine falls back,
in order, to full-logit readout and then to direct (non-shared) scoring. The result is exposed
at `GET /health`.

Questions are independent by construction: no question can see another question or its answer,
so adding questions cannot change existing answers (no context rot between questions).

## 4. Removing position bias with cyclic views

Language models prefer some answer positions and letters regardless of content (Zheng et al.,
2024; Pezeshkpour & Hruschka, 2024). Model the preference as multiplicative:

```
p_v(c) ∝ p(c) · π(pos_v(c))
```

where *p* is the content-based preference we want and *π* a positional prior. Show the options in
all *n* cyclic shifts, so every candidate occupies every position exactly once, and pool the
views by their **geometric mean**:

```
log p̂(c) = (1/n) Σ_v log p_v(c) + const
         = log p(c) + (1/n) Σ_pos log π(pos) + const
```

The positional term is the same for every candidate, so it disappears after renormalization:
**the pooled distribution equals p exactly**, whatever π is (`readout.pool_views`, tested in
`tests/test_readout.py`). For two options the only non-trivial shift is the reversal, so `swap`
is also exact for `noul`.

Default policy (`debias="auto"`): all cyclic shifts for choices with ≤ 4 options, original +
reversed order otherwise (scores keep a monotone scale on screen). Views share the question
header in the prefix tree, so extra views cost only their option lines.

**Agreement.** Each view also casts a vote (its own argmax). `diagnostics.agreement` is the share
of views that agree with the pooled answer. Disagreement means the answer depends on how the
options were displayed: a label-free instability signal that is useful for routing to a human.

## 5. Large choices: a tournament merged with Luce's axiom

Single-letter slots cap a view at 26 options. A choice with *n* > 26 options runs in two stages:

1. **Chunks.** Options are split into *m* = ⌈n/20⌉ balanced chunks; each chunk is a normal
   question (with swap views), giving within-chunk distributions *q_k*.
2. **Final.** The top *k* of each chunk (k chosen so the final round fits 26 letters, at most 4)
   meet in a final question, giving *r* over the finalists.

The two stages are merged into one distribution over all *n* options using **Luce's choice
axiom**: if every option has a weight *w(c)* with *P(c | T) = w(c) / Σ_{T} w* for any subset *T*,
then ratios inside a chunk come from stage 1 and ratios across chunks from stage 2. Anchoring
each chunk on its best finalist *a*:

```
log w(c) = log r(a) + log q_k(c) − log q_k(a)      for non-finalists c in chunk k
log w(c) = log r(c)                                 for finalists
```

For a model that satisfies the axiom this reproduces the single-shot softmax over all *n*
options **exactly** (`readout.luce_chain`, verified to 1e-12 in the tests). The cost is
*O(n/20 + 1)* views instead of *O(n)* independent yes/no probes.

## 6. From distributions to typed answers

* `noul`: *P(yes)*, pooled over the two orders.
* `choice`: argmax plus the full distribution in the caller's key order.
* `score`: `score = Σ_i i · p(i)` (lands between levels), plus the distribution.
* `confidence = (n · p_max − 1) / (n − 1)`: 1 when all mass is on one answer, 0 when uniform.
  Diagnostics also report the top-2 margin and 1 − H(p)/log n.

## 7. Calibration you can fit on your own data

Instruction-tuned models are usually **over-confident**: they put ~all probability on their
first choice even when wrong. Two post-hoc tools fix what confidence *means*, per primitive
(`calibration.py`):

**Temperature scaling** (Guo et al., 2017). `p_T ∝ exp(log p̂ / T)` with *T* fitted by
minimizing negative log-likelihood on labelled decisions. The NLL is convex in β = 1/T, so a
golden-section search finds the global optimum. The argmax never changes.

**Split-conformal prediction sets** (Vovk et al., 2005; Sadinle et al., 2019; Angelopoulos &
Bates, 2023). On a held-out calibration set compute nonconformity scores *sᵢ = 1 − p(yᵢ)* and
take q̂ as the ⌈(N+1)(1−α)⌉-th smallest. At inference

```
C(x) = { c : p(c) ≥ 1 − q̂ }
```

contains the true answer with probability ≥ 1 − α (marginally, for exchangeable data). A
**singleton set** is a principled "act automatically" rule with a known error budget α; larger
sets say exactly which options remain plausible. `localdecision calibrate` fits both on a
different window of data from the one you evaluate on.

## 8. What this is not

* It is not a new model and involves no training: any instruction-tuned causal LM with a chat
  template works. Quality is bounded by the base model.
* Raw probabilities are the model's, not calibrated estimates. Calibrate on labelled data from
  your domain before using thresholds.
* Views remove *positional* bias, not bias toward particular option *content*.
* Numbers, dates and counting are weak spots of small models: compute them in code and ask the
  model the semantic part.

## References

* Zhao, Wallace, Feng, Klein, Singh. *Calibrate Before Use: Improving Few-Shot Performance of
  Language Models.* ICML 2021.
* Zheng, Zhou, Meng, Zhou, Huang. *Large Language Models Are Not Robust Multiple Choice
  Selectors.* ICLR 2024.
* Pezeshkpour, Hruschka. *Large Language Models Sensitivity to The Order of Options in
  Multiple-Choice Questions.* Findings of NAACL 2024.
* Guo, Pleiss, Sun, Weinberger. *On Calibration of Modern Neural Networks.* ICML 2017.
* Kadavath et al. *Language Models (Mostly) Know What They Know.* 2022.
* Luce. *Individual Choice Behavior: A Theoretical Analysis.* Wiley, 1959.
* Vovk, Gammerman, Shafer. *Algorithmic Learning in a Random World.* Springer, 2005.
* Sadinle, Lei, Wasserman. *Least Ambiguous Set-Valued Classifiers with Bounded Error Levels.*
  JASA 2019.
* Angelopoulos, Bates. *Conformal Prediction: A Gentle Introduction.* Foundations and Trends in
  Machine Learning, 2023.
* Kumar et al. *Conformal Prediction with Large Language Models for Multi-Choice Question
  Answering.* 2023.
* Juravsky et al. *Hydragen: High-Throughput LLM Inference with Shared Prefixes.* 2024.
* Zheng et al. *SGLang: Efficient Execution of Structured Language Model Programs* (RadixAttention).
  NeurIPS 2024.
