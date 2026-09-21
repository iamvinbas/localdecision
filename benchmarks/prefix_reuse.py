"""How much the prefix tree saves: one realistic request, three execution strategies.

    python benchmarks/prefix_reuse.py [--model MODEL] [--runs 5]

* tree   - default: header cached per process, state once, question headers once
* flat   - state once, but every view repeats its full question block
* direct - no reuse at all: every view is a full prompt (like one LLM call per view)

All three must give the same answers; only the work changes.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import localdecision as ld
from localdecision import engine as engine_module

STATE = {
    "ticket": {
        "id": "T-88213",
        "channel": "email",
        "subject": "Payouts failing since Monday - suppliers unpaid",
        "customer": {"plan": "Business", "seats": 42, "tenure_months": 26, "region": "EU"},
    },
    "thread": [
        {
            "from": "customer",
            "text": "Hi, since Monday every payout to our suppliers fails with error PX-203. "
            "We have 14 suppliers waiting and two of them threatened to stop deliveries. We changed nothing "
            "on our side. Can you check what is going on? This is really urgent for us.",
        },
        {
            "from": "agent",
            "text": "Thanks for reaching out. Could you confirm the bank account the payouts "
            "should go to and whether you recently changed your verification documents?",
        },
        {
            "from": "customer",
            "text": "Same account as always, IBAN ending 4471. We uploaded a new company "
            "registration certificate last Friday because the old one expired. Third email now, "
            "honestly I'm losing patience. If this is not fixed by tomorrow we will look at other providers.",
        },
    ],
    "account_flags": {"kyc_status": "under_review", "payouts_paused": True, "open_invoices": 0},
}

QUESTIONS = {
    "team": ld.Choice(
        "Which team should own this ticket?",
        {
            "billing": "Invoices, charges, refunds",
            "payments_ops": "Payouts, transfers, bank errors",
            "compliance": "KYC, verification documents, account reviews",
            "technical": "Bugs, API errors, outages",
        },
    ),
    "root_cause": ld.Choice(
        "What most likely blocks the payouts?",
        {
            "kyc_review": "Verification documents are under review",
            "bank_error": "The receiving bank rejects them",
            "platform_outage": "A general platform outage",
            "unknown": "The state does not say",
        },
    ),
    "urgent": ld.Noul("Is the customer's problem time-sensitive?"),
    "churn_risk": ld.Noul("Does the customer threaten to leave?"),
    "changed_docs": ld.Noul("Did the customer recently upload new verification documents?"),
    "frustration": ld.Score(
        "How frustrated is the customer?", ["Calm", "Frustrated but civil", "Very angry"]
    ),
    "clarity": ld.Score(
        "How much actionable detail does the customer give?",
        [
            "None",
            "Some context but no identifiers",
            "Identifiers or error codes",
            "Identifiers, error codes and a timeline",
        ],
    ),
    "reply_tone": ld.Choice(
        "Which reply tone fits best?",
        {
            "apologetic": "Apologize and reassure",
            "neutral": "Plain status update",
            "firm": "Set boundaries",
        },
    ),
}


def measure(engine: ld.Engine, runs: int) -> tuple[float, int, ld.SystemOneResponse]:
    engine.system_one(STATE, QUESTIONS)  # warm-up
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        response = engine.system_one(STATE, QUESTIONS)
        times.append((time.perf_counter() - t0) * 1000)
    t = response.timing
    return statistics.median(times), t.prefix_tokens + t.suffix_tokens, response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    engine = ld.load(args.model)
    results = {}
    results["tree"] = measure(engine, args.runs)

    engine_module.MIN_SHARED_HEADER = 10**9  # disable the question level of the tree
    results["flat"] = measure(engine, args.runs)

    backend = engine.backend
    backend.shared, engine._head = False, None  # disable every level
    results["direct"] = measure(engine, args.runs)

    reference = results["tree"][2]
    print(f"{len(QUESTIONS)} questions, debias=auto, model={engine.model_name}")
    print(f"{'strategy':<8} {'tokens run':>11} {'median ms':>10} {'max |Δp| vs tree':>18}")
    report = {}
    for name, (ms, tokens, response) in results.items():
        worst = 0.0
        for key, answer in response.answers.items():
            ref = reference.answers[key]
            if answer.type == "noul":
                worst = max(worst, abs(answer.noul - ref.noul))
            else:
                worst = max(
                    worst,
                    max(abs(p - ref.probabilities[k]) for k, p in answer.probabilities.items()),
                )
        report[name] = {
            "tokens": tokens,
            "median_ms": round(ms, 1),
            "max_abs_diff": round(worst, 4),
        }
        print(f"{name:<8} {tokens:>11} {ms:>10.0f} {worst:>18.4f}")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
