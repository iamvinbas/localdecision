"""Confidence-gated support routing: the model decides, your code stays in control.

    python examples/support_router.py            # uses the default local model
    python examples/support_router.py --mock     # no weights, just the plumbing

Every ticket gets three independent questions in ONE call (one shared prefill). The code then
combines them with plain rules and routes low-confidence cases to a human.
"""

from __future__ import annotations

import sys

import localdecision as ld

TICKETS = [
    "I was charged twice for invoice #4411, can you refund the duplicate? Thanks!",
    "Your API returns 500 on every call since 9am. Checkout is DOWN and we are losing money.",
    "Can't log in, the reset email never arrives. I need access before a demo in one hour.",
    "Hi, do you offer discounts for non-profits on the Pro plan?",
    "hmm the thing from last week again",
]

QUESTIONS = {
    "team": ld.Choice(
        "Which team should handle this message?",
        {
            "billing": "Payments, invoices, charges, refunds",
            "technical": "Bugs, errors, crashes, outages",
            "account": "Login, passwords, account access",
            "sales": "Pricing, plans, new purchases",
        },
    ),
    "urgent": ld.Noul("Does the message convey urgency?"),
    "anger": ld.Score("How frustrated is the customer?", ["Calm", "Frustrated", "Very angry"]),
}


def route(result: ld.SystemOneResponse) -> str:
    team = result.answers["team"]
    urgent = result.answers["urgent"].noul
    anger = result.answers["anger"].score
    agreement = result.diagnostics["team"].agreement if result.diagnostics else 1.0

    if team.confidence < 0.6 or agreement < 1.0:
        return "human triage (model unsure)"
    priority = "P1" if urgent > 0.8 or anger > 1.5 else "P3"
    return f"{team.choice} queue, {priority}"


def main() -> None:
    engine = ld.load(backend="mock") if "--mock" in sys.argv else ld.load()
    for ticket in TICKETS:
        result = engine.system_one(ticket, QUESTIONS)
        team = result.answers["team"]
        print(
            f"{ticket[:60]:<62} -> {route(result):<30} "
            f"(team={team.choice} conf={team.confidence:.2f}, "
            f"urgent={result.answers['urgent'].noul:.2f}, {result.timing.total_ms:.0f} ms)"
        )


if __name__ == "__main__":
    main()
