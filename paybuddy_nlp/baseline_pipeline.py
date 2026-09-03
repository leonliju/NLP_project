"""
Phase 3 — Baseline rule-based pipeline.

This reconstructs the CURRENT categorise() logic found in the actual
PayBuddy repo (backend/routers/transactions.py), bugs included:

  - Stage 1: merchant dictionary substring match -> confidence 0.95
  - Stage 2: UNANCHORED regex keyword match (no \\b word boundaries)
             -> confidence 0.75
  - No Stage 3 at all: anything that falls through both stages is
    labelled 'Other' with a flat, hardcoded confidence of 0.50.

Kept deliberately "buggy" (unanchored patterns) so we can measure, not
just claim, how much the hybrid pipeline improves on it.
"""

import re
import pandas as pd
from dataclasses import dataclass

from preprocessing import normalise

# Same dictionary as the hybrid pipeline, for a fair comparison — the
# dictionary itself isn't the problem, the regex stage is.
from merchant_dictionary import MERCHANT_DICTIONARY

# Unanchored keyword patterns, deliberately mirroring the substring-matching
# style found in the real repo (no \b boundaries).
BASELINE_KEYWORD_RULES = {
    "Bills": ["bill", "recharge", "electricity", "broadband", "emi", "insurance", "rent"],
    "Education": ["school", "college", "course", "book", "exam", "fee", "tuition"],
    "Entertainment": ["movie", "cinema", "concert", "game", "subscription", "ott"],
    "Food": ["food", "restaurant", "cafe", "hotel", "eat", "lunch", "dinner", "breakfast"],
    "Medical": ["pharmacy", "hospital", "clinic", "medical", "doctor", "medicine"],
    "Shopping": ["mart", "store", "mall", "shopping", "purchase", "order"],
    "Travel": ["cab", "taxi", "flight", "train", "bus", "metro", "fuel", "petrol"],
    "Income": ["salary", "stipend", "refund", "cashback", "credited", "income"],
}


@dataclass
class BaselineResult:
    category: str
    confidence: float
    stage: str


def baseline_categorise(description: str) -> BaselineResult:
    norm = normalise(description)

    # Stage 1: dictionary (identical to hybrid pipeline)
    for merchant_key, category in MERCHANT_DICTIONARY.items():
        if merchant_key in norm:
            return BaselineResult(category, 0.95, "dictionary")

    # Stage 2: unanchored substring keyword match (the actual repo's bug)
    for category, keywords in BASELINE_KEYWORD_RULES.items():
        for kw in keywords:
            if kw in norm:  # <-- no \b, this is the bug under study
                return BaselineResult(category, 0.75, "regex")

    # No fallback stage: dump into 'Other' at a flat, meaningless confidence
    return BaselineResult("Other", 0.50, "default_other")


if __name__ == "__main__":
    tests = [
        ("POS purchase at Costa Coffee", "should NOT be Education, but 'fee' isn't in this one"),
        ("Payment to Third Wave Coffee via UPI", "contains 'fee'? no -> check 'eat'? no"),
        ("Facebook Ads payment", "'book' substring bug -> wrongly Education"),
        ("Auto-debit - repeat prescription", "'eat' substring inside 'repeat' -> wrongly Food"),
        ("NEFT credit from parent", "'rent' substring inside 'parent' -> wrongly Bills"),
    ]
    for desc, note in tests:
        r = baseline_categorise(desc)
        print(f"{desc!r:45s} -> {r.category:12s} ({r.stage:14s}) | {note}")
