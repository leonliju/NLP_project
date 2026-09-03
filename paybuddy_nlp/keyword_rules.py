"""
Stage 2 — Keyword Regex Rules
Applied only when Stage 1 (merchant dictionary) finds no match.
Catches merchants PayBuddy has never seen before, but whose description
contains a generic category-indicating phrase. Confidence for a regex
hit = 0.75.

Patterns are deliberately specific phrases (not single common words) so
that Stage 3 (NLP fallback) still has genuine, non-overlapping work to do
on descriptions that mention a category concept indirectly.
"""

import re

KEYWORD_RULES = {
    "Bills": [
        r"electricity bill", r"power bill", r"broadband", r"wifi bill",
        r"mobile recharge", r"\brecharge\b", r"rent due", r"maintenance charge",
        r"water bill", r"utility bill", r"gas bill", r"dth recharge",
    ],
    "Education": [
        r"tuition fee", r"semester fee", r"exam fee", r"college fee",
        r"university fee", r"course fee", r"library fine", r"hostel fee",
        r"registration fee", r"workshop fee",
    ],
    "Entertainment": [
        r"movie ticket", r"\bcinema\b", r"concert ticket", r"\barcade\b",
        r"ott subscription", r"streaming subscription", r"gaming zone",
        r"amusement park",
    ],
    "Food": [
        r"\brestaurant\b", r"\bcafe\b", r"food delivery", r"\btiffin\b",
        r"\bdine\b|\bdining\b", r"\bbakery\b", r"\bcanteen\b", r"\bdhaba\b",
        r"\bmess\b",
    ],
    "Medical": [
        r"\bpharmacy\b", r"hospital bill", r"\bclinic\b", r"medical store",
        r"doctor fee", r"\bmedicine\b", r"diagnostic", r"lab test",
    ],
    "Shopping": [
        r"shopping mall", r"electronics store", r"clothing store",
        r"online order", r"department store", r"supermarket", r"general store",
    ],
    "Travel": [
        r"cab ride", r"\btaxi\b", r"flight ticket", r"train ticket",
        r"bus ticket", r"metro card", r"\bfuel\b", r"\bpetrol\b", r"\bdiesel\b",
        r"\bola\b", r"\buber\b", r"\birctc\b",
    ],
    "Income": [
        r"salary credited", r"stipend credited", r"refund received",
        r"cashback credited", r"prize money", r"freelance payment",
        r"interest credited",
    ],
    "Other": [
        r"atm withdrawal", r"bank charge", r"\bmisc\b", r"transfer to",
        r"unidentified",
    ],
}

# Pre-compile for speed
COMPILED_RULES = {
    category: [re.compile(p) for p in patterns]
    for category, patterns in KEYWORD_RULES.items()
}
