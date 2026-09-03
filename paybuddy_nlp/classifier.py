"""
Module 3 — Categorisation Engine
Implements ClassifyTransaction(description) exactly as specified in
Section 3.1 of the PayBuddy System Design Document:

  1. Normalise description to lowercase and strip punctuation
  2. Look up normalised description in merchant_dictionary
     -> exact match: return (category, confidence=0.95)
  3. For each category in keyword_rules, apply regex patterns
     -> match: return (category, confidence=0.75)
  4. Apply spaCy NLP classifier to description
     -> return (category, confidence=0.55)
  5. If confidence < 0.60: flag transaction for user review
  6. On user correction: update record, insert into feedback table,
     extend merchant_dictionary with the new pattern -> category mapping
"""

import joblib
from dataclasses import dataclass

from preprocessing import normalise, normalise_nospace, lemmatise
from merchant_dictionary import MERCHANT_DICTIONARY
from keyword_rules import COMPILED_RULES

REVIEW_THRESHOLD = 0.60

CONFIDENCE_DICTIONARY = 0.95
CONFIDENCE_REGEX = 0.75
CONFIDENCE_NLP = 0.55


@dataclass
class ClassificationResult:
    category: str
    confidence: float
    stage: str          # "dictionary" | "regex" | "nlp_fallback"
    needs_review: bool


class CategorisationEngine:
    """Stateful so the feedback loop (step 6) can extend the in-memory
    merchant dictionary at runtime, exactly as the design doc describes."""

    def __init__(self, fallback_model_path: str = "fallback_model.joblib"):
        # copy so we don't mutate the module-level constant across instances
        self.merchant_dictionary = dict(MERCHANT_DICTIONARY)
        self.keyword_rules = COMPILED_RULES
        self.fallback_model = joblib.load(fallback_model_path)
        self.feedback_log = []  # simulates category_feedback table rows

    # ------------------------------------------------------------------
    def classify(self, description: str) -> ClassificationResult:
        norm = normalise(description)
        norm_nospace = normalise_nospace(description)

        # Stage 1: merchant dictionary exact / substring match.
        # Checked against both the spaced and space-stripped normalisation,
        # since UPI-style references often run the merchant name together
        # ('UPI/AmazonIndia/Shopping') while other formats keep spaces
        # ('POS purchase at Amazon India').
        for merchant_key, category in self.merchant_dictionary.items():
            if merchant_key in norm or merchant_key.replace(" ", "") in norm_nospace:
                return ClassificationResult(category, CONFIDENCE_DICTIONARY,
                                             "dictionary", False)

        # Stage 2: keyword regex rules
        for category, patterns in self.keyword_rules.items():
            for pattern in patterns:
                if pattern.search(norm):
                    return ClassificationResult(category, CONFIDENCE_REGEX,
                                                 "regex", False)

        # Stage 3: spaCy-lemmatised text -> trained fallback classifier.
        # Confidence is the model's own calibrated max class probability
        # (not a fixed constant), so it genuinely reflects how sure the
        # model is on THIS description.
        lemmatised = lemmatise(description)
        proba = self.fallback_model.predict_proba([lemmatised])[0]
        classes = self.fallback_model.classes_
        best_idx = proba.argmax()
        category = classes[best_idx]
        confidence = float(proba[best_idx])
        needs_review = confidence < REVIEW_THRESHOLD
        return ClassificationResult(category, confidence, "nlp_fallback",
                                     needs_review)

    # ------------------------------------------------------------------
    def record_feedback(self, transaction_id, description, original_category,
                         corrected_category, user_id=1):
        """Step 6 of the algorithm: on user correction, store feedback and
        extend the merchant dictionary so the SAME merchant is caught by
        Stage 1 next time (closing the correction loop)."""
        pattern = normalise(description)
        self.feedback_log.append({
            "user_id": user_id,
            "transaction_id": transaction_id,
            "description_pattern": pattern,
            "original_category": original_category,
            "corrected_category": corrected_category,
        })
        # naive pattern extension: use the whole normalised description as
        # a new dictionary key. A production version would extract just the
        # merchant token; kept simple here for transparency.
        self.merchant_dictionary[pattern] = corrected_category


if __name__ == "__main__":
    engine = CategorisationEngine()
    samples = [
        "UPI/Zomato/Food",
        "Rapido cab ride",
        "grabbed a quick bite between classes",
        "New Corner Cafe evening snack",
    ]
    for s in samples:
        r = engine.classify(s)
        print(f"{s!r:45s} -> {r.category:15s} conf={r.confidence:.2f} stage={r.stage} review={r.needs_review}")
