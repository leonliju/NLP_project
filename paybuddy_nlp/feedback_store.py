"""
Phase 7 — Human-in-the-loop feedback mechanism.

Replaces the in-memory-only feedback_log on CategorisationEngine with a
persistent JSON store, so corrections survive a restart and can be
audited over time (who corrected what, when, and what the dictionary
looked like before/after — enabling rollback).

Design mirrors the PayBuddy schema's category_feedback table (user_id,
transaction_id, original vs corrected category) but adds what the actual
repo's version is missing: an append-only, timestamped audit trail, and
a snapshot of the dictionary before each change so a bad correction can
be rolled back.
"""

import json
import os
from datetime import datetime, timezone

FEEDBACK_LOG_PATH = "feedback_log.json"
DICTIONARY_SNAPSHOT_PATH = "merchant_dictionary_snapshots.json"


class FeedbackStore:
    def __init__(self, log_path=FEEDBACK_LOG_PATH, snapshot_path=DICTIONARY_SNAPSHOT_PATH):
        self.log_path = log_path
        self.snapshot_path = snapshot_path
        self._load()

    def _load(self):
        self.entries = []
        if os.path.exists(self.log_path):
            with open(self.log_path) as f:
                self.entries = json.load(f)
        self.snapshots = []
        if os.path.exists(self.snapshot_path):
            with open(self.snapshot_path) as f:
                self.snapshots = json.load(f)

    def _persist(self):
        with open(self.log_path, "w") as f:
            json.dump(self.entries, f, indent=2)
        with open(self.snapshot_path, "w") as f:
            json.dump(self.snapshots, f, indent=2)

    def record(self, engine, transaction_id, description, original_category,
               original_confidence, corrected_category, user_id=1):
        """Records a correction, snapshots the dictionary state BEFORE the
        change (for rollback), then applies the update to the live engine."""
        pattern = engine.record_feedback.__self__  # noop placeholder for clarity
        from preprocessing import normalise
        norm_pattern = normalise(description)

        entry_id = len(self.entries) + 1
        timestamp = datetime.now(timezone.utc).isoformat()

        # snapshot dictionary state before mutation, keyed by entry_id, so
        # this specific correction can be rolled back independently
        self.snapshots.append({
            "entry_id": entry_id,
            "timestamp": timestamp,
            "dictionary_key_added": norm_pattern,
            "dictionary_value_before": engine.merchant_dictionary.get(norm_pattern),  # None if new
        })

        engine.record_feedback(transaction_id, description, original_category,
                                corrected_category, user_id=user_id)

        self.entries.append({
            "entry_id": entry_id,
            "timestamp": timestamp,
            "user_id": user_id,
            "transaction_id": transaction_id,
            "description": description,
            "description_pattern": norm_pattern,
            "original_category": original_category,
            "original_confidence": original_confidence,
            "corrected_category": corrected_category,
        })
        self._persist()
        return entry_id

    def rollback(self, engine, entry_id):
        """Reverts the dictionary mutation made by a specific feedback entry."""
        snap = next((s for s in self.snapshots if s["entry_id"] == entry_id), None)
        if snap is None:
            raise ValueError(f"No snapshot found for entry_id={entry_id}")
        key = snap["dictionary_key_added"]
        before = snap["dictionary_value_before"]
        if before is None:
            engine.merchant_dictionary.pop(key, None)
        else:
            engine.merchant_dictionary[key] = before
        self.entries.append({
            "entry_id": len(self.entries) + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "rollback",
            "rolled_back_entry_id": entry_id,
        })
        self._persist()

    def history_for(self, description=None, transaction_id=None):
        results = self.entries
        if description is not None:
            from preprocessing import normalise
            norm = normalise(description)
            results = [e for e in results if e.get("description_pattern") == norm]
        if transaction_id is not None:
            results = [e for e in results if e.get("transaction_id") == transaction_id]
        return results


if __name__ == "__main__":
    from classifier import CategorisationEngine

    # clean slate for the demo
    for p in (FEEDBACK_LOG_PATH, DICTIONARY_SNAPSHOT_PATH):
        if os.path.exists(p):
            os.remove(p)

    engine = CategorisationEngine()
    store = FeedbackStore()

    desc = "Sold old textbooks to a junior"
    r1 = engine.classify(desc)
    print("before:", r1)

    entry_id = store.record(engine, transaction_id=999, description=desc,
                             original_category=r1.category,
                             original_confidence=r1.confidence,
                             corrected_category="Income")
    r2 = engine.classify(desc)
    print("after correction:", r2)
    print("audit trail:", json.dumps(store.entries, indent=2))

    store.rollback(engine, entry_id)
    r3 = engine.classify(desc)
    print("after rollback:", r3)
