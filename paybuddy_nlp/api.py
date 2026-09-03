"""
Phase 8 — Deployment.

Exposes the hybrid pipeline as:
  1. categorise_transaction(description) -> dict   (importable function)
  2. A minimal FastAPI service with /categorise and /feedback endpoints

Run with:  uvicorn api:app --reload --port 8010
Then:      curl -X POST localhost:8010/categorise -H "Content-Type: application/json" \
                -d '{"description": "Rapido cab ride"}'
"""

from fastapi import FastAPI
from pydantic import BaseModel

from classifier import CategorisationEngine
from feedback_store import FeedbackStore

engine = CategorisationEngine()
store = FeedbackStore()

app = FastAPI(title="PayBuddy Categorisation Engine (hybrid NLP pipeline)")


def categorise_transaction(description: str) -> dict:
    """The single function this whole project builds toward, per the
    workflow plan's Phase 8: categorise_transaction(description) ->
    {category, confidence, stage_used}."""
    result = engine.classify(description)
    return {
        "category": result.category,
        "confidence": round(result.confidence, 4),
        "stage_used": result.stage,
        "needs_review": result.needs_review,
    }


class CategoriseRequest(BaseModel):
    description: str


class FeedbackRequest(BaseModel):
    transaction_id: int
    description: str
    original_category: str
    original_confidence: float
    corrected_category: str
    user_id: int = 1


@app.post("/categorise")
def categorise(req: CategoriseRequest):
    return categorise_transaction(req.description)


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    entry_id = store.record(
        engine, req.transaction_id, req.description,
        req.original_category, req.original_confidence,
        req.corrected_category, user_id=req.user_id,
    )
    return {"status": "recorded", "entry_id": entry_id}


@app.get("/feedback/history")
def feedback_history(description: str | None = None, transaction_id: int | None = None):
    return store.history_for(description=description, transaction_id=transaction_id)


if __name__ == "__main__":
    # quick smoke test without needing a running server
    print(categorise_transaction("Rapido cab ride"))
    print(categorise_transaction("grabbed a quick bite between classes"))
    print(categorise_transaction("UPI/Zomato/Food"))
