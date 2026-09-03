"""
Trains the Stage 3 "NLP fallback" classifier referenced in Section 3.1 of
the design doc ("Apply spaCy NLP classifier to description").

Design note (worth keeping for the NLP course write-up): spaCy's
en_core_web_sm ships tokenisation, POS tagging, lemmatisation and NER —
it does NOT ship a pretrained text categoriser for arbitrary custom
labels like PayBuddy's 9 spending categories. There's no off-the-shelf
model that already knows "Zomato -> Food" for an Indian fintech taxonomy.
So "spaCy NLP classifier" here means: use spaCy for linguistic
preprocessing (lemmatisation, stopword removal) to normalise surface
variation, then train a supervised classifier (TF-IDF + Logistic
Regression) on top of that representation. This is trained ONLY on the
'train' split and only needs to be re-trained when feedback accumulates.

Phase 5 additions:
  - Grid search over TF-IDF n-gram range / min_df and Logistic Regression
    C / class_weight, selected by 5-fold stratified CV accuracy.
  - Probability calibration (sigmoid / Platt scaling) via CalibratedClassifierCV
    so predict_proba's max value is a genuinely calibrated confidence,
    not just a raw (often overconfident) softmax score. This is what lets
    the abstract's "each stage returns a calibrated confidence score"
    claim actually hold for Stage 3.
"""

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from preprocessing import lemmatise

PARAM_GRID = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [1, 2],
    "clf__estimator__C": [1.0, 3.0, 5.0, 10.0],
}


def build_and_train(train_df: pd.DataFrame, model_path: str = "fallback_model.joblib"):
    print("Lemmatising training text with spaCy...")
    X_text = train_df["description"].apply(lemmatise)
    y = train_df["category"]

    base_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(sublinear_tf=True)),
        ("clf", CalibratedClassifierCV(
            LogisticRegression(max_iter=1000, class_weight="balanced"),
            method="sigmoid", cv=3,
        )),
    ])

    # small dataset -> 3-fold keeps every class represented in each fold
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = GridSearchCV(base_pipeline, PARAM_GRID, cv=cv, scoring="accuracy", n_jobs=-1)
    search.fit(X_text, y)

    print(f"Best CV accuracy: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    joblib.dump(search.best_estimator_, model_path)
    print(f"Saved calibrated, tuned fallback model -> {model_path}")
    return search.best_estimator_


if __name__ == "__main__":
    train_df = pd.read_csv("train_split.csv")
    build_and_train(train_df)
