import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from classifier import CategorisationEngine

CATEGORY_ORDER = ["Bills", "Education", "Entertainment", "Food", "Income",
                   "Medical", "Other", "Shopping", "Travel"]


def run_evaluation(test_csv="test_split.csv"):
    df = pd.read_csv(test_csv)
    engine = CategorisationEngine()

    preds, confs, stages, reviews = [], [], [], []
    for desc in df["description"]:
        r = engine.classify(desc)
        preds.append(r.category)
        confs.append(r.confidence)
        stages.append(r.stage)
        reviews.append(r.needs_review)

    df = df.copy()
    df["predicted_category"] = preds
    df["confidence"] = confs
    df["stage_used"] = stages
    df["flagged_for_review"] = reviews
    df["correct"] = df["predicted_category"] == df["category"]

    overall_acc = accuracy_score(df["category"], df["predicted_category"])
    print(f"\n=== Overall accuracy: {overall_acc:.4f} ({df['correct'].sum()}/{len(df)}) ===\n")

    print("--- Accuracy by input bucket (ground truth of which stage SHOULD catch it) ---")
    print(df.groupby("bucket")["correct"].agg(["mean", "count"]).rename(columns={"mean": "accuracy"}))

    print("\n--- Accuracy by stage actually used ---")
    print(df.groupby("stage_used")["correct"].agg(["mean", "count"]).rename(columns={"mean": "accuracy"}))

    print("\n--- Per-category accuracy ---")
    print(df.groupby("category")["correct"].agg(["mean", "count"]).rename(columns={"mean": "accuracy"}))

    print("\n--- Bucket -> Stage routing table (did each input get handled by the intended stage?) ---")
    print(pd.crosstab(df["bucket"], df["stage_used"]))

    print("\n--- Full classification report ---")
    print(classification_report(df["category"], df["predicted_category"], zero_division=0))

    # Confusion matrix plot
    cm = confusion_matrix(df["category"], df["predicted_category"], labels=CATEGORY_ORDER)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CATEGORY_ORDER)))
    ax.set_yticks(range(len(CATEGORY_ORDER)))
    ax.set_xticklabels(CATEGORY_ORDER, rotation=45, ha="right")
    ax.set_yticklabels(CATEGORY_ORDER)
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("True category")
    ax.set_title(f"PayBuddy Categorisation Engine — Confusion Matrix\nOverall accuracy: {overall_acc:.1%}")
    for i in range(len(CATEGORY_ORDER)):
        for j in range(len(CATEGORY_ORDER)):
            val = cm[i, j]
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center",
                         color="white" if val > cm.max() / 2 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("\nSaved confusion_matrix.png")

    df.to_csv("test_predictions.csv", index=False)
    print("Saved test_predictions.csv")

    # misclassification review, useful for the design doc's Section 10.1
    print("\n--- Misclassified examples ---")
    wrong = df[~df["correct"]][["description", "category", "predicted_category", "stage_used", "bucket"]]
    print(wrong.to_string(index=False))

    return df, overall_acc


if __name__ == "__main__":
    run_evaluation()
