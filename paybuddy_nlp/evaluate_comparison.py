"""
Phase 6 — Evaluation and robustness analysis.

Produces, on the same held-out test_split.csv for a fair comparison:
  1. Baseline (naive rule-only) pipeline metrics
  2. Hybrid pipeline metrics
  3. Stratified accuracy by difficulty partition (dictionary / regex / nlp_hard)
     for BOTH pipelines, to substantiate the abstract's core claim.
  4. Per-partition confusion matrices for the hybrid pipeline.
  5. Stage-routing diagnostic (already existed, kept).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

from classifier import CategorisationEngine
from baseline_pipeline import baseline_categorise

CATEGORY_ORDER = ["Bills", "Education", "Entertainment", "Food", "Income",
                   "Medical", "Other", "Shopping", "Travel"]
PARTITIONS = ["dictionary", "regex", "nlp_hard"]


def run_pipeline(df, engine):
    preds, confs, stages = [], [], []
    for desc in df["description"]:
        r = engine.classify(desc)
        preds.append(r.category)
        confs.append(r.confidence)
        stages.append(r.stage)
    return preds, confs, stages


def run_baseline(df):
    preds, confs, stages = [], [], []
    for desc in df["description"]:
        r = baseline_categorise(desc)
        preds.append(r.category)
        confs.append(r.confidence)
        stages.append(r.stage)
    return preds, confs, stages


def plot_confusion(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=CATEGORY_ORDER)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CATEGORY_ORDER)))
    ax.set_yticks(range(len(CATEGORY_ORDER)))
    ax.set_xticklabels(CATEGORY_ORDER, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(CATEGORY_ORDER, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    for i in range(len(CATEGORY_ORDER)):
        for j in range(len(CATEGORY_ORDER)):
            val = cm[i, j]
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center",
                         color="white" if val > cm.max() / 2 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)


def main():
    df = pd.read_csv("test_split.csv")
    engine = CategorisationEngine()

    h_preds, h_confs, h_stages = run_pipeline(df, engine)
    b_preds, b_confs, b_stages = run_baseline(df)

    df["hybrid_pred"] = h_preds
    df["hybrid_conf"] = h_confs
    df["hybrid_stage"] = h_stages
    df["hybrid_correct"] = df["hybrid_pred"] == df["category"]

    df["baseline_pred"] = b_preds
    df["baseline_conf"] = b_confs
    df["baseline_stage"] = b_stages
    df["baseline_correct"] = df["baseline_pred"] == df["category"]

    print("=" * 70)
    print("OVERALL COMPARISON")
    print("=" * 70)
    baseline_acc = accuracy_score(df["category"], df["baseline_pred"])
    hybrid_acc = accuracy_score(df["category"], df["hybrid_pred"])
    baseline_f1 = f1_score(df["category"], df["baseline_pred"], average="macro", zero_division=0)
    hybrid_f1 = f1_score(df["category"], df["hybrid_pred"], average="macro", zero_division=0)
    print(f"{'Metric':<20}{'Baseline':<15}{'Hybrid':<15}{'Delta':<10}")
    print(f"{'Accuracy':<20}{baseline_acc:<15.4f}{hybrid_acc:<15.4f}{hybrid_acc-baseline_acc:+.4f}")
    print(f"{'Macro F1':<20}{baseline_f1:<15.4f}{hybrid_f1:<15.4f}{hybrid_f1-baseline_f1:+.4f}")

    print("\n" + "=" * 70)
    print("STRATIFIED ACCURACY BY DIFFICULTY PARTITION (the key result)")
    print("=" * 70)
    strat = df.groupby("bucket").agg(
        n=("category", "count"),
        baseline_acc=("baseline_correct", "mean"),
        hybrid_acc=("hybrid_correct", "mean"),
    )
    strat["delta"] = strat["hybrid_acc"] - strat["baseline_acc"]
    strat = strat.reindex(PARTITIONS)
    print(strat.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 70)
    print("PER-CATEGORY REPORT — BASELINE")
    print("=" * 70)
    print(classification_report(df["category"], df["baseline_pred"], zero_division=0))

    print("\n" + "=" * 70)
    print("PER-CATEGORY REPORT — HYBRID")
    print("=" * 70)
    print(classification_report(df["category"], df["hybrid_pred"], zero_division=0))

    print("\n" + "=" * 70)
    print("STAGE ROUTING (hybrid) — did each partition get handled by the intended stage?")
    print("=" * 70)
    print(pd.crosstab(df["bucket"], df["hybrid_stage"]))

    print("\n" + "=" * 70)
    print("BASELINE STAGE ROUTING — note 'default_other' dominates nlp_hard")
    print("=" * 70)
    print(pd.crosstab(df["bucket"], df["baseline_stage"]))

    # Overall confusion matrices
    plot_confusion(df["category"], df["baseline_pred"],
                    f"Baseline pipeline — overall accuracy {baseline_acc:.1%}",
                    "confusion_baseline_overall.png")
    plot_confusion(df["category"], df["hybrid_pred"],
                    f"Hybrid pipeline — overall accuracy {hybrid_acc:.1%}",
                    "confusion_hybrid_overall.png")

    # Per-partition confusion matrices for the hybrid pipeline
    for part in PARTITIONS:
        sub = df[df["bucket"] == part]
        acc = sub["hybrid_correct"].mean()
        plot_confusion(sub["category"], sub["hybrid_pred"],
                        f"Hybrid — '{part}' partition (n={len(sub)}, acc={acc:.1%})",
                        f"confusion_hybrid_{part}.png")
        b_acc = sub["baseline_correct"].mean()
        plot_confusion(sub["category"], sub["baseline_pred"],
                        f"Baseline — '{part}' partition (n={len(sub)}, acc={b_acc:.1%})",
                        f"confusion_baseline_{part}.png")

    df.to_csv("comparison_results.csv", index=False)
    print("\nSaved comparison_results.csv and confusion_*.png files")

    return df


if __name__ == "__main__":
    main()
