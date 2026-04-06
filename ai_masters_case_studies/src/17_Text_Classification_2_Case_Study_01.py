"""
Module 17: Text Classification 2
Case Study 01 – Data splitting, label encoding, confusion matrix & ROC
"""

import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve, auc

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_dir     = os.path.join(project_root, "data", "17_Text_Classification_2_Case_Study_01")


# =============================================================================
# Question 1 – Random 60/40 split -> First.csv / Second.csv
# =============================================================================

def SplitData(csv_path):
    """
    Reads a CSV file, randomly splits its rows into 60 % and 40 %,
    and saves them as 'First.csv' and 'Second.csv' in the same folder.
    Returns the two DataFrames.
    """
    df      = pd.read_csv(csv_path)
    first   = df.sample(frac=0.60, random_state=42)
    second  = df.drop(first.index)

    out_dir = os.path.dirname(csv_path)
    first.to_csv(os.path.join(out_dir, "First.csv"),  index=False)
    second.to_csv(os.path.join(out_dir, "Second.csv"), index=False)

    return first, second


print("=" * 60)
print("QUESTION 1 – Random 60/40 Split (HouseData.csv)")
print("=" * 60)

house_path     = os.path.join(data_dir, "HouseData.csv")
first_df, second_df = SplitData(house_path)

print(f"  Original rows : {len(first_df) + len(second_df)}")
print(f"  First.csv     : {len(first_df)} rows  ({len(first_df)/(len(first_df)+len(second_df))*100:.1f} %)")
print(f"  Second.csv    : {len(second_df)} rows  ({len(second_df)/(len(first_df)+len(second_df))*100:.1f} %)")
print("  Saved -> 'First.csv' and 'Second.csv'")


# =============================================================================
# Question 2 – Label Encoding of 'class' column -> 'Label' column
# =============================================================================

def LabelEncode(csv_path):
    """
    Reads a CSV file, applies LabelEncoder to the 'class' column (case-
    insensitive lookup), stores encoded values in a new 'Label' column,
    and overwrites the original file.
    Returns the updated DataFrame and the fitted LabelEncoder.
    """
    df = pd.read_csv(csv_path)

    # Find the class column (case-insensitive)
    class_col = next((c for c in df.columns if c.lower() == "class"), None)
    if class_col is None:
        raise ValueError(f"No 'class' column found in {csv_path}")

    le = LabelEncoder()
    df["Label"] = le.fit_transform(df[class_col].astype(str))
    df.to_csv(csv_path, index=False)
    return df, le


print("\n" + "=" * 60)
print("QUESTION 2 – Label Encoding (Marketing.csv)")
print("=" * 60)

marketing_path = os.path.join(data_dir, "Marketing.csv")
mkt_df, le     = LabelEncode(marketing_path)

print(f"  Column encoded : '{next(c for c in mkt_df.columns if c.lower() == 'class')}'")
print(f"  Unique classes : {sorted(le.classes_)}")
print(f"\n  Encoding mapping:")
for label, cls in enumerate(le.classes_):
    print(f"    {cls!r:20s} -> {label}")
print(f"\n  'Label' column added and saved to 'Marketing.csv'")
print(f"  Sample:\n{mkt_df[['class', 'Label']].drop_duplicates().sort_values('Label').to_string(index=False)}")


# =============================================================================
# Question 3 – Confusion Matrix, Metrics & ROC Curve (Results.csv)
# =============================================================================

def EvaluateClassifier(csv_path):
    """
    Reads a CSV with 'ActualValues' and 'PredictedValues' columns,
    prints confusion-matrix metrics with plain-English explanations,
    plots and saves the ROC curve, and returns a metrics dict.
    """
    df     = pd.read_csv(csv_path)
    actual = df["ActualValues"]
    pred   = df["PredictedValues"]

    # Confusion matrix components
    cm         = confusion_matrix(actual, pred)
    TN, FP, FN, TP = cm.ravel()
    total      = TP + TN + FP + FN

    accuracy          = (TP + TN) / total
    misclass_rate     = (FP + FN) / total
    true_pos_rate     = TP / (TP + FN)      # Sensitivity / Recall
    false_pos_rate    = FP / (FP + TN)
    specificity       = TN / (TN + FP)
    precision         = TP / (TP + FP)
    majority_class    = max((actual == 0).sum(), (actual == 1).sum())
    null_error_rate   = 1 - (majority_class / total)

    metrics = {
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "Accuracy":              accuracy,
        "Misclassification Rate": misclass_rate,
        "True Positive Rate":    true_pos_rate,
        "False Positive Rate":   false_pos_rate,
        "Specificity":           specificity,
        "Precision":             precision,
        "Null Error Rate":       null_error_rate,
    }

    # ---------- Print Confusion Matrix ----------
    print(f"\n  Confusion Matrix:")
    print(f"  {'':25s}  Predicted 0   Predicted 1")
    print(f"  {'Actual 0':25s}  {TN:10d}   {FP:10d}")
    print(f"  {'Actual 1':25s}  {FN:10d}   {TP:10d}")

    # ---------- Print Metrics with Explanations ----------
    explanations = {
        "Accuracy":
            "Proportion of all predictions that were correct. "
            "Higher is better.",
        "Misclassification Rate":
            "Proportion of all predictions that were wrong "
            "(complement of Accuracy).",
        "True Positive Rate":
            "Of all actual positives, the fraction the model correctly "
            "identified (also called Sensitivity or Recall).",
        "False Positive Rate":
            "Of all actual negatives, the fraction the model incorrectly "
            "labelled as positive (Type I error rate).",
        "Specificity":
            "Of all actual negatives, the fraction correctly identified as "
            "negative (complement of False Positive Rate).",
        "Precision":
            "Of all predicted positives, the fraction that were truly "
            "positive; measures prediction trustworthiness.",
        "Null Error Rate":
            "Error rate of a naive baseline that always predicts the "
            "majority class; our model should beat this to be useful.",
    }

    print()
    for name, expl in explanations.items():
        val = metrics[name]
        print(f"  {name:<25s}: {val:.4f}")
        print(f"    -> {expl}\n")

    # ---------- ROC Curve ----------
    fpr_pts, tpr_pts, _ = roc_curve(actual, pred)
    roc_auc              = auc(fpr_pts, tpr_pts)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr_pts, tpr_pts, color="#4e79a7", lw=2,
             label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="grey", lw=1.2, linestyle="--",
             label="Random classifier")
    plt.scatter([false_pos_rate], [true_pos_rate], color="#e15759", zorder=5,
                label=f"Classifier  (FPR={false_pos_rate:.4f}, TPR={true_pos_rate:.4f})")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve – Results.csv", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()

    roc_path = os.path.join(os.path.dirname(csv_path), "ROC_Curve.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"  ROC curve saved -> 'ROC_Curve.png'  (AUC = {roc_auc:.4f})")

    return metrics


print("\n" + "=" * 60)
print("QUESTION 3 – Confusion Matrix & Metrics (Results.csv)")
print("=" * 60)

results_path = os.path.join(data_dir, "Results.csv")
metrics      = EvaluateClassifier(results_path)
