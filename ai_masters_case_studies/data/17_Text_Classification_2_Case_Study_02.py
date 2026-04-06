"""
Module 17: Text Classification 2
Case Study 02 – Eopinions.csv text classification (Auto vs Camera)
"""

import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, ConfusionMatrixDisplay,
)

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, script_dir)

from PreProcess import Refine

data_dir = os.path.join(project_root, "data", "17_Text_Classification_2_Case_Study_02")
csv_path = os.path.join(data_dir, "Eopinions.csv")

# ---------------------------------------------------------------------------
# Step 1 – Read the file
# ---------------------------------------------------------------------------
print("=" * 60)
print("STEP 1 – Read Eopinions.csv")
print("=" * 60)

df = pd.read_csv(csv_path, encoding="latin-1")
print(f"  Shape   : {df.shape}")
print(f"  Columns : {list(df.columns)}")
print(f"  Classes :\n{df['class'].value_counts().to_string()}")

# ---------------------------------------------------------------------------
# Step 2 – Label Encoding on 'class' column
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2 – Label Encoding ('class' column)")
print("=" * 60)

le = LabelEncoder()
df["Label"] = le.fit_transform(df["class"])

print(f"  Mapping: { {cls: lbl for lbl, cls in enumerate(le.classes_)} }")
print(df[["class", "Label"]].drop_duplicates().sort_values("Label").to_string(index=False))

# ---------------------------------------------------------------------------
# Step 3 – Bar graph: class frequencies
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3 – Bar graph of class frequencies")
print("=" * 60)

class_counts = df["class"].value_counts()
colors = ["#4e79a7", "#e15759"]

plt.figure(figsize=(6, 5))
plt.bar(class_counts.index, class_counts.values, color=colors, edgecolor="black", width=0.5)
plt.title("Frequency of Each Class in Eopinions Dataset", fontsize=13, fontweight="bold")
plt.xlabel("Class", fontsize=12)
plt.ylabel("Count", fontsize=12)
for i, (cls, cnt) in enumerate(class_counts.items()):
    plt.text(i, cnt + 4, str(cnt), ha="center", fontsize=11, fontweight="bold")
plt.tight_layout()
bar_path = os.path.join(data_dir, "ClassFrequencies.png")
plt.savefig(bar_path, dpi=150)
plt.close()
print(f"  Auto: {class_counts.get('Auto', 0)}  |  Camera: {class_counts.get('Camera', 0)}")
print(f"  Saved -> 'ClassFrequencies.png'")

# ---------------------------------------------------------------------------
# Step 4 – Preprocess 'text' column
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4 – Preprocessing 'text' column via PreProcess.Refine()")
print("=" * 60)

df["Processed-Text"] = df["text"].apply(lambda t: " ".join(Refine(str(t))))
print(f"  Done. Sample:")
print(f"  Original  : {df['text'].iloc[0][:80]}...")
print(f"  Processed : {df['Processed-Text'].iloc[0][:80]}...")

# ---------------------------------------------------------------------------
# Step 5 – Vectorize using CountVectorizer
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5 – CountVectorizer")
print("=" * 60)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["Processed-Text"])
y = df["Label"]

print(f"  Feature matrix shape : {X.shape}  (600 docs x {X.shape[1]} vocab terms)")

# ---------------------------------------------------------------------------
# Step 6 – Stratified 80/20 split -> train.csv / test.csv
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 6 – Stratified 80/20 split -> train.csv / test.csv")
print("=" * 60)

(X_train, X_test,
 y_train, y_test,
 idx_train, idx_test) = train_test_split(
    X, y, df.index,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

df.loc[idx_train].to_csv(os.path.join(data_dir, "train.csv"), index=False)
df.loc[idx_test].to_csv(os.path.join(data_dir,  "test.csv"),  index=False)

print(f"  Train : {X_train.shape[0]} rows  "
      f"(Auto={sum(y_train==0)}, Camera={sum(y_train==1)})")
print(f"  Test  : {X_test.shape[0]} rows   "
      f"(Auto={sum(y_test==0)}, Camera={sum(y_test==1)})")
print(f"  Saved -> 'train.csv' and 'test.csv'")

# ---------------------------------------------------------------------------
# Step 7 – Train: Logistic Regression
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 7 – Training: Logistic Regression")
print("=" * 60)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
print(f"  Model trained: {model.__class__.__name__}")
print(f"  Train accuracy: {model.score(X_train, y_train):.4f}")

# ---------------------------------------------------------------------------
# Step 8 – Evaluate: Confusion Matrix
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 8 – Evaluation: Confusion Matrix")
print("=" * 60)

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]   # probability of class=1 (Camera)

cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

print(f"\n  Confusion Matrix (rows=Actual, cols=Predicted):")
print(f"  {'':20s}  Pred Auto   Pred Camera")
print(f"  {'Actual Auto':20s}  {TN:9d}   {FP:11d}")
print(f"  {'Actual Camera':20s}  {FN:9d}   {TP:11d}")

print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save confusion matrix plot
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(
    ax=ax, colorbar=False, cmap="Blues"
)
ax.set_title("Confusion Matrix – Eopinions Test Set", fontweight="bold")
plt.tight_layout()
cm_path = os.path.join(data_dir, "ConfusionMatrix.png")
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"  Confusion matrix plot saved -> 'ConfusionMatrix.png'")

# ---------------------------------------------------------------------------
# Step 9 – ROC Curve
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 9 – ROC Curve")
print("=" * 60)

fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc     = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color="#4e79a7", lw=2,
         label=f"Logistic Regression (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color="grey", lw=1.2, linestyle="--",
         label="Random classifier")
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curve – Eopinions Classification", fontsize=13, fontweight="bold")
plt.legend(loc="lower right")
plt.tight_layout()
roc_path = os.path.join(data_dir, "ROC_Curve.png")
plt.savefig(roc_path, dpi=150)
plt.close()

print(f"  AUC = {roc_auc:.4f}")
print(f"  ROC curve saved -> 'ROC_Curve.png'")
print(f"\n  Test accuracy : {model.score(X_test, y_test):.4f}")
