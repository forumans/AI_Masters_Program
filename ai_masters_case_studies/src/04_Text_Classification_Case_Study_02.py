"""
Module 4: Text Classification 1
Case Study 02 – Wine.csv text preprocessing and vectorization
"""

import os
import sys

import pandas as pd

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, script_dir)

from PreProcess import Refine
from Vectorization import CountVectorization, TFIDFVectorization

data_dir = os.path.join(project_root, "data", "04_Text_Classification_Case_Study_02")
csv_path = os.path.join(data_dir, "Wine.csv")

# ---------------------------------------------------------------------------
# Read Wine.csv
# ---------------------------------------------------------------------------
df = pd.read_csv(csv_path, encoding="latin-1")

print("=" * 60)
print("Wine.csv loaded successfully.")
print(f"Shape   : {df.shape}")
print(f"Columns : {list(df.columns)}")
print("=" * 60)

# ---------------------------------------------------------------------------
# Step 1 – Refine each description -> "Refined-Description"
# ---------------------------------------------------------------------------
print("\nStep 1: Refining descriptions with PreProcess.Refine() ...")

df["Refined-Description"] = df["description"].apply(
    lambda desc: " ".join(Refine(str(desc)))
)

print(f"  Done. Sample refined description:")
print(f"  Original : {df['description'].iloc[0][:80]}...")
print(f"  Refined  : {df['Refined-Description'].iloc[0][:80]}...")

# ---------------------------------------------------------------------------
# Step 2 – Count Vectorization -> "CountVectorizer"
# ---------------------------------------------------------------------------
print("\nStep 2: Count vectorization (building corpus over all descriptions) ...")

refined_texts = df["Refined-Description"].tolist()
count_vectors = CountVectorization(refined_texts)

df["CountVectorizer"] = count_vectors

print(f"  Done. Vector length: {len(count_vectors[0])} (corpus size)")
print(f"  Sample vector (first 10 values): {count_vectors[0][:10]}")

# ---------------------------------------------------------------------------
# Step 3 – TF-IDF Vectorization -> "TF-IDF Vectorizer"
# ---------------------------------------------------------------------------
print("\nStep 3: TF-IDF vectorization ...")

tfidf_vectors = TFIDFVectorization(refined_texts)

df["TF-IDF Vectorizer"] = [
    [round(v, 6) for v in row] for row in tfidf_vectors
]

print(f"  Done. Vector length: {len(tfidf_vectors[0])} (vocabulary size)")
print(f"  Sample vector (first 10 values): "
      f"{[round(v, 4) for v in tfidf_vectors[0][:10]]}")

# ---------------------------------------------------------------------------
# Save back to Wine.csv
# ---------------------------------------------------------------------------
df.to_csv(csv_path, index=False, encoding="utf-8")

print("\n" + "=" * 60)
print("Wine.csv updated and saved successfully.")
print(f"New columns : Refined-Description | CountVectorizer | TF-IDF Vectorizer")
print(f"Final shape : {df.shape}")
print("=" * 60)
