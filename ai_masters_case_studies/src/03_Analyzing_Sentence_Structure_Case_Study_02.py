"""
Module 3: Analyzing Sentence Structure
Case Study 02 - Tweets.csv Analysis (Twitter Airline Sentiment)
"""

import os
import re

import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag, RegexpParser

nltk.download("punkt",                        quiet=True)
nltk.download("punkt_tab",                    quiet=True)
nltk.download("averaged_perceptron_tagger",   quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_dir     = os.path.join(project_root, "data", "03_Analyzing_Sentence_Structure_Case_Study_02")

# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------
csv_path = os.path.join(data_dir, "Tweets.csv")
df = pd.read_csv(csv_path, encoding="latin-1")
df = df[["text", "airline_sentiment"]].dropna()

print("=" * 60)
print("Tweets.csv loaded successfully.")
print(f"Total rows  : {len(df)}")
print(f"Sentiments  : {sorted(df['airline_sentiment'].unique().tolist())}")
print(df["airline_sentiment"].value_counts().to_string())
print("=" * 60)

SENTIMENTS = ["positive", "negative", "neutral"]


# =============================================================================
# Question 1 – Extract all @mentions -> References.txt
# =============================================================================

print("\n" + "=" * 60)
print("QUESTION 1 – Extracting @mentions -> References.txt")
print("=" * 60)

all_tags = []
for tweet in df["text"]:
    all_tags.extend(re.findall(r"@\w+", str(tweet)))

references_path = os.path.join(data_dir, "References.txt")
with open(references_path, "w", encoding="utf-8") as f:
    f.write("\n".join(all_tags))

print(f"Total @mentions found  : {len(all_tags)}")
print(f"Unique @mentions       : {len(set(all_tags))}")
print(f"Saved -> 'References.txt'")


# =============================================================================
# Helpers – phrase extraction with RegexpParser
# =============================================================================

NP_GRAMMAR = r"NP: {<DT>?<JJ>*<NN.*>+}"
VP_GRAMMAR = r"VP: {<VB.*>(<RB.*>|<JJ.*>|<DT>|<NN.*>)*}"

_np_parser = RegexpParser(NP_GRAMMAR)
_vp_parser = RegexpParser(VP_GRAMMAR)


def extract_phrases(tweets, parser, label):
    """
    Tokenise, POS-tag and chunk each tweet.
    Returns a list of phrase strings matching the given chunk label.
    """
    phrases = []
    for tweet in tweets:
        tokens = word_tokenize(str(tweet))
        tagged = pos_tag(tokens)
        tree   = parser.parse(tagged)
        for subtree in tree.subtrees(filter=lambda t: t.label() == label):
            phrase = " ".join(w for w, _ in subtree.leaves())
            if phrase.strip():
                phrases.append(phrase)
    return phrases


# =============================================================================
# Question 2 – Noun Phrases per sentiment
# =============================================================================

print("\n" + "=" * 60)
print("QUESTION 2 – Extracting Noun Phrases per sentiment")
print("(This may take a few minutes for 14 K tweets)")
print("=" * 60)

np_counts = {}
for sentiment in SENTIMENTS:
    tweets = df[df["airline_sentiment"] == sentiment]["text"].tolist()
    print(f"\n  [{sentiment}]  {len(tweets)} tweets ... ", end="", flush=True)
    phrases = extract_phrases(tweets, _np_parser, "NP")
    np_counts[sentiment] = len(phrases)

    fname = os.path.join(data_dir, f"Noun Phrases for {sentiment} Review.txt")
    with open(fname, "w", encoding="utf-8") as f:
        f.write("\n".join(phrases))
    print(f"{len(phrases)} noun phrases saved.")

print("\nAll Noun Phrase files saved.")


# =============================================================================
# Question 3 – Verb Phrases per sentiment
# =============================================================================

print("\n" + "=" * 60)
print("QUESTION 3 – Extracting Verb Phrases per sentiment")
print("=" * 60)

vp_counts = {}
for sentiment in SENTIMENTS:
    tweets = df[df["airline_sentiment"] == sentiment]["text"].tolist()
    print(f"\n  [{sentiment}]  {len(tweets)} tweets ... ", end="", flush=True)
    phrases = extract_phrases(tweets, _vp_parser, "VP")
    vp_counts[sentiment] = len(phrases)

    fname = os.path.join(data_dir, f"Verb Phrases for {sentiment} Review.txt")
    with open(fname, "w", encoding="utf-8") as f:
        f.write("\n".join(phrases))
    print(f"{len(phrases)} verb phrases saved.")

print("\nAll Verb Phrase files saved.")


# =============================================================================
# Question 4 – Pie charts: NP vs VP distribution per sentiment
# =============================================================================

print("\n" + "=" * 60)
print("QUESTION 4 – Pie charts per sentiment")
print("=" * 60)

colors = ["#4e79a7", "#e15759"]

for sentiment in SENTIMENTS:
    np_count = np_counts[sentiment]
    vp_count = vp_counts[sentiment]

    labels = [f"Noun Phrases\n({np_count})", f"Verb Phrases\n({vp_count})"]
    sizes  = [np_count, vp_count]

    plt.figure(figsize=(7, 7))
    plt.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    plt.title(
        f"Noun Phrases vs Verb Phrases\n({sentiment.capitalize()} Sentiment Reviews)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    chart_path = os.path.join(data_dir, f"PhraseDistribution_{sentiment}.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"  {sentiment:8s}: NP={np_count:6d}, VP={vp_count:6d}  "
          f"-> 'PhraseDistribution_{sentiment}.png'")

print("\nAll pie charts saved.")
