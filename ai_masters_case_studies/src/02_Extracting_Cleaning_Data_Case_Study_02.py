"""
Module 2: Extracting, Cleaning and Preprocessing Data
Case Study 02 - Brexit.docx Analysis
"""

import os
from collections import Counter

import spacy
import matplotlib.pyplot as plt
from docx import Document
from nltk import ngrams
from nltk.tokenize import word_tokenize
import nltk

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_dir     = os.path.join(project_root, "data", "02_Extracting_Cleaning_Data_Case_Study_02")
docx_path    = os.path.join(data_dir, "Brexit.docx")

# ---------------------------------------------------------------------------
# Load spaCy model (used for POS tagging and NER)
# ---------------------------------------------------------------------------
nlp = spacy.load("en_core_web_sm")

# ---------------------------------------------------------------------------
# Read Brexit.docx
# ---------------------------------------------------------------------------
doc_obj  = Document(docx_path)
text     = "\n".join(para.text for para in doc_obj.paragraphs if para.text.strip())
spacy_doc = nlp(text)

print("=" * 60)
print("Brexit.docx loaded successfully.")
print(f"Total characters : {len(text)}")
print(f"Total words      : {len(text.split())}")
print("=" * 60)


# =============================================================================
# Question 1 – N-Grams
# =============================================================================

def GetNGrams(string, n):
    """Returns a list of n-grams (as tuples) from the input string."""
    tokens  = word_tokenize(string)
    n_grams = list(ngrams(tokens, n))
    return n_grams


print("\n" + "=" * 60)
print("QUESTION 1 – N-Grams")
print("=" * 60)

for n in [2, 3, 4]:
    result = GetNGrams(text, n)
    print(f"\nn={n} grams (first 5 shown):")
    for gram in result[:5]:
        print(" ", gram)
    print(f"  ... total {len(result)} {n}-grams")


# =============================================================================
# Question 2 – POS Tag Counts & Pie Chart
# =============================================================================

# POS tag prefixes used by spaCy:
#   Nouns      : NN, NNS, NNP, NNPS  -> pos_ == "NOUN" or "PROPN"
#   Pronouns   : PRP, PRP$, WP, WP$  -> pos_ == "PRON"
#   Adjectives : JJ, JJR, JJS        -> pos_ == "ADJ"
#   Verbs      : VB, VBD, VBG, ...   -> pos_ == "VERB"
#   Adverbs    : RB, RBR, RBS, WRB   -> pos_ == "ADV"

def NounsCount(string):
    """Returns the count of all noun forms (common + proper) in the string."""
    doc = nlp(string)
    return sum(1 for token in doc if token.pos_ in ("NOUN", "PROPN"))


def PronounsCount(string):
    """Returns the count of all pronoun forms in the string."""
    doc = nlp(string)
    return sum(1 for token in doc if token.pos_ == "PRON")


def AdjectivesCount(string):
    """Returns the count of all adjective forms in the string."""
    doc = nlp(string)
    return sum(1 for token in doc if token.pos_ == "ADJ")


def VerbsCount(string):
    """Returns the count of all verb forms in the string."""
    doc = nlp(string)
    return sum(1 for token in doc if token.pos_ == "VERB")


def AdverbsCount(string):
    """Returns the count of all adverb forms in the string."""
    doc = nlp(string)
    return sum(1 for token in doc if token.pos_ == "ADV")


print("\n" + "=" * 60)
print("QUESTION 2 – POS Tag Counts")
print("=" * 60)

nouns      = NounsCount(text)
pronouns   = PronounsCount(text)
adjectives = AdjectivesCount(text)
verbs      = VerbsCount(text)
adverbs    = AdverbsCount(text)

print(f"  Nouns      : {nouns}")
print(f"  Pronouns   : {pronouns}")
print(f"  Adjectives : {adjectives}")
print(f"  Verbs      : {verbs}")
print(f"  Adverbs    : {adverbs}")

# Pie chart
labels = ["Nouns", "Pronouns", "Adjectives", "Verbs", "Adverbs"]
sizes  = [nouns, pronouns, adjectives, verbs, adverbs]
colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"]

plt.figure(figsize=(8, 8))
plt.pie(
    sizes,
    labels=labels,
    colors=colors,
    autopct="%1.1f%%",
    startangle=140,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
)
plt.title("Distribution of POS Tags in Brexit.docx", fontsize=14, fontweight="bold")
plt.tight_layout()
pie_path = os.path.join(data_dir, "POSDistribution.png")
plt.savefig(pie_path, dpi=150)
plt.show()
print(f"\nPie chart saved as 'POSDistribution.png'")


# =============================================================================
# Question 3 – Named Entity Counts (NER)
# =============================================================================

def GeoPoliticalCount(string):
    """Returns the count of geo-political entities (GPE) in the string."""
    doc = nlp(string)
    return sum(1 for ent in doc.ents if ent.label_ == "GPE")


def PersonsCount(string):
    """Returns the count of person entities (PERSON) in the string."""
    doc = nlp(string)
    return sum(1 for ent in doc.ents if ent.label_ == "PERSON")


def OrganizationsCount(string):
    """Returns the count of organisation entities (ORG) in the string."""
    doc = nlp(string)
    return sum(1 for ent in doc.ents if ent.label_ == "ORG")


print("\n" + "=" * 60)
print("QUESTION 3 – Named Entity Recognition (NER)")
print("=" * 60)

geo_count  = GeoPoliticalCount(text)
per_count  = PersonsCount(text)
org_count  = OrganizationsCount(text)

print(f"  Geo-Political entities : {geo_count}")
print(f"  Persons                : {per_count}")
print(f"  Organizations          : {org_count}")


# =============================================================================
# Question 4 – Most Frequent Items
# =============================================================================

print("\n" + "=" * 60)
print("QUESTION 4 – Most Frequent Items")
print("=" * 60)

# Most frequent bi-gram
bigrams     = GetNGrams(text, 2)
bigram_freq = Counter(bigrams)
top_bigram, top_bigram_count = bigram_freq.most_common(1)[0]
print(f"\n  Most frequent bi-gram  : {top_bigram}  (count: {top_bigram_count})")

# Most frequent Noun
nouns_list = [
    token.text for token in spacy_doc if token.pos_ in ("NOUN", "PROPN")
]
noun_freq  = Counter(nouns_list)
top_noun, top_noun_count = noun_freq.most_common(1)[0]
print(f"  Most frequent Noun     : '{top_noun}'  (count: {top_noun_count})")

# Most frequent GeoPolitical Entity
gpe_list  = [ent.text for ent in spacy_doc.ents if ent.label_ == "GPE"]
gpe_freq  = Counter(gpe_list)
top_gpe, top_gpe_count = gpe_freq.most_common(1)[0]
print(f"  Most frequent GPE      : '{top_gpe}'  (count: {top_gpe_count})")

# Most frequent Person
per_list  = [ent.text for ent in spacy_doc.ents if ent.label_ == "PERSON"]
per_freq  = Counter(per_list)
top_per, top_per_count = per_freq.most_common(1)[0]
print(f"  Most frequent Person   : '{top_per}'  (count: {top_per_count})")
