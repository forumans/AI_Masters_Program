"""
Module 4: Text Classification 1
Case Study 01 – Corpus building and text vectorization
"""

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from Corpus import MakeCorpus
from Vectorization import (
    PresenceAbsenceVectorization,
    CountVectorization,
    TFIDFVectorization,
)


def _print_vectors(strings, vectors, corpus):
    """Pretty-print strings alongside their vectors."""
    print(f"\n  Corpus: {corpus}\n")
    for i, (s, v) in enumerate(zip(strings, vectors), 1):
        print(f"  S{i}: {s!r}")
        print(f"       {v}\n")


# =============================================================================
# Question 1 – Corpus
# =============================================================================

print("=" * 60)
print("QUESTION 1 – Build Corpus from 3 sentences")
print("=" * 60)

strings_q1 = []
for i in range(1, 4):
    strings_q1.append(input(f"  Enter sentence {i}: "))

corpus_q1 = MakeCorpus(strings_q1)
print(f"\nCorpus (union of all words, order of first appearance):")
print(f"  {corpus_q1}")


# =============================================================================
# Question 2 – Presence / Absence Vectorization
# =============================================================================

print("\n" + "=" * 60)
print("QUESTION 2 – Presence / Absence Vectorization")
print("=" * 60)

strings_q2 = []
for i in range(1, 4):
    strings_q2.append(input(f"  Enter sentence {i}: "))

corpus_q2  = MakeCorpus(strings_q2)
vectors_q2 = PresenceAbsenceVectorization(strings_q2)

print("\nPresence / Absence vectors (1 = word present, 0 = absent):")
_print_vectors(strings_q2, vectors_q2, corpus_q2)


# =============================================================================
# Question 3 – Count Vectorization
# =============================================================================

print("=" * 60)
print("QUESTION 3 – Count Vectorization")
print("=" * 60)

strings_q3 = []
for i in range(1, 4):
    strings_q3.append(input(f"  Enter sentence {i}: "))

corpus_q3  = MakeCorpus(strings_q3)
vectors_q3 = CountVectorization(strings_q3)

print("\nCount vectors (count of each corpus word in the string):")
_print_vectors(strings_q3, vectors_q3, corpus_q3)


# =============================================================================
# Question 4 – TF-IDF Vectorization
# =============================================================================

print("=" * 60)
print("QUESTION 4 – TF-IDF Vectorization")
print("=" * 60)

strings_q4 = []
for i in range(1, 4):
    strings_q4.append(input(f"  Enter sentence {i}: "))

vectors_q4 = TFIDFVectorization(strings_q4)

print("\nTF-IDF vectors:")
for i, (s, v) in enumerate(zip(strings_q4, vectors_q4), 1):
    # Round for readability
    v_rounded = [round(x, 4) for x in v]
    print(f"  S{i}: {s!r}")
    print(f"       {v_rounded}\n")
