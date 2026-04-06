"""
Vectorization.py
Reusable text vectorization functions for NLP pipelines.

Exports:
    PresenceAbsenceVectorization(strings) -> list[list[int]]
    CountVectorization(strings)           -> list[list[int]]
    TFIDFVectorization(strings)           -> list[list[float]]
"""

import os
import sys
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Corpus import MakeCorpus


def PresenceAbsenceVectorization(strings):
    """
    Takes a list of strings and returns a list of binary (0/1)
    presence-absence vectors based on the combined corpus.

    Example:
        strings = ["India won the match",
                   "England won the cricket match",
                   "Australia won the final match"]
        corpus  = ['India','won','the','match','England','cricket',
                   'Australia','final']
        -> [[1,1,1,1,0,0,0,0],
            [0,1,1,1,1,1,0,0],
            [0,1,1,1,0,0,1,1]]
    """
    corpus = MakeCorpus(strings)
    vectors = []
    for s in strings:
        words  = set(str(s).split())
        vector = [1 if w in words else 0 for w in corpus]
        vectors.append(vector)
    return vectors


def CountVectorization(strings):
    """
    Takes a list of strings and returns a list of count vectors based on
    the combined corpus (term frequency per document).

    Example:
        strings = ["A lives with B. A plays with C.",
                   "B lives with C. B plays with D",
                   "C lives with D. C plays with A"]
        corpus  = ['A','lives','with','B.','plays','C.','B','D','D.','A']
        -> count of each corpus word in each string
    """
    corpus = MakeCorpus(strings)
    vectors = []
    for s in strings:
        counts = Counter(str(s).split())
        vector = [counts.get(w, 0) for w in corpus]
        vectors.append(vector)
    return vectors


def TFIDFVectorization(strings):
    """
    Takes a list of strings and returns a list of TF-IDF vectors using
    sklearn's TfidfVectorizer.

    Returns a list of lists (one dense vector per input string).
    """
    vectorizer  = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([str(s) for s in strings])
    return tfidf_matrix.toarray().tolist()
