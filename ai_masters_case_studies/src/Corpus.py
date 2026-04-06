"""
Corpus.py
Reusable corpus-building utility for NLP pipelines.

Exports:
    MakeCorpus(strings) -> list[str]
"""


def MakeCorpus(strings):
    """
    Takes a list of strings and returns the corpus: a list of all unique words
    (union) across all strings, preserving order of first appearance.

    Example:
        MakeCorpus(["India won the match",
                    "England won the cricket match",
                    "Australia won the final match"])
        -> ['India', 'won', 'the', 'match', 'England', 'cricket',
            'Australia', 'final']
    """
    seen   = set()
    corpus = []
    for s in strings:
        for word in str(s).split():
            if word not in seen:
                seen.add(word)
                corpus.append(word)
    return corpus
