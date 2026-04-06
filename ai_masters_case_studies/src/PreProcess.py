"""
PreProcess.py
Reusable NLP preprocessing functions for upcoming assignments.

Pipeline order (as required):
    Tokenize → RemoveStopWords → Lemmatize
"""

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)
nltk.download("omw-1.4",   quiet=True)

_lemmatizer = WordNetLemmatizer()


def Tokenize(text):
    """Tokenizes the input string and returns a list of tokens."""
    return word_tokenize(text)


def RemoveStopWords(tokens):
    """
    Accepts a list of tokens, removes stop words, and returns the filtered
    token list along with a dict of stop word frequencies.
    """
    stop_words = set(stopwords.words("english"))
    filtered   = [t for t in tokens if t.lower() not in stop_words]
    return filtered


def Lemmatize(tokens):
    """Lemmatizes a list of tokens and returns the lemmatized list."""
    return [_lemmatizer.lemmatize(token) for token in tokens]


def Refine(text):
    """
    Full preprocessing pipeline:
        1. Tokenize the input string
        2. Remove stop words from the tokens
        3. Lemmatize the remaining tokens
    Returns the final list of refined tokens.
    """
    tokens            = Tokenize(text)
    tokens_no_stop    = RemoveStopWords(tokens)
    lemmatized_tokens = Lemmatize(tokens_no_stop)
    return lemmatized_tokens
