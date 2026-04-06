"""
Module 3: Analyzing Sentence Structure
Case Study 01 - FIFAWorldCup2018.txt Analysis
"""

import os
import re
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, RegexpParser

nltk.download("punkt",                       quiet=True)
nltk.download("punkt_tab",                   quiet=True)
nltk.download("averaged_perceptron_tagger",  quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_dir     = os.path.join(project_root, "data", "03_Analyzing_Sentence_Structure_Case_Study_01")
os.makedirs(data_dir, exist_ok=True)

txt_path = os.path.join(data_dir, "FIFAWorldCup2018.txt")

with open(txt_path, "r", encoding="cp1252") as f:
    text = f.read().strip()

print("=" * 60)
print("FIFAWorldCup2018.txt loaded successfully.")
print(f"Total characters : {len(text)}")
print(f"Total words      : {len(text.split())}")
print("=" * 60)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _tagged(string):
    """Tokenise and POS-tag the string."""
    return pos_tag(word_tokenize(string))


def _first_sentence_tagged(string):
    """Tokenise and POS-tag the first sentence of the string."""
    return pos_tag(word_tokenize(sent_tokenize(string)[0]))


# =============================================================================
# Question 1 – N Most Frequent POS Items
# =============================================================================

def GetNMostFrequentNouns(string, n):
    """Returns the N most frequent nouns (NN, NNS, NNP, NNPS) from the string."""
    nouns = [w for w, t in _tagged(string) if t.startswith("NN")]
    return Counter(nouns).most_common(n)


def GetNMostFrequentVerbs(string, n):
    """Returns the N most frequent verbs (VB*) from the string."""
    verbs = [w for w, t in _tagged(string) if t.startswith("VB")]
    return Counter(verbs).most_common(n)


def GetNMostFrequentDelimiters(string, n):
    """Returns the N most frequent delimiters/determiners (DT) from the string."""
    delims = [w for w, t in _tagged(string) if t == "DT"]
    return Counter(delims).most_common(n)


def GetNMostFrequentPrepositions(string, n):
    """Returns the N most frequent prepositions (IN) from the string."""
    preps = [w for w, t in _tagged(string) if t == "IN"]
    return Counter(preps).most_common(n)


print("\n" + "=" * 60)
print("QUESTION 1 – N Most Frequent POS Items (n=5)")
print("=" * 60)

n = 5

print(f"\nTop {n} most frequent Nouns:")
for word, count in GetNMostFrequentNouns(text, n):
    print(f"  {word!r:20s} : {count}")

print(f"\nTop {n} most frequent Verbs:")
for word, count in GetNMostFrequentVerbs(text, n):
    print(f"  {word!r:20s} : {count}")

print(f"\nTop {n} most frequent Delimiters:")
for word, count in GetNMostFrequentDelimiters(text, n):
    print(f"  {word!r:20s} : {count}")

print(f"\nTop {n} most frequent Prepositions:")
for word, count in GetNMostFrequentPrepositions(text, n):
    print(f"  {word!r:20s} : {count}")


# =============================================================================
# Question 2 – Syntax Tree for First Sentence
# =============================================================================

def PrintSyntaxTree(string):
    """Prints the first sentence in the string along with its chunked syntax tree."""
    first = sent_tokenize(string)[0]
    print(f"\nFirst sentence:\n  {first}")

    tagged = pos_tag(word_tokenize(first))

    grammar = r"""
        NP: {<DT>?<JJ>*<NN.*>+}
        VP: {<VB.*><NP|IN>*}
        PP: {<IN><NP>}
    """
    cp   = RegexpParser(grammar)
    tree = cp.parse(tagged)

    print("\nSyntax Tree:")
    print(tree)


print("\n" + "=" * 60)
print("QUESTION 2 – Syntax Tree (first sentence)")
print("=" * 60)
PrintSyntaxTree(text)


# =============================================================================
# Question 3 – Regular Expression Functions
# =============================================================================

def TextAfterRemovingPunctuations(string):
    """Returns text with all punctuation characters removed."""
    return re.sub(r'[^\w\s]', '', string)


def TextAfterRemovingDigits(string):
    """Returns text with all digit characters removed."""
    return re.sub(r'\d+', '', string)


def AllCapitalizedWordsFromText(string):
    """Returns all words that begin with a capital letter."""
    return re.findall(r'\b[A-Z][a-zA-Z]*\b', string)


def AllEmailsFromText(string):
    """Returns all email addresses found in the string."""
    return re.findall(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', string)


print("\n" + "=" * 60)
print("QUESTION 3 – Regular Expression Functions")
print("=" * 60)

no_punct = TextAfterRemovingPunctuations(text)
print("\nText after removing punctuations (first 200 chars):")
print(no_punct[:200])

no_digits = TextAfterRemovingDigits(text)
print("\nText after removing digits (first 200 chars):")
print(no_digits[:200])

cap_words = AllCapitalizedWordsFromText(text)
print(f"\nCapitalized words ({len(cap_words)} found, first 20 shown):")
print(cap_words[:20])

emails = AllEmailsFromText(text)
print(f"\nEmails found: {emails if emails else '(none found in this text)'}")


# =============================================================================
# Question 4 – Chunking (all functions run on first sentence)
# =============================================================================

def _parse(string, grammar):
    """Tokenise, POS-tag first sentence and apply RegexpParser with given grammar."""
    tagged = _first_sentence_tagged(string)
    return RegexpParser(grammar).parse(tagged)


def ChunkingVer1(string):
    """Phrases having Proper Nouns (NNP) followed by Verbs."""
    return _parse(string, r"CHUNK: {<NNP>+<VB.*>}")


def ChunkingVer2(string):
    """Verb Phrases having Verbs followed by Adjectives."""
    return _parse(string, r"CHUNK: {<VB.*><JJ.*>+}")


def ChunkingVer3(string):
    """Noun Phrases having Determiners followed by Nouns."""
    return _parse(string, r"CHUNK: {<DT><NN.*>+}")


def ChunkingVer4(string):
    """Verb Phrases having Verbs followed by Adverbs."""
    return _parse(string, r"CHUNK: {<VB.*><RB.*>+}")


def ChunkingVer5(string):
    """Phrases having Delimiter, Adjectives and Nouns in the respective order."""
    return _parse(string, r"CHUNK: {<DT><JJ.*>*<NN.*>+}")


def ChunkingVer6(string):
    """Noun Phrases having Nouns and Adjectives, terminated with Nouns."""
    return _parse(string, r"CHUNK: {<NN.*>+<JJ.*>*<NN.*>+}")


print("\n" + "=" * 60)
print("QUESTION 4 – Chunking (first sentence of FIFAWorldCup2018.txt)")
print("=" * 60)

print("\nChunkingVer1 – Proper Nouns followed by Verbs:")
print(ChunkingVer1(text))

print("\nChunkingVer2 – Verbs followed by Adjectives:")
print(ChunkingVer2(text))

print("\nChunkingVer3 – Determiners followed by Nouns:")
print(ChunkingVer3(text))

print("\nChunkingVer4 – Verbs followed by Adverbs:")
print(ChunkingVer4(text))

print("\nChunkingVer5 – Delimiter, Adjectives and Nouns:")
print(ChunkingVer5(text))

print("\nChunkingVer6 – Nouns and Adjectives, terminated with Nouns:")
print(ChunkingVer6(text))


# =============================================================================
# Question 5 – Context-Free Grammar (CFG)
# =============================================================================

print("\n" + "=" * 60)
print("QUESTION 5 – Context-Free Grammar (CFG)")
print("=" * 60)

# 2 most frequent of each POS type from the full text
top2_nouns = [w for w, _ in GetNMostFrequentNouns(text, 2)]
top2_verbs = [w for w, _ in GetNMostFrequentVerbs(text, 2)]
top2_delims = [w for w, _ in GetNMostFrequentDelimiters(text, 2)]
top2_preps  = [w for w, _ in GetNMostFrequentPrepositions(text, 2)]

print(f"\nTop 2 Nouns        : {top2_nouns}")
print(f"Top 2 Verbs        : {top2_verbs}")
print(f"Top 2 Delimiters   : {top2_delims}")
print(f"Top 2 Prepositions : {top2_preps}")

# Format terminal rules
def _terminals(words):
    return " | ".join(f"'{w}'" for w in words)

cfg_content = f"""# Context-Free Grammar generated from FIFAWorldCup2018.txt
# Terminals are the 2 most frequent words of each POS type.

# Structural rules
S  -> NP VP
VP -> V NP
VP -> V NP PP
NP -> DT N
PP -> P NP

# Lexical rules (2 most frequent words per category)
N  -> {_terminals(top2_nouns)}
V  -> {_terminals(top2_verbs)}
DT -> {_terminals(top2_delims)}
P  -> {_terminals(top2_preps)}
"""

cfg_path = os.path.join(data_dir, "CFG.txt")
with open(cfg_path, "w", encoding="utf-8") as f:
    f.write(cfg_content)

print("\nGenerated CFG:")
print(cfg_content)
print(f"CFG saved as 'CFG.txt'")
