"""
Module 2: Extracting, Cleaning and Preprocessing Data
Case Study 01
"""

import os
import csv
import string
from collections import Counter

import nltk
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

nltk.download("punkt",        quiet=True)
nltk.download("punkt_tab",    quiet=True)
nltk.download("stopwords",    quiet=True)
nltk.download("wordnet",      quiet=True)
nltk.download("omw-1.4",      quiet=True)

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_dir     = os.path.join(project_root, "data", "02_Extracting_Cleaning_Data_Case_Study_01")


# =============================================================================
# Question 1 – Tokenize
# =============================================================================

def Tokenize(text):
    """Tokenizes the input string and returns a list of tokens."""
    tokens = word_tokenize(text)
    return tokens


print("=" * 60)
print("QUESTION 1 – Tokenization")
print("=" * 60)

user_input_q1 = input("Enter a string for tokenization: ")
tokens_q1 = Tokenize(user_input_q1)

# Token frequencies
freq_q1 = Counter(tokens_q1)
print("\nTokens and their frequencies:")
for token, freq in freq_q1.most_common():
    print(f"  {token!r:20s} : {freq}")

# 5 least occurring tokens
print("\n5 least occurring tokens:")
least_5 = freq_q1.most_common()[:-6:-1]          # last 5 in ascending order
for token, freq in least_5:
    print(f"  {token!r:20s} : {freq}")


# =============================================================================
# Question 2 – RemoveStopWords
# =============================================================================

def RemoveStopWords(tokens):
    """
    Removes stop words from a list of tokens.
    Returns the filtered token list and a Counter of removed stop words.
    """
    stop_words = set(stopwords.words("english"))
    stop_word_freq = Counter(t.lower() for t in tokens if t.lower() in stop_words)
    filtered = [t for t in tokens if t.lower() not in stop_words]
    return filtered, stop_word_freq


print("\n" + "=" * 60)
print("QUESTION 2 – Remove Stop Words")
print("=" * 60)

user_input_q2 = input("Enter a string to remove stop words: ")
tokens_q2     = Tokenize(user_input_q2)
filtered_q2, stop_freq_q2 = RemoveStopWords(tokens_q2)

print("\nString after removing stop words:")
print(" ".join(filtered_q2))

print("\nStop word frequencies in the input string:")
for word, freq in stop_freq_q2.most_common():
    print(f"  {word!r:20s} : {freq}")

# Bar graph – stop words and their frequencies
if stop_freq_q2:
    words_sw  = list(stop_freq_q2.keys())
    counts_sw = list(stop_freq_q2.values())

    plt.figure(figsize=(max(8, len(words_sw)), 5))
    plt.bar(words_sw, counts_sw, color="coral", edgecolor="black")
    plt.title("Stop Word Frequencies")
    plt.xlabel("Stop Words")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "StopWordFrequencies.png"), dpi=150)
    plt.show()
    print("\nBar graph saved as 'StopWordFrequencies.png'")
else:
    print("\nNo stop words found in the input string.")


# =============================================================================
# Question 3 – Lemmatize & Stemmed
# =============================================================================

lemmatizer = WordNetLemmatizer()
stemmer    = PorterStemmer()


def Lemmatize(tokens):
    """Lemmatizes a list of tokens and returns the lemmatized list."""
    return [lemmatizer.lemmatize(token) for token in tokens]


def Stemmed(tokens):
    """Stems a list of tokens using the Porter Stemmer and returns the stemmed list."""
    return [stemmer.stem(token) for token in tokens]


print("\n" + "=" * 60)
print("QUESTION 3 – Lemmatization and Stemming")
print("=" * 60)

user_input_q3  = input("Enter a string for lemmatization and stemming: ")
tokens_q3      = Tokenize(user_input_q3)
lemmatized_q3  = Lemmatize(tokens_q3)
stemmed_q3     = Stemmed(tokens_q3)

print(f"\n{'Original':<25} {'Lemmatized':<25} {'Stemmed'}")
print("-" * 70)
for orig, lem, stem in zip(tokens_q3, lemmatized_q3, stemmed_q3):
    print(f"{orig:<25} {lem:<25} {stem}")

# Save to CSV
csv_path = os.path.join(data_dir, "LemmatizedStemmedWords.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Original Word", "Lemmatized Form", "Stemmed Form"])
    writer.writerows(zip(tokens_q3, lemmatized_q3, stemmed_q3))
print(f"\nResults saved to 'LemmatizedStemmedWords.csv'")


# =============================================================================
# Question 4 – Import and demonstrate the PreProcess module
# =============================================================================

print("\n" + "=" * 60)
print("QUESTION 4 – PreProcess Module (Refine pipeline)")
print("=" * 60)

import sys
sys.path.insert(0, script_dir)
from PreProcess import Refine

user_input_q4 = input("Enter a string to run through the full Refine pipeline: ")
refined_result = Refine(user_input_q4)

print("\nRefined output (Tokenized -> Stop Words Removed -> Lemmatized):")
print(refined_result)
