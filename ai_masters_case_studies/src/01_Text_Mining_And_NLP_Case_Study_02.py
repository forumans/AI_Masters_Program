"""
Module 1: Introduction to Text Mining and NLP
Case Study 02
"""

import os
import string
import matplotlib.pyplot as plt

# =============================================================================
# 1. Import nltk module in python
# =============================================================================
import nltk

# =============================================================================
# 2. List all the documents present in NLTK 'corpora'
# =============================================================================
print("=" * 60)
print("All documents present in NLTK corpora:")
print("=" * 60)
print(dir(nltk.corpus))

# =============================================================================
# 3. Import 'twitter_samples' document
# =============================================================================
nltk.download("twitter_samples", quiet=True)
from nltk.corpus import twitter_samples

# =============================================================================
# 4. Print all the files present in 'twitter_samples' file
# =============================================================================
print("\n" + "=" * 60)
print("Files present in 'twitter_samples':")
print("=" * 60)
print(twitter_samples.fileids())

# =============================================================================
# 5. Download 'swadesh' corpora using download in nltk
# =============================================================================
print("\n" + "=" * 60)
print("Downloading 'swadesh' corpora...")
print("=" * 60)
nltk.download("swadesh")

# =============================================================================
# 6. Print all the files present in nltk Gutenberg corpora
# =============================================================================
nltk.download("gutenberg", quiet=True)
from nltk.corpus import gutenberg

print("\n" + "=" * 60)
print("Files present in NLTK Gutenberg corpora:")
print("=" * 60)
print(gutenberg.fileids())

# =============================================================================
# 7. Print the contents of 'shakespeare-macbeth.txt'
# =============================================================================
macbeth_words = gutenberg.words("shakespeare-macbeth.txt")
macbeth_raw = gutenberg.raw("shakespeare-macbeth.txt")

print("\n" + "=" * 60)
print("Contents of 'shakespeare-macbeth.txt':")
print("=" * 60)
print(macbeth_raw)

# =============================================================================
# 8. Save 'shakespeare-macbeth.txt' contents in a text file as 'Macbeth.txt'
# =============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_dir = os.path.join(project_root, "data", "01_Text_Mining_And_NLP_Case_Study_02")

macbeth_path = os.path.join(data_dir, "Macbeth.txt")
with open(macbeth_path, "w", encoding="utf-8") as f:
    f.write(macbeth_raw)
print("\nSaved 'shakespeare-macbeth.txt' content as 'Macbeth.txt'")

# =============================================================================
# 9. Count articles ('a', 'an', 'the') in 'Macbeth.txt' and save as
#    'Macbeth-ArticlesRemoved.txt'
# =============================================================================
with open(macbeth_path, "r", encoding="utf-8") as f:
    macbeth_content = f.read()

articles = {"a", "an", "the"}
words = macbeth_content.split()

article_count = sum(1 for w in words if w.lower() in articles)
print("\n" + "=" * 60)
print(f"Total number of articles ('a', 'an', 'the') in Macbeth.txt: {article_count}")
print("=" * 60)

# Remove articles (case-insensitive, preserve spacing)
filtered_words = [w for w in words if w.lower() not in articles]
macbeth_no_articles = " ".join(filtered_words)

articles_removed_path = os.path.join(data_dir, "Macbeth-ArticlesRemoved.txt")
with open(articles_removed_path, "w", encoding="utf-8") as f:
    f.write(macbeth_no_articles)
print("Saved as 'Macbeth-ArticlesRemoved.txt'")

# =============================================================================
# 10. Remove all punctuations and save as 'Macbeth-punctuationsRemoved.txt'
# =============================================================================
with open(articles_removed_path, "r", encoding="utf-8") as f:
    content = f.read()

translator = str.maketrans("", "", string.punctuation)
macbeth_no_punct = content.translate(translator)

punct_removed_path = os.path.join(data_dir, "Macbeth-punctuationsRemoved.txt")
with open(punct_removed_path, "w", encoding="utf-8") as f:
    f.write(macbeth_no_punct)
print("Removed all punctuations and saved as 'Macbeth-punctuationsRemoved.txt'")

# =============================================================================
# 11. Download 'names' corpora using download in nltk
# =============================================================================
print("\n" + "=" * 60)
print("Downloading 'names' corpora...")
print("=" * 60)
nltk.download("names")
from nltk.corpus import names

# =============================================================================
# 12. Print the total number of words present in 'names' corpora
# =============================================================================
all_names = names.words()
print("\n" + "=" * 60)
print(f"Total number of words in 'names' corpora: {len(all_names)}")
print("=" * 60)

# =============================================================================
# 13. For each alphabet, print the frequency of names present in 'names' corpora
# =============================================================================
print("\n" + "=" * 60)
print("Frequency of names for each alphabet:")
print("=" * 60)

freq_by_alpha = {}
for letter in string.ascii_uppercase:
    count = sum(1 for name in all_names if name.upper().startswith(letter))
    freq_by_alpha[letter] = count
    print(f"  {letter}: {count}")

# =============================================================================
# 14 & 15. Plot a bar graph with title and axis labels
# =============================================================================
letters = list(freq_by_alpha.keys())
counts = list(freq_by_alpha.values())

plt.figure(figsize=(14, 6))
plt.bar(letters, counts, color="steelblue", edgecolor="black")
plt.title("Frequency of Names for Each Alphabet in NLTK Names Corpus")
plt.xlabel("Alphabet")
plt.ylabel("Frequency of Names")
plt.tight_layout()

# =============================================================================
# 16. Save the bar graph as "FrequencyOfNamesOfEachAlphabet.png"
# =============================================================================
graph_path = os.path.join(data_dir, "FrequencyOfNamesOfEachAlphabet.png")
plt.savefig(graph_path, dpi=150)
plt.show()
print(f"\nBar graph saved as 'FrequencyOfNamesOfEachAlphabet.png'")
