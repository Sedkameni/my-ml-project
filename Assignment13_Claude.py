"""
PART 1: TEXT PREPROCESSING EXERCISE
IMDB Movie Reviews Dataset

This script performs comprehensive text preprocessing including:
1. Data loading and exploration
2. Text cleaning (HTML removal, contractions, special characters)
3. Tokenization
4. Stopword removal (preserving negations)
5. Lemmatization
6. Data validation and quality checks
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Download necessary NLTK resources
print("Downloading NLTK resources...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

print("="*80)
print("TEXT PREPROCESSING PIPELINE - IMDB MOVIE REVIEWS")
print("="*80)

# =============================================================================
# STEP 1: LOAD THE DATASET
# =============================================================================
print("\n[STEP 1] Loading Dataset...")
print("-" * 80)

df = pd.read_csv('IMDB Dataset.csv')
original_size = len(df)

print(f"✓ Dataset loaded successfully")
print(f"  Total reviews: {len(df)}")
print(f"  Columns: {list(df.columns)}")

# Display sample data
print("\nSample reviews (first 2):")
for idx in range(2):
    print(f"\n  Review {idx+1}:")
    print(f"  Sentiment: {df.iloc[idx]['sentiment']}")
    print(f"  Text: {df.iloc[idx]['review'][:150]}...")

# Check sentiment distribution
print("\nSentiment Distribution:")
sentiment_counts = df['sentiment'].value_counts()
for sentiment, count in sentiment_counts.items():
    percentage = (count / len(df)) * 100
    print(f"  {sentiment}: {count} ({percentage:.1f}%)")

# =============================================================================
# STEP 2: TEXT CLEANING (Pre-processing before tokenization)
# =============================================================================
print("\n[STEP 2] Text Cleaning...")
print("-" * 80)

# 2.1 Expand contractions
print("✓ Expanding contractions...")
contractions_dict = {
    "don't": "do not",
    "won't": "will not",
    "can't": "cannot",
    "it's": "it is",
    "i'm": "i am",
    "you're": "you are",
    "didn't": "did not",
    "doesn't": "does not",
    "hasn't": "has not",
    "haven't": "have not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'll": "i will",
    "you'll": "you will",
    "we'll": "we will",
    "they'll": "they will"
}

for contraction, expansion in contractions_dict.items():
    df['review'] = df['review'].str.replace(contraction, expansion, regex=False)

# 2.2 Remove HTML tags (CRITICAL for IMDB dataset!)
print("✓ Removing HTML tags...")
df['review'] = df['review'].apply(lambda x: re.sub(r'<.*?>', '', x))

# Show example of HTML removal
print("  Example: '<br />' tags removed from reviews")

# 2.3 Convert to lowercase
print("✓ Converting to lowercase...")
df['review'] = df['review'].str.lower()

# 2.4 Remove special characters and digits (keep only letters and spaces)
print("✓ Removing special characters and digits...")
df['review'] = df['review'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))

# 2.5 Remove extra whitespace
print("✓ Removing extra whitespace...")
df['review'] = df['review'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())

# Show example of cleaned text
print("\nExample of cleaned text:")
print(f"  Original length: ~{len(df.iloc[0]['review'])} characters")
print(f"  Cleaned text: {df.iloc[0]['review'][:150]}...")

# =============================================================================
# STEP 3: TOKENIZATION
# =============================================================================
print("\n[STEP 3] Tokenization...")
print("-" * 80)

df['tokens'] = df['review'].apply(word_tokenize)

# Show tokenization statistics
token_counts = df['tokens'].apply(len)
print(f"✓ Tokenization complete")
print(f"  Average tokens per review: {token_counts.mean():.1f}")
print(f"  Min tokens: {token_counts.min()}")
print(f"  Max tokens: {token_counts.max()}")

print("\nExample tokenization (first review):")
print(f"  Tokens: {df.iloc[0]['tokens'][:15]}...")

# =============================================================================
# STEP 4: STOPWORD REMOVAL (Preserving negations!)
# =============================================================================
print("\n[STEP 4] Stopword Removal...")
print("-" * 80)

# Create custom stopword list that preserves negations
stop_words = set(stopwords.words('english'))
negation_words = {
    'not', 'no', 'nor', 'never', 'neither', 'nobody',
    'nothing', 'nowhere', 'none', 'cannot'
}

print(f"✓ Standard stopwords: {len(stop_words)}")
print(f"✓ Negation words preserved: {len(negation_words)}")

# Remove negation words from stopwords
custom_stop_words = stop_words - negation_words

print(f"✓ Custom stopwords: {len(custom_stop_words)}")
print(f"  Preserved negations: {negation_words}")

# Apply stopword removal
df['filtered_tokens'] = df['tokens'].apply(
    lambda x: [word for word in x if word not in custom_stop_words and len(word) > 2]
)

# Show filtering statistics
filtered_counts = df['filtered_tokens'].apply(len)
reduction_pct = ((token_counts.mean() - filtered_counts.mean()) / token_counts.mean()) * 100

print(f"\n✓ Stopword removal complete")
print(f"  Average tokens after filtering: {filtered_counts.mean():.1f}")
print(f"  Token reduction: {reduction_pct:.1f}%")

print("\nExample after stopword removal (first review):")
print(f"  Filtered tokens: {df.iloc[0]['filtered_tokens'][:15]}...")

# =============================================================================
# STEP 5: STEMMING (Alternative approach)
# =============================================================================
print("\n[STEP 5] Stemming...")
print("-" * 80)

stemmer = PorterStemmer()
df['stemmed_tokens'] = df['filtered_tokens'].apply(
    lambda x: [stemmer.stem(word) for word in x]
)

print(f"✓ Stemming complete using Porter Stemmer")
print("\nStemming examples:")
sample_words = ['movies', 'running', 'better', 'amazing', 'disappointing']
for word in sample_words:
    stemmed = stemmer.stem(word)
    print(f"  {word} → {stemmed}")

# =============================================================================
# STEP 6: LEMMATIZATION (Recommended approach)
# =============================================================================
print("\n[STEP 6] Lemmatization...")
print("-" * 80)

lemmatizer = WordNetLemmatizer()
df['lemmatized_tokens'] = df['filtered_tokens'].apply(
    lambda x: [lemmatizer.lemmatize(word) for word in x]
)

print(f"✓ Lemmatization complete using WordNet Lemmatizer")
print("\nLemmatization examples:")
for word in sample_words:
    lemmatized = lemmatizer.lemmatize(word)
    print(f"  {word} → {lemmatized}")

print("\nComparison (Stemming vs Lemmatization) on sample tokens:")
sample_filtered = df.iloc[0]['filtered_tokens'][:10]
sample_stemmed = df.iloc[0]['stemmed_tokens'][:10]
sample_lemmatized = df.iloc[0]['lemmatized_tokens'][:10]

for i in range(min(10, len(sample_filtered))):
    print(f"  Original: {sample_filtered[i]:15} | Stemmed: {sample_stemmed[i]:15} | Lemmatized: {sample_lemmatized[i]:15}")

# =============================================================================
# STEP 7: DATA QUALITY CHECKS
# =============================================================================
print("\n[STEP 7] Data Quality Checks...")
print("-" * 80)

# Check for empty reviews after preprocessing
df['word_count'] = df['lemmatized_tokens'].apply(len)
empty_reviews = len(df[df['word_count'] < 3])

print(f"✓ Reviews with less than 3 tokens: {empty_reviews}")

if empty_reviews > 0:
    print(f"  Removing {empty_reviews} reviews...")
    df = df[df['word_count'] >= 3]
    df = df.reset_index(drop=True)
    print(f"  ✓ Dataset size after filtering: {len(df)}")
    print(f"  Reviews removed: {original_size - len(df)} ({((original_size - len(df))/original_size)*100:.2f}%)")

# Check final sentiment distribution
print("\nFinal Sentiment Distribution:")
final_sentiment_counts = df['sentiment'].value_counts()
for sentiment, count in final_sentiment_counts.items():
    percentage = (count / len(df)) * 100
    print(f"  {sentiment}: {count} ({percentage:.1f}%)")

# Calculate balance ratio
balance_ratio = final_sentiment_counts.min() / final_sentiment_counts.max()
print(f"\nDataset Balance Ratio: {balance_ratio:.3f}")
if balance_ratio < 0.9:
    print("  ⚠ Warning: Dataset shows class imbalance")
else:
    print("  ✓ Dataset is well balanced")

# =============================================================================
# STEP 8: VISUALIZATIONS
# =============================================================================
print("\n[STEP 8] Generating Visualizations...")
print("-" * 80)

fig = plt.figure(figsize=(16, 10))

# 1. Sentiment Distribution
ax1 = plt.subplot(2, 3, 1)
sentiment_counts = df['sentiment'].value_counts()
colors = ['#2ecc71', '#e74c3c']
ax1.pie(sentiment_counts.values, labels=sentiment_counts.index,
        autopct='%1.1f%%', colors=colors, startangle=90)
ax1.set_title('Sentiment Distribution', fontsize=14, fontweight='bold')

# 2. Token Count Distribution (Before and After Filtering)
ax2 = plt.subplot(2, 3, 2)
original_token_counts = df['tokens'].apply(len)
filtered_token_counts = df['filtered_tokens'].apply(len)
ax2.hist(original_token_counts, bins=50, alpha=0.5, label='Before Stopword Removal', color='blue')
ax2.hist(filtered_token_counts, bins=50, alpha=0.5, label='After Stopword Removal', color='green')
ax2.set_xlabel('Number of Tokens', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title('Token Count Distribution', fontsize=14, fontweight='bold')
ax2.legend()

# 3. Average Token Length by Sentiment
ax3 = plt.subplot(2, 3, 3)
sentiment_groups = df.groupby('sentiment')['word_count'].mean()
bars = ax3.bar(sentiment_groups.index, sentiment_groups.values, color=['#2ecc71', '#e74c3c'])
ax3.set_ylabel('Average Token Count', fontsize=12)
ax3.set_title('Average Tokens by Sentiment', fontsize=14, fontweight='bold')
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}', ha='center', va='bottom')

# 4. Most Common Words (After Preprocessing)
ax4 = plt.subplot(2, 3, 4)
all_tokens = [token for tokens in df['lemmatized_tokens'] for token in tokens]
most_common = Counter(all_tokens).most_common(20)
words, counts = zip(*most_common)
ax4.barh(range(len(words)), counts, color='skyblue')
ax4.set_yticks(range(len(words)))
ax4.set_yticklabels(words)
ax4.set_xlabel('Frequency', fontsize=12)
ax4.set_title('Top 20 Most Common Words', fontsize=14, fontweight='bold')
ax4.invert_yaxis()

# 5. Most Common Positive Words
ax5 = plt.subplot(2, 3, 5)
positive_tokens = [token for idx, tokens in enumerate(df['lemmatized_tokens'])
                  if df.iloc[idx]['sentiment'] == 'positive' for token in tokens]
most_common_pos = Counter(positive_tokens).most_common(15)
words_pos, counts_pos = zip(*most_common_pos)
ax5.barh(range(len(words_pos)), counts_pos, color='#2ecc71')
ax5.set_yticks(range(len(words_pos)))
ax5.set_yticklabels(words_pos)
ax5.set_xlabel('Frequency', fontsize=12)
ax5.set_title('Top 15 Positive Review Words', fontsize=14, fontweight='bold')
ax5.invert_yaxis()

# 6. Most Common Negative Words
ax6 = plt.subplot(2, 3, 6)
negative_tokens = [token for idx, tokens in enumerate(df['lemmatized_tokens'])
                  if df.iloc[idx]['sentiment'] == 'negative' for token in tokens]
most_common_neg = Counter(negative_tokens).most_common(15)
words_neg, counts_neg = zip(*most_common_neg)
ax6.barh(range(len(words_neg)), counts_neg, color='#e74c3c')
ax6.set_yticks(range(len(words_neg)))
ax6.set_yticklabels(words_neg)
ax6.set_xlabel('Frequency', fontsize=12)
ax6.set_title('Top 15 Negative Review Words', fontsize=14, fontweight='bold')
ax6.invert_yaxis()

plt.tight_layout()
plt.savefig('preprocessing_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Visualizations saved as 'preprocessing_analysis.png'")
plt.show()

# =============================================================================
# STEP 9: SAVE PREPROCESSED DATA
# =============================================================================
print("\n[STEP 9] Saving Preprocessed Data...")
print("-" * 80)

# Create final preprocessed text column (joining lemmatized tokens)
df['preprocessed_text'] = df['lemmatized_tokens'].apply(lambda x: ' '.join(x))

# Save to CSV
output_columns = ['review', 'sentiment', 'preprocessed_text', 'lemmatized_tokens', 'word_count']
df[output_columns].to_csv('IMDB_preprocessed.csv', index=False)
print("✓ Preprocessed data saved as 'IMDB_preprocessed.csv'")

# =============================================================================
# STEP 10: SUMMARY REPORT
# =============================================================================
print("\n" + "="*80)
print("PREPROCESSING SUMMARY REPORT")
print("="*80)

print("\n1. DATASET STATISTICS")
print("-" * 80)
print(f"  Original dataset size: {original_size:,} reviews")
print(f"  Final dataset size: {len(df):,} reviews")
print(f"  Reviews removed: {original_size - len(df):,} ({((original_size - len(df))/original_size)*100:.2f}%)")

print("\n2. PREPROCESSING STEPS COMPLETED")
print("-" * 80)
print("  ✓ Contraction expansion (e.g., don't → do not)")
print("  ✓ HTML tag removal (e.g., <br /> tags)")
print("  ✓ Lowercase conversion")
print("  ✓ Special character and digit removal")
print("  ✓ Tokenization")
print("  ✓ Custom stopword removal (preserved negations)")
print("  ✓ Stemming (Porter Stemmer)")
print("  ✓ Lemmatization (WordNet Lemmatizer)")

print("\n3. TOKEN STATISTICS")
print("-" * 80)
print(f"  Average tokens (original): {df['tokens'].apply(len).mean():.1f}")
print(f"  Average tokens (after filtering): {df['filtered_tokens'].apply(len).mean():.1f}")
print(f"  Average tokens (final): {df['word_count'].mean():.1f}")
print(f"  Token reduction: {((df['tokens'].apply(len).mean() - df['word_count'].mean()) / df['tokens'].apply(len).mean() * 100):.1f}%")

print("\n4. SENTIMENT DISTRIBUTION")
print("-" * 80)
for sentiment, count in df['sentiment'].value_counts().items():
    percentage = (count / len(df)) * 100
    print(f"  {sentiment}: {count:,} ({percentage:.1f}%)")

print("\n5. MOST COMMON WORDS (Overall)")
print("-" * 80)
top_10 = Counter(all_tokens).most_common(10)
for word, count in top_10:
    print(f"  {word}: {count:,}")

print("\n6. KEY INSIGHTS")
print("-" * 80)
print(f"  • Dataset is {'balanced' if balance_ratio >= 0.9 else 'imbalanced'} (ratio: {balance_ratio:.3f})")
print(f"  • Average review length: {df['word_count'].mean():.1f} tokens")
print(f"  • Negation words preserved for sentiment accuracy")
print(f"  • Lemmatization preferred over stemming for better word quality")

print("\n7. FILES GENERATED")
print("-" * 80)
print("  ✓ IMDB_preprocessed.csv - Preprocessed dataset")
print("  ✓ preprocessing_analysis.png - Visualization dashboard")

print("\n" + "="*80)
print("PREPROCESSING COMPLETE!")
print("="*80)

# Display first few rows of preprocessed data
print("\nSample of Preprocessed Data:")
print(df[['sentiment', 'preprocessed_text', 'word_count']].head(3))