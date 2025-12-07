### PArt 1: Text Preprocessing Exercise:

## 1 - Choose a text dataset of your choice (e.g., movie reviews, news articles, tweets)
# ## and perform text preprocessing using Python. You may use the IMDB dataset attached at the bottom of this page.
## 2 - Implement tokenization, remove stopwords, and perform either stemming or lemmatization.
## 3 - Document each step of the preprocessing process and provide code snippets along with explanations.
## 4 - Discuss any challenges encountered during preprocessing and how you addressed them.

# Import Libraries and Necessary Resources

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from textblob import TextBlob

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('stopwords')
# Download required NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from nltk.stem import WordNetLemmatizer
import re

## II - Load the dataset
# 1 - Load the dataset into Python using pandas.
df = pd.read_csv('IMDB Dataset.csv')

## 2 - Convert to lowercase
df['review'] = df['review'].str.lower()

## 3 - Tokenize the text into words.
# Tokenization
df['tokens'] = df['review'].apply(word_tokenize)

## 4 - Remove stopwords to filter out common words with little meaning.
# Remove stopwords
stop_words = set(stopwords.words('english'))
df['filtered_tokens'] = df['tokens'].apply(lambda x: [word for word in x if word not in stop_words])

## 5 - Perform stemming or lemmatization to reduce words to their base forms.
# Stemming
stemmer = PorterStemmer()
df['stemmed_tokens'] = df['filtered_tokens'].apply(lambda x: [stemmer.stem(word) for word in x])
print(df.head())

## =======================================
### Part 2: Sentiment Analysis Task:
## =======================================

## 1 - Conduct sentiment analysis on a text dataset of customer reviews or social media posts using Python.
## 2 - Preprocess the text data by tokenizing, removing stopwords, and performing either stemming or lemmatization.
## 3 - Implement a sentiment analysis model using a suitable algorithm (e.g., Naive Bayes, Support Vector Machines) and evaluate its performance.
## 4 - Analyze the sentiment distribution of the dataset and identify any insights or trends.
## 5 - Present your findings in a report format, including visualizations and explanations of the sentiment analysis results.


# STEP 1: CREATE SAMPLE DATA
print("\nLoading dataset...")

reviews = [
              ("This product is amazing! Exceeded all my expectations.", "positive"),
              ("Terrible quality. Waste of money.", "negative"),
              ("It's okay, nothing special.", "neutral"),
              ("Absolutely love it! Best purchase ever.", "positive"),
              ("Disappointing. Doesn't work as advertised.", "negative"),
              ("Good product, fast shipping, happy customer!", "positive"),
              ("Not what I expected. Very disappointed.", "negative"),
              ("Average product. Does the job.", "neutral"),
              ("Outstanding quality and great customer service!", "positive"),
              ("Broken on arrival. Very frustrating.", "negative"),
              ("Decent product for the price.", "neutral"),
              ("Highly recommend! Five stars!", "positive"),
              ("Poor quality materials. Not durable.", "negative"),
              ("It works fine. No complaints.", "neutral"),
              ("Fantastic! Will buy again.", "positive"),
              ("Complete waste of time and money.", "negative"),
              ("Acceptable but could be better.", "neutral"),
              ("Superb quality! Impressed.", "positive"),
              ("Defective product. Requesting refund.", "negative"),
              ("Standard quality, meets expectations.", "neutral"),
          ] * 20  # Multiply to get more samples

df = pd.DataFrame(reviews, columns=['review', 'sentiment'])
print(f"Dataset loaded: {len(df)} reviews")

# STEP 2: TEXT PREPROCESSING
print("\nPreprocessing text data...")

# Initialize preprocessing tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Preprocess each review
processed_texts = []

for text in df['review']:
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords and lemmatize
    processed_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words and len(token) > 2
    ]

    processed_text = ' '.join(processed_tokens)
    processed_texts.append(processed_text)

# Add processed text to dataframe
df['processed_text'] = processed_texts

# Remove empty texts
df = df[df['processed_text'].str.strip() != '']

X = df['processed_text']
y = df['sentiment']

# STEP 3: VECTORIZATION
print("\nVectorizing text using TF-IDF...")

vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# STEP 4: SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# STEP 5: TRAIN MODELS
print("\nTraining models...")

# Train Naive Bayes
print("Training Naive Bayes...")
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_pred)
nb_report = classification_report(y_test, nb_pred, output_dict=True)

# Train Support Vector Machine
print("Training SVM...")
svm_model = LinearSVC(max_iter=2000, random_state=42)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)
svm_report = classification_report(y_test, svm_pred, output_dict=True)

print("\nModel Training Complete!")

# STEP 6: VISUALIZATIONS

# ++++++++++++++++++++++++++++
# FIGURE 1: OVERVIEW VISUALS
# +++++++++++++++++++++++++++
fig1 = plt.figure(figsize=(14, 6))

# 1. Sentiment Distribution in Dataset
ax1 = plt.subplot(1, 3, 1)
sentiment_counts = df['sentiment'].value_counts()
colors = ['#2ecc71', '#e74c3c', '#95a5a6']
ax1.pie(sentiment_counts.values, labels=sentiment_counts.index,
        autopct='%1.1f%%', colors=colors[:len(sentiment_counts)],
        startangle=90)
ax1.set_title('Sentiment Distribution', fontsize=13, fontweight='bold')

# 2. Model Accuracy Comparison
ax2 = plt.subplot(1, 3, 2)
models = ['Naive Bayes', 'SVM']
accuracies = [nb_accuracy, svm_accuracy]
bars = ax2.bar(models, accuracies, color=['#3498db', '#9b59b6'])
ax2.set_ylabel('Accuracy', fontsize=11)
ax2.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
ax2.set_ylim([0, 1])
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2., height,
             f'{height:.3f}', ha='center', va='bottom', fontsize=10)

# 3. Text Length Distribution by Sentiment
ax3 = plt.subplot(1, 3, 3)
df['text_length'] = df['review'].str.len()
for sentiment in df['sentiment'].unique():
    data = df[df['sentiment'] == sentiment]['text_length']
    ax3.hist(data, alpha=0.5, label=sentiment, bins=30)
ax3.set_xlabel('Text Length (characters)', fontsize=11)
ax3.set_ylabel('Frequency', fontsize=11)
ax3.set_title('Text Length by Sentiment', fontsize=13, fontweight='bold')
ax3.legend()

plt.tight_layout()
plt.savefig('sentiment_analysis_overview.png', dpi=300, bbox_inches='tight')
print("Visualization saved as 'sentiment_analysis_overview.png'")
plt.show()


# ++++++++++++++++++++++++++++++++++++
# FIGURE 2: MODEL PERFORMANCE DETAILS
# ++++++++++++++++++++++++++++++++++++
fig2 = plt.figure(figsize=(14, 8))

# 4. Confusion Matrix - Naive Bayes
ax4 = plt.subplot(2, 2, 1)
cm_nb = confusion_matrix(y_test, nb_pred)
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', ax=ax4)
ax4.set_title('Confusion Matrix - Naive Bayes', fontsize=13, fontweight='bold')
ax4.set_ylabel('True Label')
ax4.set_xlabel('Predicted Label')

# 5. Confusion Matrix - SVM
ax5 = plt.subplot(2, 2, 2)
cm_svm = confusion_matrix(y_test, svm_pred)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Purples', ax=ax5)
ax5.set_title('Confusion Matrix - SVM', fontsize=13, fontweight='bold')
ax5.set_ylabel('True Label')
ax5.set_xlabel('Predicted Label')

# 6. Precision, Recall, F1-Score Comparison
ax6 = plt.subplot(2, 2, (3, 4))
metrics = ['precision', 'recall', 'f1-score']
nb_scores = [nb_report['weighted avg'][m] for m in metrics]
svm_scores = [svm_report['weighted avg'][m] for m in metrics]
x = np.arange(len(metrics))
width = 0.35
ax6.bar(x - width / 2, nb_scores, width, label='Naive Bayes', color='#3498db')
ax6.bar(x + width / 2, svm_scores, width, label='SVM', color='#9b59b6')
ax6.set_ylabel('Score', fontsize=11)
ax6.set_title('Precision, Recall, and F1 Comparison', fontsize=13, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(metrics)
ax6.legend()
ax6.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('sentiment_analysis_model_details.png', dpi=300, bbox_inches='tight')
print("Visualization saved as 'sentiment_analysis_model_details.png'")
plt.show()

# =============================================================================
# STEP 7: GENERATE COMPREHENSIVE REPORT
# =============================================================================
print("\n" + "=" * 80)
print("SENTIMENT ANALYSIS REPORT")
print("=" * 80)

print("\n1. DATASET OVERVIEW")
print("-" * 80)
print(f"Total samples: {len(df)}")
print(f"\nSentiment Distribution:")
sentiment_dist = df['sentiment'].value_counts()
for sentiment, count in sentiment_dist.items():
    percentage = (count / len(df)) * 100
    print(f"  {sentiment}: {count} ({percentage:.2f}%)")

print("\n2. TEXT PREPROCESSING")
print("-" * 80)
print("Steps performed:")
print(" Tokenization: Split text into individual words")
print(" Stopword Removal: Removed common words (the, is, at, etc.)")
print(" Lemmatization: Reduced words to root form (running → run)")
print(" Cleaning: Removed URLs, special characters, and digits")

print("\n3. MODEL PERFORMANCE")
print("-" * 80)

# Naive Bayes Results
print(f"\nNaive Bayes:")
print(f"  Accuracy: {nb_accuracy:.4f}")
print("\n  Detailed Classification Report:")

for label in [k for k in nb_report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]:
    print(f"\n  {label}:")
    print(f"    Precision: {nb_report[label]['precision']:.4f}")
    print(f"    Recall:    {nb_report[label]['recall']:.4f}")
    print(f"    F1-Score:  {nb_report[label]['f1-score']:.4f}")
    print(f"    Support:   {nb_report[label]['support']}")

# SVM Results
print(f"\nSVM:")
print(f"  Accuracy: {svm_accuracy:.4f}")
print("\n  Detailed Classification Report:")

for label in [k for k in svm_report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]:
    print(f"\n  {label}:")
    print(f"    Precision: {svm_report[label]['precision']:.4f}")
    print(f"    Recall:    {svm_report[label]['recall']:.4f}")
    print(f"    F1-Score:  {svm_report[label]['f1-score']:.4f}")
    print(f"    Support:   {svm_report[label]['support']}")

print("\n4. KEY INSIGHTS & TRENDS")
print("-" * 80)

# Determine best model
if nb_accuracy > svm_accuracy:
    best_model = "Naive Bayes"
    best_accuracy = nb_accuracy
else:
    best_model = "SVM"
    best_accuracy = svm_accuracy

print(f"\n• Best Performing Model: {best_model} (Accuracy: {best_accuracy:.4f})")

# Sentiment distribution insights
most_common_sentiment = sentiment_dist.index[0]
most_common_pct = (sentiment_dist.iloc[0] / len(df)) * 100
print(f"\n• Dominant Sentiment: {most_common_sentiment} ({most_common_pct:.1f}% of reviews)")

# Class balance
balance_ratio = sentiment_dist.min() / sentiment_dist.max()
if balance_ratio < 0.5:
    print(f"\n• Dataset Imbalance: Significant class imbalance detected (ratio: {balance_ratio:.2f})")
    print("  Recommendation: Consider using techniques like SMOTE or class weights")
else:
    print(f"\n• Dataset Balance: Relatively balanced dataset (ratio: {balance_ratio:.2f})")

# Model comparison
acc_diff = abs(nb_accuracy - svm_accuracy)
if acc_diff < 0.02:
    print("\n• Model Comparison: Both models show similar performance")
else:
    print(f"\n• Model Comparison: {best_model} outperforms by {acc_diff:.4f}")

print("\n5. RECOMMENDATIONS")
print("-" * 80)
print("\n• For deployment, consider using the", best_model, "model")
print("• Monitor model performance on new data regularly")
print("• Consider ensemble methods to combine both models")
print("• Collect more data for underrepresented sentiment classes if applicable")

print("\n" + "=" * 80)
print("END OF REPORT")
print("=" * 80 + "\n")

