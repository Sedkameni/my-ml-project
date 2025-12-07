## Import Libraries and Necessary Resources
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

## II - Load the dataset
# 1 - Load the dataset into Python using pandas.
df = pd.read_csv('datasetLab.csv')

## 2 - Convert to lowercase
df['text'] = df['text'].str.lower()

## 3 - Tokenize the text into words.
# Tokenization
df['tokens'] = df['text'].apply(word_tokenize)

## 4 - Remove stopwords to filter out common words with little meaning.
# Remove stopwords
stop_words = set(stopwords.words('english'))
df['filtered_tokens'] = df['tokens'].apply(lambda x: [word for word in x if word not in stop_words])

## 5 - Perform stemming or lemmatization to reduce words to their base forms.
# Stemming
stemmer = PorterStemmer()
df['stemmed_tokens'] = df['filtered_tokens'].apply(lambda x: [stemmer.stem(word) for word in x])
print(df.head())

## III - Sentiment Analysis

## 1 -  Preprocess the dataset as performed in Step 2.
## 2 -  Implement a sentiment analysis model using a suitable algorithm (e.g., Naive Bayes, Support Vector Machines, or pre-trained models like VADER or TextBlob).
## 3 -  Evaluate the performance of the sentiment analysis model.
## 4 - Analyze sentiment distribution within the dataset and identify key trends.
from textblob import TextBlob

# Function to analyze sentiment
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

df['sentiment_score'] = df['text'].apply(get_sentiment)

# Categorizing sentiment
conditions = [
    (df['sentiment_score'] > 0),
    (df['sentiment_score'] == 0),
    (df['sentiment_score'] < 0)
]
labels = ['Positive', 'Neutral', 'Negative']
df['sentiment'] = pd.cut(df['sentiment_score'], bins=[-1, 0, 1], labels=['Negative', 'Positive'], include_lowest=True, duplicates='drop')

print("Sentiment Analysis Results:")
print(df[['text', 'sentiment']].head())

### IV - Data Visualization and Insights

## 1 - Visualize the sentiment distribution using bar charts or pie charts.
## 2 -  Discuss key findings and insights from the sentiment analysis results.
## 3 - Identify any trends or patterns in the dataset.
df['sentiment'].value_counts().plot(kind='bar', color=['green', 'gray', 'red'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()



