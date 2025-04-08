import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')


# Module 1: Data Loading & Exploration

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Yelp reviews (first 10K) from full JSON dataset
df = pd.read_json('/Users/abhinavpatel/Downloads/yelp json/yelp_academic_dataset_review.json', 
                  lines=True, nrows=10000)

# Keep only review text and star rating
df = df[['text', 'stars']]

# Add review length column
df['review_length'] = df['text'].apply(lambda x: len(str(x).split()))

# Preview sample
print("Sample Reviews:")
print(df.head())

# Plot review length distribution
sns.histplot(df['review_length'], bins=50, kde=True)
plt.title("Distribution of Review Word Counts")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.show()

# Filter for longer reviews (optional)
df = df[df['review_length'] > 50].reset_index(drop=True)

# Save filtered sample for next module
df.to_csv('yelp_reviews_sample.csv', index=False)
print("✅ Saved cleaned sample to yelp_reviews_sample.csv")



# Module 2: Text Preprocessing

import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Load the sample dataset
df = pd.read_csv('yelp_reviews_sample.csv')

# Prepare custom stopwords list
custom_stopwords = set(stopwords.words('english'))
custom_stopwords.update(['would', 'could', 'also', 'one', 'get', 'got'])  # Add any domain-specific terms

# Function to clean a review
def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in custom_stopwords]
    # Remove short tokens (e.g., 'ok', 'br')
    tokens = [word for word in tokens if len(word) > 2]
    return ' '.join(tokens)

# Apply cleaning
print("Cleaning reviews...")
df['cleaned_text'] = df['text'].apply(clean_text)

# Preview cleaned data
print(df[['text', 'cleaned_text']].head())

# Save cleaned output
df.to_csv('yelp_reviews_cleaned.csv', index=False)
print("✅ Cleaned data saved to yelp_reviews_cleaned.csv")


# Module 3: Tokenization & Word Frequency Analysis

import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize

# Load cleaned data
df = pd.read_csv('yelp_reviews_cleaned.csv')

# Combine all cleaned reviews into one large text
all_text = ' '.join(df['cleaned_text'].dropna())

# Tokenize
tokens = word_tokenize(all_text)

# Count word frequencies
word_freq = Counter(tokens)

# Extract top 25 most common words
top_words = word_freq.most_common(25)

# Print top words
print("Top 25 Most Common Words:")
for word, freq in top_words:
    print(f"{word}: {freq}")

# Plotting
words, freqs = zip(*top_words)
plt.figure(figsize=(12,6))
plt.bar(words, freqs)
plt.title("Top 25 Most Frequent Words in Reviews")
plt.xticks(rotation=45)
plt.xlabel("Word")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Save frequency data to CSV
freq_df = pd.DataFrame(top_words, columns=['word', 'frequency'])
freq_df.to_csv('word_frequencies.csv', index=False)
print("✅ Word frequency data saved to word_frequencies.csv")



# Module 4: Keyword Extraction using TF-IDF

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Load cleaned data
df = pd.read_csv('yelp_reviews_cleaned.csv')

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_df=0.85, min_df=5, stop_words='english')
X = vectorizer.fit_transform(df['cleaned_text'])

# Get feature names
features = vectorizer.get_feature_names_out()

# Sum TF-IDF scores for each term across all documents
tfidf_scores = np.sum(X.toarray(), axis=0)
tfidf_df = pd.DataFrame({'term': features, 'score': tfidf_scores})

# Sort by score descending
tfidf_df = tfidf_df.sort_values(by='score', ascending=False)

# Extract top 20% keywords
top_20_percent = int(len(tfidf_df) * 0.2)
top_keywords_df = tfidf_df.head(top_20_percent)

# Display top keywords
print("Top 20% Unique Keywords:")
print(top_keywords_df.head(20))

# Save to CSV
top_keywords_df.to_csv('top_keywords_tfidf.csv', index=False)
print("✅ Top keywords saved to top_keywords_tfidf.csv")




# Module 5: Sentiment Analysis & Keyword Clustering

import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')

# Load cleaned data
df = pd.read_csv('yelp_reviews_cleaned.csv')

# --- Sentiment Analysis ---
print("Running sentiment analysis...")
df['polarity'] = df['cleaned_text'].apply(lambda text: TextBlob(str(text)).sentiment.polarity)
df['sentiment'] = df['polarity'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))

# Save sentiment labeled data
df.to_csv('yelp_reviews_with_sentiment.csv', index=False)
print("✅ Sentiment analysis complete and saved to yelp_reviews_with_sentiment.csv")

# --- Keyword Clustering ---
print("Running keyword clustering...")
# Use same TF-IDF setup
vectorizer = TfidfVectorizer(max_df=0.85, min_df=5, stop_words='english')
X = vectorizer.fit_transform(df['cleaned_text'])

# Cluster keywords using KMeans
num_clusters = 5
km = KMeans(n_clusters=num_clusters, random_state=42)
km.fit(X)

# Add cluster labels to DataFrame
df['cluster'] = km.labels_

# Plot sample cluster distribution
plt.figure(figsize=(8,5))
df['cluster'].value_counts().sort_index().plot(kind='bar')
plt.title("Review Cluster Distribution")
plt.xlabel("Cluster Label")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Save clustered data
df.to_csv('yelp_reviews_clustered.csv', index=False)
print("✅ Clustering complete and saved to yelp_reviews_clustered.csv")
