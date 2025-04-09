import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="🧠 Yelp Review Analyzer", layout="wide")
st.title("🧠 Word Wrangling & Yelp Review Dashboard")

# Load datasets
@st.cache_data
def load_data():
    reviews = pd.read_csv("yelp_reviews_with_sentiment.csv")
    freq = pd.read_csv("word_frequencies.csv")
    tfidf = pd.read_csv("top_keywords_tfidf.csv")
    return reviews, freq, tfidf

reviews, freq_df, tfidf_df = load_data()

# --- Sidebar Filters ---
st.sidebar.header("🔍 Filters")
selected_sentiment = st.sidebar.multiselect(
    "Filter by Sentiment", reviews['sentiment'].unique(), default=reviews['sentiment'].unique()
)

filtered_reviews = reviews[reviews['sentiment'].isin(selected_sentiment)]

# --- Sentiment Breakdown ---
st.subheader("📊 Sentiment Distribution")
sentiment_counts = filtered_reviews['sentiment'].value_counts()
st.bar_chart(sentiment_counts)

# --- Word Frequency ---
st.subheader("🔠 Top Frequent Words")
top_n = st.slider("Top N Words", min_value=5, max_value=30, value=15)
st.dataframe(freq_df.head(top_n))

fig1, ax1 = plt.subplots()
ax1.bar(freq_df['word'].head(top_n), freq_df['frequency'].head(top_n))
plt.xticks(rotation=45)
st.pyplot(fig1)

# --- TF-IDF Keywords ---
st.subheader("⭐ Top Unique Keywords (TF-IDF)")
tfidf_n = st.slider("Top N Keywords (TF-IDF)", min_value=5, max_value=30, value=15)
st.dataframe(tfidf_df.head(tfidf_n))

# --- Review Samples ---
st.subheader("📝 Sample Reviews by Sentiment")
num_samples = st.slider("Number of Reviews to Show", 1, 10, 5)
for i, row in filtered_reviews.head(num_samples).iterrows():
    st.markdown(f"**⭐ {row['stars']} Stars | Sentiment: {row['sentiment']}**")
    st.write(row['text'])
    st.markdown("---")
