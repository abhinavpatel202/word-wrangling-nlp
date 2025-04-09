# 🧠 Word Wrangling & Text Analytics Tool
![Python](https://img.shields.io/badge/python-3.12-blue?logo=python)
![NLP](https://img.shields.io/badge/NLP-TFIDF%20%7C%20KMeans-lightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)
![Status](https://img.shields.io/badge/status-Complete-brightgreen)


A full-stack NLP pipeline built to preprocess, analyze, and extract insights from messy, user-generated reviews. This tool uses the **Yelp Open Dataset** to simulate real-world data challenges and demonstrates scalable techniques in text cleaning, keyword extraction, sentiment analysis, and clustering.

---

## 📦 Features

- ✅ Load & explore large-scale JSON datasets  
- 🧹 Clean and normalize raw text (punctuation, stopwords, emojis, etc.)  
- 🔍 Extract high-frequency and high-value keywords using `Counter` and `TF-IDF`  
- ❤️ Analyze sentiment polarity with `TextBlob`  
- 🔄 Cluster reviews by content using `KMeans` (unsupervised topic segmentation)  
- 📊 Visualize review lengths, word frequencies, and cluster distributions  

---

## 🔧 Tech Stack

| Component     | Library/Tool           |
|---------------|------------------------|
| Language      | Python 3.12+           |
| Data Analysis | pandas, nltk, TextBlob |
| NLP           | nltk, scikit-learn     |
| Clustering    | KMeans (sklearn)       |
| Visualization | matplotlib, seaborn    |
| Dataset       | Yelp Open Dataset      |

---

## 🧪 Project Modules

### Module 1: Data Loading & Exploration
- Load `yelp_academic_dataset_review.json`
- Filter for longer reviews
- Plot review word count distributions

### Module 2: Text Preprocessing
- Lowercasing, punctuation removal
- Stopword filtering with `nltk`
- Tokenization & normalization

### Module 3: Word Frequency Analysis
- Tokenize and extract frequent terms using `collections.Counter`
- Save and plot top terms

### Module 4: Unique Keyword Extraction
- Apply `TF-IDF` vectorization
- Extract and rank top 20% informative keywords

### Module 5: Sentiment Analysis & Clustering
- Use `TextBlob` to label each review as positive/negative/neutral
- Cluster reviews with `KMeans` and visualize topic groupings

---

## 📂 Output Files

| Filename                          | Description                      |
|----------------------------------|----------------------------------|
| `yelp_reviews_sample.csv`        | Initial long-review sample       |
| `yelp_reviews_cleaned.csv`       | Cleaned version of review text   |
| `word_frequencies.csv`           | Top frequent terms               |
| `top_keywords_tfidf.csv`         | Top 20% most unique keywords     |
| `yelp_reviews_with_sentiment.csv`| Sentiment scores & labels        |
| `yelp_reviews_clustered.csv`     | KMeans cluster assignments       |

---

## 🚀 Getting Started

### 1. Clone this repository

```bash
git clone https://github.com/abhinavpatel202/word-wrangling-nlp.git
cd word-wrangling-nlp
```

### 2. Download the Yelp Open Dataset

Visit the official Yelp dataset page:  
👉 https://www.yelp.com/dataset

- Download and extract the `.tar` file  
- Place `yelp_academic_dataset_review.json` into the project folder

### 3. Install required dependencies

If you don’t have a `requirements.txt`, install manually:

```bash
pip install pandas nltk textblob scikit-learn matplotlib seaborn
python -m textblob.download_corpora
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### 4. Run the main pipeline

```bash
python DataWrangling.py
```

---

## 📌 Author

**Abhinav Patel**  
Data Analyst | NLP Enthusiast  
📫 [https://www.linkedin.com/in/patel-abhinav-ms/](https://www.linkedin.com/in/patel-abhinav-ms/)

---

## 🧠 Bonus Ideas

Want to extend this?

- Add Named Entity Recognition (NER) with `spaCy`  
- Use `LDA` or `BERTopic` for better topic modeling  
- Build an interactive dashboard with `Streamlit`

---

⭐ If you liked this project, drop a ⭐ on the repo!










