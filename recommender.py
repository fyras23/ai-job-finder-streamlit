# recommender.py
import pandas as pd
import numpy as np
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and prepare data (only once)
print("Loading job dataset and building recommender... (this takes ~20 seconds)")
df = pd.read_csv("postings_cleaned.csv")

# Combine title + description
df['text'] = (df['title'].fillna('') + " " + df['description'].fillna('')).str.lower()
df['text'] = df['text'].apply(lambda x: re.sub(r'[^a-z0-9\s]', ' ', x))
df['text'] = df['text'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())

# Extract city/state
def extract_loc(loc):
    if pd.isna(loc): return None, None
    parts = [p.strip() for p in str(loc).split(',')]
    city = parts[0] if len(parts) > 0 else None
    state = parts[-1] if len(parts) > 1 else None
    return city, state

df[['city', 'state']] = df['location'].apply(lambda x: pd.Series(extract_loc(x)))
df['salary'] = df['normalized_salary']

# Train TF-IDF (this is the heavy part — we save it)
print("Training TF-IDF on 123,849 jobs...")
tfidf = TfidfVectorizer(stop_words='english', max_features=20000, ngram_range=(1,2))
tfidf_matrix = tfidf.fit_transform(df['text'])

# Save everything
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)
with open("tfidf_matrix.pkl", "wb") as f:
    pickle.dump(tfidf_matrix, f)
df.to_pickle("jobs_df.pkl")

print("Recommender ready and saved!")