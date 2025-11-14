#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- FIX: tokenizer definition (needed for joblib.load) ---
def ensure_nltk():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4', quiet=True)

ensure_nltk()
sw = set(stopwords.words('english'))
lemm = WordNetLemmatizer()
regex = re.compile(r"[A-Za-z]+")

def tokenize(text):
    text = text.lower()
    tokens = regex.findall(text)
    tokens = [t for t in tokens if t not in sw and len(t) > 2]
    tokens = [lemm.lemmatize(t) for t in tokens]
    return tokens
# ----------------------------------------------------------

import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import NMF

# Load artifacts
ART_DIR = Path(r"C:\Users\LAP-STORE\Desktop\Amit\NLP_Intern\Topic Modeling\artifacts")
vectorizer = joblib.load(ART_DIR / "vectorizer.joblib")
lda = joblib.load(ART_DIR / "lda_model.joblib")

# Load dataset
df = pd.read_csv(r"C:\Users\LAP-STORE\Desktop\Amit\NLP_Intern\Topic Modeling\bbc-text.csv")
texts = df["text"].astype(str).tolist()
X = vectorizer.transform(texts)

# Train NMF model for comparison
nmf = NMF(n_components=5, random_state=42)
W = nmf.fit_transform(X)
H = nmf.components_

feature_names = vectorizer.get_feature_names_out()

def print_top_words(model, feature_names, n_top_words=10, title=None):
    if title:
        print(f"\n=== {title} ===")
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        print(f"Topic {topic_idx}: {', '.join(top_words)}")

# Compare
print_top_words(lda, feature_names, 10, title="🔹 LDA Topics")
print_top_words(nmf, feature_names, 10, title="🔸 NMF Topics")

print("\n✅ Comparison complete! Check which model produced more coherent topics.")
