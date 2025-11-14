#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train an LDA topic model on BBC News (bbc-text.csv) and save artifacts for a Streamlit app.

Usage:
    python train_topic_model.py --csv bbc-text.csv --topics 5 --max_features 5000

Outputs:
    artifacts/
        vectorizer.joblib
        lda_model.joblib
        topic_top_words.json
        pyldavis.html
        config.json
"""

import argparse
import json
import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Optional: pyLDAvis for visualization
try:
    import pyLDAvis
    import pyLDAvis.sklearn
    HAVE_PYLDAVIS = True
except Exception:
    HAVE_PYLDAVIS = False

# NLTK for lemmatization/stopwords
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ======================
#  GLOBAL TOKENIZER
# ======================
def ensure_nltk():
    """Download necessary NLTK data files if not found."""
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


# initialize tokenizer dependencies
ensure_nltk()
sw = set(stopwords.words('english'))
lemm = WordNetLemmatizer()
regex = re.compile(r"[A-Za-z]+")


def tokenize(text):
    """Tokenize, clean, and lemmatize text."""
    text = text.lower()
    tokens = regex.findall(text)
    tokens = [t for t in tokens if t not in sw and len(t) > 2]
    tokens = [lemm.lemmatize(t) for t in tokens]
    return tokens


# ======================
#  HELPER FUNCTIONS
# ======================
def get_top_words(model, feature_names, n_top_words=10):
    """Extract top words per topic."""
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[-n_top_words:][::-1]
        words = [feature_names[i] for i in top_indices]
        weights = [float(topic[i]) for i in top_indices]
        topics.append({
            "topic": int(topic_idx),
            "top_words": words,
            "weights": weights
        })
    return topics


# ======================
#  MAIN SCRIPT
# ======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default=r"C:\Users\LAP-STORE\Desktop\Amit\NLP_Intern\Topic Modeling\bbc-text.csv",
        help="Path to BBC csv (expects columns: 'text' and optional 'category')"
    )
    parser.add_argument("--text_col", type=str, default="text", help="Name of the text column")
    parser.add_argument("--topics", type=int, default=5, help="Number of topics")
    parser.add_argument("--max_features", type=int, default=5000, help="Max features for CountVectorizer")
    parser.add_argument("--max_df", type=float, default=0.95, help="Ignore terms in > max_df of documents")
    parser.add_argument("--min_df", type=int, default=5, help="Ignore terms in < min_df documents")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts", help="Directory to save artifacts")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.csv)
    if args.text_col not in df.columns:
        raise ValueError(f"Column '{args.text_col}' not found. Available columns: {list(df.columns)}")

    texts = df[args.text_col].astype(str).tolist()

    # Vectorize
    vectorizer = CountVectorizer(
        tokenizer=tokenize,
        max_features=args.max_features,
        max_df=args.max_df,
        min_df=args.min_df
    )
    X = vectorizer.fit_transform(texts)

    # LDA
    lda = LatentDirichletAllocation(
        n_components=args.topics,
        learning_method="batch",
        random_state=42,
        max_iter=20,
        evaluate_every=5,
        n_jobs=-1
    )
    lda.fit(X)

    # Prepare artifacts dir
    art_dir = Path(args.artifacts_dir)
    art_dir.mkdir(parents=True, exist_ok=True)

    # Save models
    joblib.dump(vectorizer, art_dir / "vectorizer.joblib")
    joblib.dump(lda, art_dir / "lda_model.joblib")

    # Save top words
    feature_names = vectorizer.get_feature_names_out()
    topics = get_top_words(lda, feature_names, n_top_words=15)
    with open(art_dir / "topic_top_words.json", "w", encoding="utf-8") as f:
        json.dump(topics, f, ensure_ascii=False, indent=2)

    # Save pyLDAvis
    if HAVE_PYLDAVIS:
        try:
            vis = pyLDAvis.sklearn.prepare(lda, X, vectorizer)
            pyLDAvis.save_html(vis, str(art_dir / "pyldavis.html"))
        except Exception as e:
            print(f"[WARN] Failed to create pyLDAvis: {e}")

    # Save config (for Streamlit app)
    config = {
        "topics": int(args.topics),
        "text_col": args.text_col,
        "max_features": int(args.max_features),
        "max_df": float(args.max_df),
        "min_df": int(args.min_df),
        "csv": os.path.basename(args.csv)
    }
    with open(art_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # Print a small summary
    print("=== Training Summary ===")
    print(f"Documents: {len(texts)}")
    print(f"Vocabulary size: {len(feature_names)}")
    print(f"Topics: {args.topics}")
    print("\nTop words per topic:")
    for t in topics:
        top10 = ", ".join(t["top_words"][:10])
        print(f"Topic {t['topic']}: {top10}")


if __name__ == "__main__":
    main()
