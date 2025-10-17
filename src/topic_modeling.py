
"""
Topic Modeling & Document Clustering Template (TF-IDF + KMeans + LDA)

Usage examples:
    # CSV with a 'text' column
    python src/topic_modeling.py --input data/sample.csv --text_col text --k 8 --n_topics 10

    # Directory of .txt files
    python src/topic_modeling.py --input data/texts_dir --k 8 --n_topics 10

Outputs (saved to results/):
    - clusters.csv               : doc_id, text (truncated), cluster_label
    - kmeans_metrics.json        : inertia, silhouette, params
    - lda_topics.csv             : topic_id, top_words
    - lda_doc_topic.csv          : doc_id, topic_distribution...
"""

import argparse
import os
import json
import glob
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import silhouette_score

def read_corpus(input_path: str, text_col: Optional[str] = None):
    \"\"\"Read documents either from a CSV file or a directory of .txt files.\"\"\"
    if os.path.isdir(input_path):
        texts = []
        for p in sorted(glob.glob(os.path.join(input_path, "*.txt"))):
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                texts.append(f.read())
        if not texts:
            raise FileNotFoundError(f"No .txt files found under directory: {input_path}")
        return texts

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")
    df = pd.read_csv(input_path)
    if text_col is None:
        candidates = [c for c in df.columns if str(c).lower() in {"text","content","abstract","body"}]
        if candidates:
            text_col = candidates[0]
        else:
            obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
            if not obj_cols:
                raise ValueError("No text-like column found. Please specify --text_col.")
            text_col = obj_cols[0]
    if text_col not in df.columns:
        raise ValueError(f"text_col='{text_col}' not in CSV columns: {list(df.columns)}")
    texts = df[text_col].astype(str).fillna("").tolist()
    return texts

def top_terms_per_cluster(tfidf, kmeans, n_terms=15):
    terms = np.array(tfidf.get_feature_names_out())
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    clusters_terms = []
    for i in range(kmeans.n_clusters):
        clusters_terms.append(terms[order_centroids[i, :n_terms]].tolist())
    return clusters_terms

def top_words_per_topic(count_vectorizer, lda_model, n_words=15):
    vocab = np.array(count_vectorizer.get_feature_names_out())
    topics = []
    for comp in lda_model.components_:
        top_idx = np.argsort(comp)[::-1][:n_words]
        topics.append(vocab[top_idx].tolist())
    return topics

def run_pipeline(texts, k=8, n_topics=10, max_features=20000, results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)

    # TF-IDF for clustering
    tfidf = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        lowercase=True,
        token_pattern=r"(?u)\\b[A-Za-z][A-Za-z]+\\b"
    )
    X_tfidf = tfidf.fit_transform(texts)

    # KMeans
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_tfidf)
    inertia = float(kmeans.inertia_)
    sil = float(silhouette_score(X_tfidf, labels)) if k > 1 and X_tfidf.shape[0] > k else float("nan")

    # Save clusters.csv
    trunc = [t[:200].replace("\\n"," ") for t in texts]
    pd.DataFrame({"doc_id": range(len(texts)), "text": trunc, "cluster_label": labels}).to_csv(
        os.path.join(results_dir, "clusters.csv"), index=False
    )
    # Save kmeans_metrics.json
    with open(os.path.join(results_dir, "kmeans_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "k": k,
            "inertia": inertia,
            "silhouette": sil,
            "n_docs": int(X_tfidf.shape[0]),
            "n_features": int(X_tfidf.shape[1])
        }, f, indent=2)

    # LDA with CountVectorizer
    count_vec = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        lowercase=True,
        token_pattern=r"(?u)\\b[A-Za-z][A-Za-z]+\\b"
    )
    X_count = count_vec.fit_transform(texts)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method="batch",
        random_state=42
    )
    doc_topic = lda.fit_transform(X_count)

    # Save topics
    topics = top_words_per_topic(count_vec, lda, n_words=15)
    pd.DataFrame({"topic_id": range(n_topics), "top_words": [", ".join(ws) for ws in topics]}).to_csv(
        os.path.join(results_dir, "lda_topics.csv"), index=False
    )
    # Save doc-topic distribution
    dt = pd.DataFrame(doc_topic, columns=[f"topic_{i}" for i in range(n_topics)])
    dt.insert(0, "doc_id", range(len(texts)))
    dt.to_csv(os.path.join(results_dir, "lda_doc_topic.csv"), index=False)

    print("[OK] Saved results to:", results_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data", help="Path to CSV file or directory of .txt files")
    parser.add_argument("--text_col", type=str, default=None, help="Text column for CSV input")
    parser.add_argument("--k", type=int, default=8, help="Number of KMeans clusters")
    parser.add_argument("--n_topics", type=int, default=10, help="Number of LDA topics")
    parser.add_argument("--max_features", type=int, default=20000, help="Max features for vectorizers")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    texts = read_corpus(args.input, text_col=args.text_col)
    run_pipeline(
        texts=texts,
        k=args.k,
        n_topics=args.n_topics,
        max_features=args.max_features,
        results_dir=args.results_dir
    )

if __name__ == "__main__":
    main()
