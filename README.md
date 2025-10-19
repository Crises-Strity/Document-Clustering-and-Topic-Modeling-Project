# Document Clustering and Topic Modeling

A research-grade repository for **unsupervised document analysis** using **TF-IDF + KMeans** for clustering and **Latent Dirichlet Allocation (LDA)** for topic modeling.

> Notebook: `notebooks/Document Clustering and Topic Modeling Project.ipynb`  
> Data folder: `data/` (add your own raw text data)

---

## Repository Structure
```
document-clustering-topic-modeling/
├── data/                           # Place your raw data here
├── notebooks/
│   └── Document Clustering and Topic Modeling Project.ipynb
├── src/
│   └── topic_modeling.py           # Runnable pipeline (TF-IDF, KMeans, LDA)
├── results/                        # Auto-generated outputs
├── requirements.txt
├── LICENSE                         # MIT
├── .gitignore
└── README.md
```

## Methodology
- **Vectorization**: `TfidfVectorizer` (for clustering), `CountVectorizer` (for LDA) with English stopwords
- **Clustering**: `KMeans (k clusters)` using TF-IDF features
- **Topic Modeling**: `LatentDirichletAllocation (n_topics)`
- **Evaluation**: `silhouette score` (for KMeans); topic quality via top words

## Dataset
You can supply documents in either format:
1. **CSV file** with a text column (default candidates: `text`, `content`, `abstract`, `body`).  
   Example path: `data/sample.csv`
2. **Directory of `.txt` files** (each file is one document).  
   Example path: `data/texts_dir/`

> The repository intentionally ships with an empty `data/` folder so you can insert your own raw data.

## Installation
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Usage
### Option A: CSV input
```bash
python src/topic_modeling.py   --input data/sample.csv   --text_col text   --k 8   --n_topics 10
```

### Option B: Directory of .txt files
```bash
python src/topic_modeling.py   --input data/texts_dir   --k 8   --n_topics 10
```

### Outputs (saved under `results/`)
- `clusters.csv` — `doc_id`, truncated text, and assigned `cluster_label`  
- `kmeans_metrics.json` — inertia, silhouette score, and run metadata  
- `lda_topics.csv` — each topic’s top words  
- `lda_doc_topic.csv` — per-document topic distribution

## 🧪 Notes & Tips
- Tune `--k` and `--n_topics` via small sweeps to find stable clusters and coherent topics.
- Increase `--max_features` for larger corpora, but mind memory usage.
- Pre-cleaning (lowercasing, de-noising, removing boilerplate) usually improves results.

## 🚀 Roadmap
- Add **grid search** over `k` and `n_topics` with metrics summary
- Support **n-grams** and **custom stopwords**
- Export plots (cluster sizes, topic wordbars)
- Optional **lemmatization**

## License
This project is released under the **MIT License** (see `LICENSE`).

## Citation
If you use this repository, please cite:
```
Cris Wang. Document Clustering and Topic Modeling. GitHub repository, 2025.
```
