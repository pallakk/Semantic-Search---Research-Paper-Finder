# Semantic Research Paper Finder

This project implements a semantic search engine for academic research papers using Sentence-BERT. Given a user's natural language query, it retrieves the most semantically relevant research paper abstracts from a curated arXiv dataset.

## Features

- Vector-based semantic search using `sentence-transformers` (all-MiniLM-L6-v2)
- Cosine similarity ranking over 5,000 preprocessed arXiv abstracts
- Fast in-memory querying using NumPy and scikit-learn
- Saves results to `search_results.txt` for easy analysis
- Optional browser-based UI built with Flask and HTML/CSS
- Works offline — no API keys or external services required

## Dataset

- Source: [Cornell University arXiv Metadata Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv)
- Fields used: `title`, `abstract`, `categories`, `authors`
- Processed to:
  - `cleaned_arxiv_subset.csv` — sampled and cleaned subset
  - `abstract_embeddings.npy` — precomputed SBERT embeddings
  - `paper_metadata.csv` — title and abstract metadata

## Use Cases

- Discover relevant papers for literature reviews and research
- Explore cross-domain relationships through free-text queries
- Prototype vector search for ML or LLM-powered systems
- Integrate into larger academic research tools

## Technologies

- Python, pandas, NumPy
- sentence-transformers
- scikit-learn
- Flask (for web UI)
- HTML, CSS (basic frontend)

## Example

**Query:**  
`graph neural networks for social networks`

**Output:**  
Top 5 paper titles with semantic similarity scores and short abstract previews.

## Usage
```bash
pip install -r requirements.txt
python embed_abstracts.py
python semantic_search.py
```
OR
```bash
python app.py
```

Owner
Pallak Dhabalia
GitHub: github.com/pallakk
LinkedIn: linkedin.com/in/pallak-dhabalia
