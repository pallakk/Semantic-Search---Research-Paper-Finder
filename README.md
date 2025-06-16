# Semantic Research Paper Finder

This project implements a semantic search engine for academic research papers using SBERT (Sentence-BERT). Given a user's natural language query, it retrieves the most semantically relevant research paper abstracts from a curated arXiv dataset.

## 🔍 Features
- Vector-based semantic search using sentence-transformers (`all-MiniLM-L6-v2`)
- Cosine similarity ranking over 5,000 preprocessed arXiv papers
- Fast query processing with numpy and scikit-learn
- Offline-ready: No external APIs or internet inference required
- Saves results to a text file for easy analysis

## 📁 Dataset
- Source: [Cornell University arXiv Metadata Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv)
- Fields used: Title, Abstract, Categories, Authors
- Preprocessed and embedded using SBERT into `abstract_embeddings.npy`

## 💡 Use Cases
- Quickly find semantically related papers for literature reviews
- Discover connections between fields with free-text queries
- Lay foundation for building vector search APIs or LLM-enhanced tools

## 🚀 Technologies
- Python, pandas, numpy
- [sentence-transformers](https://www.sbert.net/)
- sklearn cosine similarity

## ✍️ Example
Query: "graph neural networks for social networks"
Output → Top 5 papers with similarity scores and abstract previews

## 🔧 To Run
bash:
pip install -r requirements.txt
python semantic_search.py
