import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load embeddings and metadata
embeddings = np.load("abstract_embeddings.npy")
metadata = pd.read_csv("paper_metadata.csv")

model = SentenceTransformer("all-MiniLM-L6-v2")

def search_papers(query, top_k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    top_indices = similarities.argsort()[::-1][:top_k]
    results = metadata.iloc[top_indices]
    scores = similarities[top_indices]
    return list(zip(results["title"], results["abstract"], scores))

if __name__ == "__main__":
    query = input("Enter search query (or 'exit'): ")
    if query.lower() == 'exit':
        print("Exiting search.")
        exit()
    results = search_papers(query)

    with open("search_results.txt", "w", encoding="utf-8") as f:
        for i, (title, abstract, score) in enumerate(results, 1):
            f.write(f"#{i}. {title}\n")
            f.write(f"Similarity: {score:.3f}\n")
            f.write(f"Abstract: {abstract[:300]}...\n\n")
    print(f"Search results saved to 'search_results.txt'. Found {len(results)} results.")