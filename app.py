from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load model and data once
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = np.load("abstract_embeddings.npy")
metadata = pd.read_csv("paper_metadata.csv")

def search_papers(query, top_k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_k]
    results = metadata.iloc[top_indices]
    scores = similarities[top_indices]
    return list(zip(results["title"], results["abstract"], scores))

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    if request.method == "POST":
        query = request.form["query"]
        if query:
            results = search_papers(query)
    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
