import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

# Load the cleaned CSV with 5000 abstracts
df = pd.read_csv("cleaned_arxiv_subset.csv")

# Convert the abstracts to a list
abstracts = df["abstract"].tolist()

if not abstracts:
    raise ValueError("No abstracts found in the dataset.")
print(f"Loaded {len(abstracts)} abstracts.")

# Load a pre-trained SBERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings for all abstracts
embeddings = model.encode(abstracts, show_progress_bar=True, convert_to_numpy=True)
# Save the embeddings to a .npy file
np.save("abstract_embeddings.npy", embeddings)
# Save the associated metadata (title + abstract) for search results
df[["title", "abstract"]].to_csv("paper_metadata.csv", index=False)
print("Saved embeddings and metadata.")

