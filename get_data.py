import pandas as pd
import json

json_path = "/Users/pallakdhabalia/.cache/kagglehub/datasets/Cornell-University/arxiv/versions/238/arxiv-metadata-oai-snapshot.json"

data = []
# Load the first 10,000 entries from the JSONL file 
with open(json_path, 'r') as f:
    for i, line in enumerate(f):
        if i >= 10000:
            break
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError:
            continue

df = pd.DataFrame(data)

# Clean data
df = df[df["abstract"].notnull() & df["title"].notnull()]
df = df[df["abstract"].str.len() > 100]
df = df.drop_duplicates(subset=["abstract"]).reset_index(drop=True)

# Take 5,000 samples and save
df.sample(n=min(5000, len(df)), random_state=42).to_csv("cleaned_arxiv_subset.csv", index=False)
print(f"Saved cleaned subset with {min(5000, len(df))} entries.")