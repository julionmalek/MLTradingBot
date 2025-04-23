from pinecone import Pinecone, ServerlessSpec
from config import PN_API_KEY, PN_ENV, INDEX_NAME
import pandas as pd

# infer dimension from your latest feature file
df = pd.read_pickle("data/nvidia_features.pkl")
dim = df.drop(columns=["date"]).shape[1]

pc = Pinecone(api_key=PN_API_KEY, environment=PN_ENV)

# delete existing index
if INDEX_NAME in pc.list_indexes().names():
    print(f"Deleting old index '{INDEX_NAME}'…")
    pc.delete_index(INDEX_NAME)

# create new index with spec
print(f"Creating index '{INDEX_NAME}' with dimension={dim}")
pc.create_index(
    name=INDEX_NAME,
    dimension=dim,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region=PN_ENV
    )
)
print("✔️  Index reset complete.")
