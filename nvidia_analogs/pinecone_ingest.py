import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from config import PN_API_KEY, PN_ENV, INDEX_NAME

# load features and infer dim
df = pd.read_pickle("data/nvidia_features.pkl")
dim = df.drop(columns=["date"]).shape[1]

# init client
pc = Pinecone(api_key=PN_API_KEY, environment=PN_ENV)

# create index if missing (must match reset spec)
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=dim,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region=PN_ENV
        )
    )

index = pc.Index(INDEX_NAME)

# build upsert payload
vectors = [
    (str(row["date"]), row.drop("date").tolist(), {"date": str(row["date"])})
    for _, row in df.iterrows()
]

# upsert in batches
batch_size = 100
for i in range(0, len(vectors), batch_size):
    batch = vectors[i : i + batch_size]
    index.upsert(vectors=batch)
    print(f"Upserted {i}-{i+len(batch)-1}")

print("✔️  Ingestion complete.")
