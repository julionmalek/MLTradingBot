import pandas as pd
from pinecone import Pinecone
from config import PN_API_KEY, PN_ENV, INDEX_NAME

# 1) Load your feature DataFrame
feat_df = pd.read_pickle("data/nvidia_features.pkl")
print(f"Loaded {len(feat_df)} feature rows.")

# 2) Grab the most recent vector
latest = feat_df.iloc[-1]
query_vec = latest.drop("date").tolist()
print(f"Querying using date = {latest['date']}")

# 3) Initialize Pinecone client & index
pc = Pinecone(api_key=PN_API_KEY, environment=PN_ENV)
index = pc.Index(INDEX_NAME)

# 4) Perform the query (single vector syntax)
response = index.query(
    vector=query_vec,
    top_k=5,
    include_metadata=True
)

# 5) The matches now live in response.matches
for match in response.matches:
    date = match.metadata.get('date', 'n/a')
    print(f"Date: {date}, Score: {match.score}")
