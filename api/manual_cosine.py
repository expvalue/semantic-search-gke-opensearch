import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm

OPENSEARCH_URL = "http://localhost:9200"
INDEX_NAME = "products"

model = SentenceTransformer("all-MiniLM-L6-v2")

query = "black leather biker jacket"
query_vec = model.encode(query)

# Pull top 20 from OpenSearch (vector only for recall)
search_body = {
    "size": 20,
    "query": {
        "knn": {
            "embedding": {
                "vector": query_vec.tolist(),
                "k": 20
            }
        }
    }
}

response = requests.post(
    f"{OPENSEARCH_URL}/{INDEX_NAME}/_search",
    json=search_body
)

hits = response.json()["hits"]["hits"]

print("\nManual Cosine Scores (0–1 scale):\n")

for hit in hits[:5]:
    doc_vec = np.array(hit["_source"]["embedding"])
    cosine = np.dot(query_vec, doc_vec) / (norm(query_vec) * norm(doc_vec))

    # Convert from [-1,1] → [0,1]
    normalized = (cosine + 1) / 2

    print(f"{normalized:.4f}  -  {hit['_source']['title']}")