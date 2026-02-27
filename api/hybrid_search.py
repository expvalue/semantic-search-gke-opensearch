import requests
from sentence_transformers import SentenceTransformer

OPENSEARCH_URL = "http://localhost:9200"
INDEX_NAME = "products"

model = SentenceTransformer("all-MiniLM-L6-v2")

query = "black leather biker jacket"

embedding = model.encode(query).tolist()

search_body = {
    "size": 5,
    "query": {
        "bool": {
            "should": [
                {
                    "knn": {
                        "embedding": {
                            "vector": embedding,
                            "k": 20
                        }
                    }
                },
                {
                    "match": {
                        "title": {
                            "query": query,
                            "boost": 0.5
                        }
                    }
                }
            ]
        }
    }
}

response = requests.post(
    f"{OPENSEARCH_URL}/{INDEX_NAME}/_search",
    json=search_body
)

for hit in response.json()["hits"]["hits"]:
    print(hit["_score"], "-", hit["_source"]["title"])