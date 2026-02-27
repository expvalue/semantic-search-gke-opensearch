import requests
from sentence_transformers import SentenceTransformer

OPENSEARCH_URL = "http://localhost:9200"
INDEX_NAME = "products"

# Adjustable weights
VECTOR_WEIGHT = 0.7
BM25_WEIGHT = 0.3

model = SentenceTransformer("all-MiniLM-L6-v2")

query = "black leather biker jacket"
embedding = model.encode(query).tolist()

search_body = {
    "size": 5,
    "explain": True,
    "query": {
        "function_score": {
            "query": {
                "bool": {
                    "should": [
                        {
                            "knn": {
                                "embedding": {
                                    "vector": embedding,
                                    "k": 50
                                }
                            }
                        },
                        {
                            "match": {
                                "title": {
                                    "query": query
                                }
                            }
                        }
                    ]
                }
            },
            "boost_mode": "sum",
            "score_mode": "sum",
            "functions": [
                {
                    "filter": {
                        "knn": {
                            "embedding": {
                                "vector": embedding,
                                "k": 50
                            }
                        }
                    },
                    "weight": VECTOR_WEIGHT
                },
                {
                    "filter": {
                        "match": {
                            "title": query
                        }
                    },
                    "weight": BM25_WEIGHT
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
    print("SCORE:", hit["_score"])
    print("TITLE:", hit["_source"]["title"])
    print("-----")