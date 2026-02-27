from sentence_transformers import SentenceTransformer
import requests, json

model = SentenceTransformer("all-MiniLM-L6-v2")

query = "black leather jacket for women"
vector = model.encode(query).tolist()

payload = {
    "size": 5,
    "query": {
        "knn": {
            "embedding": {
                "vector": vector,
                "k": 5
            }
        }
    }
}

r = requests.post(
    "http://localhost:9200/products/_search",
    headers={"Content-Type": "application/json"},
    data=json.dumps(payload)
)

print(r.json()["hits"]["hits"])