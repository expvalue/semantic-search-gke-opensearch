# semantic-search-gke-opensearch

Semantic retrieval service using OpenSearch, FastAPI, GKE.

## How search works

- **Vector side**: We use HNSW in OpenSearch so we get approximate nearest neighbors without scanning everything. Index is built for cosine similarity so it matches how we're comparing embeddings. You can tune `ef_construction` and `m` at ingest time, and `ef_search` when searching (higher = better recall, slower).
- **k-NN**: We ask for the top-k closest vectors to the query; HNSW gives us that set fast even on big datasets.
- **Re-ranking**: We take those hits and re-score in Python: (1) cosine sim between query and doc embeddings, (2) keyword stuff (tf-style, substring matches, phrase boost, and we knock down gender mismatch so "men" doesn't surface women's stuff). Then we fuse the two rank lists with RRF so we don't have to fuss with weights.

If you re-run the ingest you get the HNSW index. Old indexes without it still work, they just use the default path.

## Run search locally (what you need to do)

**1. Start OpenSearch** (Docker must be running)

```bash
cd infra
docker compose up -d opensearch
```

Wait about 30 seconds.

**2. Index data**  
You need the file `data/meta_Amazon_Fashion.jsonl.gz` in the repo root. From the repo root:

```bash
python ingest/ingest_fashion.py
```

**3. Run the API**

From the repo root:

```bash
uvicorn api.search_service:app --reload
```

Then open http://127.0.0.1:8000 and search.

---

- API and ingest hit `http://localhost:9200` by default. For GKE/Docker set `OPENSEARCH_URL` to your OpenSearch URL.
- If OpenSearch isn't up, search returns 503.
