# Semantic Fashion Search

**Vector search + hybrid re-ranking for product discovery.** Query by intent, style, and attributes—not just keywords. Built with OpenSearch (HNSW), FastAPI, and sentence-transformers; runs locally or on GKE.

---

## What it does

- **Semantic search** — "minimal leather bag" and "sleek black crossbody" both surface the right products.
- **Hybrid ranking** — Combines vector similarity with keyword signals (TF, phrase match, substring) and **gender-aware** scoring so "men" doesn’t surface women’s items.
- **Fast at scale** — HNSW approximate nearest neighbors; no full scan. Re-rank in Python with RRF so semantic and lexical play nice without tuning weights.

---

## How it works (short)

1. **Index**: Product text (title + description + features) is embedded with a shared prefix so query and docs live in the same space. Stored in OpenSearch with an HNSW index (cosine).
2. **Query**: User query gets the same prefix, we embed it, then run KNN on the vector index.
3. **Re-rank**: We take the top candidates and re-score with (a) cosine similarity (min–max normalized), (b) keyword score (BM25-style + phrase + gender match/mismatch). RRF fuses the two rank lists and we return top‑k with a 0–1 display score.

Re-running the ingest creates the HNSW index; older indexes still work with the default KNN path.

---

## Run it locally

**Prereqs:** Docker running, Python env with deps from `requirements.txt`.

| Step | Command |
|------|--------|
| 1. Start OpenSearch | `cd infra` → `docker compose up -d opensearch` (wait ~30s) |
| 2. Index data | From repo root, ensure `data/meta_Amazon_Fashion.jsonl.gz` exists, then `python ingest/ingest_fashion.py` |
| 3. Start API | From repo root: `uvicorn api.search_service:app --reload` |
| 4. Use it | Open **http://127.0.0.1:8000** and search. |

---

## Config

| What | Default | Override |
|------|---------|----------|
| OpenSearch | `http://localhost:9200` | `OPENSEARCH_URL` (e.g. `http://opensearch:9200` in Docker/GKE) |
| Index name | `products` | `INDEX_NAME` |

If OpenSearch isn’t up, the search API returns **503** with instructions.

---

## Repo layout

```
├── api/
│   ├── search_service.py   # FastAPI app, UI, search + re-rank
│   └── test_search_service.py
├── ingest/
│   └── ingest_fashion.py   # Build HNSW index from JSONL
├── infra/
│   ├── docker-compose.yml  # OpenSearch (and optional API)
│   └── Dockerfile
├── data/                   # Put meta_Amazon_Fashion.jsonl.gz here
└── requirements.txt
```

---

*Semantic retrieval with OpenSearch, FastAPI, GKE-ready.*
