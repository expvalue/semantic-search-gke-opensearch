import gzip
import json
import os
import time
import requests
from sentence_transformers import SentenceTransformer


# =============================================================================
# config
# =============================================================================

# local dev uses localhost; in docker set OPENSEARCH_URL=http://opensearch:9200
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "http://localhost:9200")
INDEX_NAME = os.getenv("INDEX_NAME", "products")
DIM = 384

# HNSW params: bigger ef_construction = better graph but slower to build. m = links per node.
HNSW_EF_CONSTRUCTION = 200
HNSW_M = 24
# how many candidates to look at when searching (more = better recall, slower)
HNSW_EF_SEARCH = 200

BATCH_SIZE = 200
EMBED_BATCH_SIZE = 32

model = SentenceTransformer("all-MiniLM-L6-v2")
session = requests.Session()


# =============================================================================
# opensearch readiness
# =============================================================================

def wait_for_opensearch(timeout_s=120):
    """poll until OpenSearch is up or we hit the timeout"""
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            r = session.get(f"{OPENSEARCH_URL}", timeout=5)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    return False


# =============================================================================
# index (recreate with HNSW + cosine)
# =============================================================================

def recreate_index():
    """drop the index if it exists and create it with HNSW knn_vector mapping (cosine space)"""
    session.delete(f"{OPENSEARCH_URL}/{INDEX_NAME}", timeout=30)

    mapping = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": HNSW_EF_SEARCH
            }
        },
        "mappings": {
            "properties": {
                "title": {"type": "text"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": DIM,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "nmslib",
                        "parameters": {
                            "ef_construction": HNSW_EF_CONSTRUCTION,
                            "m": HNSW_M
                        }
                    }
                }
            }
        }
    }

    r = session.put(
        f"{OPENSEARCH_URL}/{INDEX_NAME}",
        json=mapping,
        timeout=30
    )
    r.raise_for_status()


# same prefix as search_service so query and doc embeddings line up
DOC_EMBED_PREFIX = "Clothing, fashion product or accessory: "


def build_text(doc):
    parts = []
    if doc.get("title"):
        parts.append(str(doc["title"]))

    for key in ["description", "features"]:
        val = doc.get(key)
        if isinstance(val, list):
            parts.extend([str(x) for x in val if x])
        elif isinstance(val, str):
            parts.append(val)

    if doc.get("store"):
        parts.append(str(doc["store"]))

    raw = " ".join(parts).strip()
    return (DOC_EMBED_PREFIX + raw) if raw else raw


# =============================================================================
# bulk index
# =============================================================================

def bulk_index(items, embs):
    """send one bulk request: each item gets title + embedding, NDJSON with trailing newline"""
    lines = []
    for item, emb in zip(items, embs):
        lines.append(json.dumps({"index": {"_index": INDEX_NAME}}))
        lines.append(json.dumps({
            "title": item["title"],
            "embedding": emb
        }))
    payload = "\n".join(lines) + "\n"

    r = session.post(
        f"{OPENSEARCH_URL}/_bulk",
        headers={"Content-Type": "application/json"},
        data=payload,
        timeout=120
    )
    r.raise_for_status()

    resp = r.json()
    if resp.get("errors"):
        for it in resp.get("items", []):
            idx = it.get("index", {})
            if idx.get("error"):
                raise RuntimeError(f"Bulk error example: {idx['error']}")
        raise RuntimeError("Bulk ingest had errors, but couldn't extract example.")


# =============================================================================
# main
# =============================================================================

def main():
    """wait for OpenSearch, recreate index, stream the gzipped jsonl and index in batches"""
    if not wait_for_opensearch():
        raise RuntimeError("OpenSearch never became ready (timeout).")

    print("Recreating index with knn_vector mapping...")
    recreate_index()
    print("Index created.")

    buffer = []
    indexed = 0

    with gzip.open("data/meta_Amazon_Fashion.jsonl.gz", "rt", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            doc = json.loads(line)
            text = build_text(doc)
            if not text:
                continue

            buffer.append({"title": doc.get("title"), "text": text})

            if len(buffer) >= BATCH_SIZE:
                texts = [x["text"] for x in buffer]
                embs = model.encode(texts, batch_size=EMBED_BATCH_SIZE, show_progress_bar=False)
                embs = [e.tolist() for e in embs]

                bulk_index(buffer, embs)
                indexed += len(buffer)
                print(f"Indexed {indexed} docs (read line {i})")
                buffer = []

    if buffer:
        texts = [x["text"] for x in buffer]
        embs = model.encode(texts, batch_size=EMBED_BATCH_SIZE, show_progress_bar=False)
        embs = [e.tolist() for e in embs]
        bulk_index(buffer, embs)
        indexed += len(buffer)
        print(f"Indexed {indexed} docs (final flush)")

    print("Done.")


if __name__ == "__main__":
    main()