import os
import logging
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import requests
from sentence_transformers import SentenceTransformer

# ------------------------------------------------
# App + Logging
# ------------------------------------------------

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------
# Config
# ------------------------------------------------

OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "http://opensearch-service:9200")
INDEX_NAME = os.getenv("INDEX_NAME", "products")

VECTOR_WEIGHT = 0.4
KEYWORD_WEIGHT = 0.6

# Very small stopword list; everything else is treated as a content word.
STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "for",
    "and",
    "to",
    "in",
    "on",
    "with",
}

model = None


# ------------------------------------------------
# Startup Lifecycle
# ------------------------------------------------

@app.on_event("startup")
def load_model():
    global model
    logger.info("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Model loaded successfully.")


# ------------------------------------------------
# Health Endpoint
# ------------------------------------------------

@app.get("/health")
def health():
    return {"ok": True}


# ------------------------------------------------
# UI Homepage
# ------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Semantic Fashion Search</title>
<style>
body{font-family:system-ui;margin:0;background:#0f172a;color:#f1f5f9}
.container{max-width:900px;margin:60px auto;padding:20px}
.card{background:#1e293b;border-radius:16px;padding:24px}
h1{margin-top:0}
input,button{padding:12px;border-radius:10px;border:none;font-size:14px}
input{width:60%;margin-right:8px}
button{background:#6366f1;color:white;cursor:pointer}
button:hover{background:#4f46e5}
.result{background:#334155;margin-top:10px;padding:12px;border-radius:10px;display:flex;justify-content:space-between;gap:14px}
.score{opacity:0.7;font-size:13px;white-space:nowrap}
</style>
</head>
<body>
<div class="container">
<div class="card">
<h1>🔎 Semantic Fashion Search</h1>
<input id="query" placeholder="Try: black leather biker jacket"/>
<button onclick="search()">Search</button>
<div id="results"></div>
</div>
</div>

<script>
async function search(){
    const q = document.getElementById("query").value;
    const resDiv = document.getElementById("results");
    resDiv.innerHTML = "Searching...";
    const r = await fetch(`/search?q=${encodeURIComponent(q)}&k=5`);
    const data = await r.json();
    resDiv.innerHTML = "";
    if(!Array.isArray(data) || !data.length){
        resDiv.innerHTML = "<div class='result'>No results</div>";
        return;
    }
    data.forEach(item=>{
        resDiv.innerHTML += `
            <div class="result">
                <div>${item.title ?? ""}</div>
                <div class="score">${(item.score ?? 0).toFixed(4)}</div>
            </div>
        `;
    });
}
</script>

</body>
</html>
"""


# ------------------------------------------------
# Cached Embedding
# ------------------------------------------------

@lru_cache(maxsize=1000)
def embed_cached(q: str):
    return model.encode(q).tolist()


# ------------------------------------------------
# Semantic Search (Vector Recall + Manual Re-rank)
# ------------------------------------------------

@app.get("/search")
def search(q: str, k: int = 5):
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if k < 1 or k > 50:
        raise HTTPException(status_code=400, detail="k must be between 1 and 50")

    logger.info(f"Search query received: {q}")

    embedding = embed_cached(q)

    # Step 1: Vector recall (top 50)
    body = {
        "size": 50,
        "query": {
            "knn": {
                "embedding": {
                    "vector": embedding,
                    "k": 50
                }
            }
        }
    }

    try:
        r = requests.post(
            f"{OPENSEARCH_URL}/{INDEX_NAME}/_search",
            json=body,
            timeout=10,
        )

        if r.status_code >= 400:
            logger.error("OpenSearch error: %s", r.text)
            raise HTTPException(status_code=500, detail=r.text)

        hits = r.json().get("hits", {}).get("hits", [])

        query_tokens = q.lower().split()
        content_tokens = [t for t in query_tokens if t not in STOPWORDS]
        if not content_tokens:
            content_tokens = query_tokens

        results = []

        for h in hits:
            title = h["_source"].get("title", "")
            vector_score = h.get("_score", 0)

            title_lower = title.lower()

            # Content word overlap
            content_matches = [t for t in content_tokens if t in title_lower]
            keyword_matches = len(content_matches)

            keyword_score = keyword_matches / max(len(content_tokens), 1)

            # If there is no overlap on important words and the query is reasonably
            # descriptive, apply a penalty so obviously off-topic results are pushed down.
            if keyword_matches == 0 and len(content_tokens) >= 2:
                keyword_score -= 0.5

            # Phrase boost
            if q.lower() in title_lower:
                keyword_score += 1.0

            # Final blended score
            final_score = (
                VECTOR_WEIGHT * vector_score +
                KEYWORD_WEIGHT * keyword_score
            )

            results.append({
                "title": title,
                "score": final_score
            })

        # Manual sorting
        results = sorted(results, key=lambda x: x["score"], reverse=True)

        return results[:k]

    except requests.RequestException as e:
        logger.error("Request failure: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))