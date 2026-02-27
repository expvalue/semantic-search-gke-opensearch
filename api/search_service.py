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
*{box-sizing:border-box;margin:0;padding:0}
body{
    font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
    margin:0;
    min-height:100vh;
    background:
        radial-gradient(circle at 0 0,rgba(56,189,248,0.08),transparent 50%),
        radial-gradient(circle at 100% 0,rgba(168,85,247,0.08),transparent 55%),
        radial-gradient(circle at 50% 100%,rgba(59,130,246,0.14),transparent 55%),
        #020617;
    color:#e5e7eb;
    display:flex;
    align-items:flex-start;
    justify-content:center;
    padding:40px 16px;
}
.container{
    width:100%;
    max-width:960px;
}
.card{
    background:rgba(15,23,42,0.9);
    border-radius:20px;
    padding:24px 24px 20px;
    box-shadow:0 24px 80px rgba(15,23,42,0.9);
    border:1px solid rgba(148,163,184,0.25);
    backdrop-filter:blur(16px);
    animation:card-in 320ms ease-out;
}
.header-row{
    display:flex;
    align-items:center;
    justify-content:space-between;
    margin-bottom:18px;
}
h1{
    margin:0;
    font-size:24px;
    letter-spacing:0.02em;
    display:flex;
    align-items:center;
    gap:8px;
}
.badge{
    font-size:11px;
    padding:4px 10px;
    border-radius:999px;
    background:rgba(37,99,235,0.15);
    color:#bfdbfe;
    border:1px solid rgba(59,130,246,0.5);
}
.subtitle{
    margin-top:4px;
    font-size:13px;
    color:#9ca3af;
}
.search-row{
    display:flex;
    gap:10px;
    margin-top:18px;
}
input,button{
    padding:12px 14px;
    border-radius:999px;
    border:none;
    font-size:14px;
}
input{
    flex:1;
    background:#020617;
    color:#e5e7eb;
    border:1px solid rgba(148,163,184,0.4);
    outline:none;
}
input::placeholder{
    color:#6b7280;
}
button{
    background:linear-gradient(135deg,#6366f1,#8b5cf6);
    color:white;
    cursor:pointer;
    font-weight:500;
    padding-inline:20px;
    box-shadow:0 10px 30px rgba(79,70,229,0.6);
    transition:transform .08s ease,box-shadow .08s ease,filter .08s ease;
}
button:hover{
    transform:translateY(-1px);
    box-shadow:0 14px 40px rgba(79,70,229,0.75);
    filter:brightness(1.05);
}
button:active{
    transform:translateY(0);
    box-shadow:0 6px 18px rgba(79,70,229,0.5);
}
.results-meta{
    margin-top:18px;
    font-size:12px;
    color:#9ca3af;
}
.results-meta span{
    color:#e5e7eb;
}
.results{
    margin-top:10px;
}
.result{
    background:#020617;
    margin-top:8px;
    padding:12px 14px;
    border-radius:12px;
    display:flex;
    justify-content:space-between;
    gap:14px;
    border:1px solid rgba(30,64,175,0.3);
    box-shadow:0 4px 18px rgba(15,23,42,0.75);
    font-size:14px;
    opacity:0;
    transform:translateY(4px);
    animation:result-in 220ms ease-out forwards;
}
.result:hover{
    border-color:rgba(96,165,250,0.7);
}
.title{
    flex:1;
}
.score{
    opacity:0.7;
    font-size:12px;
    white-space:nowrap;
}
.empty{
    margin-top:12px;
    font-size:13px;
    color:#9ca3af;
}
@keyframes card-in{
    from{opacity:0;transform:translateY(8px) scale(.98)}
    to{opacity:1;transform:translateY(0) scale(1)}
}
@keyframes result-in{
    from{opacity:0;transform:translateY(4px)}
    to{opacity:1;transform:translateY(0)}
}
@media (max-width:600px){
    .card{padding:18px 16px 16px}
    .search-row{flex-direction:column}
    button{width:100%}
}
</style>
</head>
<body>
<div class="container">
<div class="card">
  <div class="header-row">
    <div>
      <h1>🔎 Semantic Fashion Search</h1>
      <p class="subtitle">Vector + keyword hybrid ranking for Amazon Fashion.</p>
    </div>
    <div class="badge">GKE · OpenSearch · MiniLM</div>
  </div>
  <div class="search-row">
    <input id="query" placeholder="Try: black leather biker jacket" autofocus/>
    <button onclick="search()">Search</button>
  </div>
  <div class="results-meta" id="results-meta"></div>
  <div class="results" id="results"></div>
</div>
</div>

<script>
async function search(){
    const q = document.getElementById("query").value;
    const resDiv = document.getElementById("results");
    const metaDiv = document.getElementById("results-meta");
    resDiv.innerHTML = "";
    metaDiv.innerHTML = "Searching…";
    const r = await fetch(`/search?q=${encodeURIComponent(q)}&k=5`);
    const data = await r.json();
    resDiv.innerHTML = "";
    if(!Array.isArray(data) || !data.length){
        metaDiv.innerHTML = "";
        resDiv.innerHTML = "<div class='empty'>No results. Try a more descriptive query like <span>\"black leather moto jacket\"</span>.</div>";
        return;
    }
    metaDiv.innerHTML = `<span>${data.length}</span> results`;
    data.forEach(item=>{
        resDiv.innerHTML += `
            <div class="result">
                <div class="title">${item.title ?? ""}</div>
                <div class="score">score ${(item.score ?? 0).toFixed(4)}</div>
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