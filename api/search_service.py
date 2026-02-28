import os
import logging
import re
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
PHRASE_WEIGHT = 0.15
FUZZY_WEIGHT = 0.05

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

TOKEN_RE = re.compile(r"[a-z0-9]+")

model = None


# ------------------------------------------------
# Startup Lifecycle
# ------------------------------------------------

def get_model():
    global model
    if model is None:
        logger.info("Loading embedding model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Model loaded successfully.")
    return model


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
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Semantic Fashion Search</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet"/>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{
    font-family:'Inter',system-ui,-apple-system,sans-serif;
    margin:0;
    min-height:100vh;
    background:#f7f7fb;
    color:#111827;
    -webkit-font-smoothing:antialiased;
}
.gradient-hero{
    background:radial-gradient(circle at top,rgba(99,102,241,0.18),transparent 55%),
        linear-gradient(120deg,rgba(15,23,42,0.04),rgba(255,255,255,0.95));
}
.wrapper{max-width:72rem;margin:0 auto;padding:0 1.5rem}
header{
    display:flex;
    align-items:center;
    justify-content:space-between;
    padding:2rem 0;
}
.brand small{
    display:block;
    font-size:0.7rem;
    font-weight:600;
    letter-spacing:0.2em;
    text-transform:uppercase;
    color:#4f46e5;
    margin-bottom:0.25rem;
}
.brand h1{font-size:1.5rem;font-weight:600;color:#0f172a}
.live-btn{
    display:inline-flex;
    align-items:center;
    gap:0.5rem;
    padding:0.5rem 1rem;
    font-size:0.875rem;
    font-weight:500;
    color:#475569;
    background:#fff;
    border:1px solid #e2e8f0;
    border-radius:999px;
    cursor:pointer;
    text-decoration:none;
    box-shadow:0 1px 3px rgba(15,23,42,0.06);
    transition:background .15s,border-color .15s;
}
.live-btn:hover{background:#f8fafc;border-color:#cbd5e1}
.glass-card{
    background:rgba(255,255,255,0.75);
    backdrop-filter:blur(12px);
    -webkit-backdrop-filter:blur(12px);
    border:1px solid rgba(255,255,255,0.6);
    border-radius:32px;
    padding:2rem;
    margin-bottom:2rem;
    box-shadow:0 10px 25px rgba(15,23,42,0.08);
    animation:card-in 320ms ease-out;
}
.hero-inner{
    display:flex;
    flex-wrap:wrap;
    gap:1.5rem;
    align-items:flex-start;
    justify-content:space-between;
    margin-bottom:2rem;
}
.hero-text h2{font-size:1.75rem;font-weight:600;color:#0f172a;margin-bottom:0.5rem}
.hero-text p{font-size:0.875rem;color:#64748b;max-width:32rem;line-height:1.5}
.signal-blend{
    min-width:260px;
    padding:1rem;
    background:rgba(255,255,255,0.85);
    border-radius:24px;
    border:1px solid #f1f5f9;
}
.signal-blend .label{font-size:0.7rem;font-weight:600;color:#64748b;margin-bottom:0.5rem}
.signal-blend .desc{font-size:0.875rem;color:#334155;line-height:1.45}
.signal-blend .note{font-size:0.75rem;color:#94a3b8;margin-top:0.75rem}
.search-row{
    display:flex;
    gap:0.75rem;
    align-items:center;
    flex-wrap:wrap;
}
.search-wrap{
    flex:1;
    min-width:200px;
    position:relative;
}
.search-wrap svg{
    position:absolute;
    left:1rem;
    top:50%;
    transform:translateY(-50%);
    width:1rem;
    height:1rem;
    color:#94a3b8;
    pointer-events:none;
}
.search-wrap input{
    width:100%;
    height:3rem;
    padding:0 1rem 0 2.75rem;
    font-size:0.875rem;
    border:1px solid #e2e8f0;
    border-radius:999px;
    background:#fff;
    color:#0f172a;
    outline:none;
    transition:border-color .15s,box-shadow .15s;
}
.search-wrap input:focus{border-color:#818cf8;box-shadow:0 0 0 3px rgba(99,102,241,0.15)}
.search-wrap input::placeholder{color:#94a3b8}
.btn-search{
    height:3rem;
    padding:0 1.5rem;
    font-size:0.875rem;
    font-weight:600;
    color:#fff;
    background:#4f46e5;
    border:none;
    border-radius:999px;
    cursor:pointer;
    box-shadow:0 4px 14px rgba(79,70,229,0.4);
    transition:transform .08s,box-shadow .08s,background .08s;
}
.btn-search:hover{background:#4338ca;transform:translateY(-1px);box-shadow:0 6px 20px rgba(79,70,229,0.45)}
.btn-search:active{transform:translateY(0)}
.pills{
    display:flex;
    flex-wrap:wrap;
    gap:0.5rem;
    margin-top:1.5rem;
}
.pill{
    padding:0.4rem 0.9rem;
    font-size:0.8rem;
    font-weight:500;
    border-radius:999px;
    border:1px solid #e2e8f0;
    background:#fff;
    color:#475569;
    cursor:pointer;
    transition:background .15s,border-color .15s,color .15s;
}
.pill:hover{background:#f8fafc;border-color:#cbd5e1}
.pill.active{background:#4f46e5;border-color:#4f46e5;color:#fff}
.results-section{margin-top:2rem;padding-bottom:3rem}
.results-header{
    display:flex;
    flex-wrap:wrap;
    gap:1rem;
    align-items:center;
    justify-content:space-between;
    margin-bottom:1rem;
}
.results-header h3{font-size:1.125rem;font-weight:600;color:#0f172a}
.results-header p{font-size:0.875rem;color:#64748b;margin-top:0.25rem}
.badge-mode{
    font-size:0.75rem;
    padding:0.35rem 0.75rem;
    border-radius:999px;
    background:#f1f5f9;
    color:#475569;
    font-weight:500;
}
.grid-results{
    display:grid;
    gap:1.5rem;
    margin-top:1.5rem;
}
@media (min-width:640px){.grid-results{grid-template-columns:repeat(2,1fr)}}
@media (min-width:1024px){.grid-results{grid-template-columns:repeat(3,1fr)}}
.card-result{
    background:#fff;
    border:1px solid #f1f5f9;
    border-radius:24px;
    padding:1.25rem;
    box-shadow:0 6px 20px rgba(15,23,42,0.06);
    animation:result-in 220ms ease-out forwards;
    opacity:0;
}
.card-result:nth-child(1){animation-delay:0ms}
.card-result:nth-child(2){animation-delay:40ms}
.card-result:nth-child(3){animation-delay:80ms}
.card-result:nth-child(4){animation-delay:120ms}
.card-result:nth-child(5){animation-delay:160ms}
.card-result:hover{border-color:#e2e8f0;box-shadow:0 10px 30px rgba(15,23,42,0.08)}
.card-result .cat{font-size:0.7rem;font-weight:600;letter-spacing:0.05em;color:#94a3b8;margin-bottom:0.35rem}
.card-result .title{font-size:0.95rem;font-weight:600;color:#0f172a;line-height:1.4;margin-bottom:0.5rem}
.card-result .meta{display:flex;align-items:center;justify-content:space-between;font-size:0.8rem;color:#64748b;margin-bottom:0.75rem}
.card-result .score-tag{font-size:0.7rem;padding:0.25rem 0.5rem;background:#eef2ff;color:#4f46e5;border-radius:999px;font-weight:500}
.btn-why{
    width:100%;
    padding:0.5rem 1rem;
    font-size:0.8rem;
    font-weight:500;
    color:#475569;
    background:transparent;
    border:1px solid #e2e8f0;
    border-radius:999px;
    cursor:pointer;
    transition:background .15s,border-color .15s;
}
.btn-why:hover{background:#f8fafc;border-color:#cbd5e1}
.card-result.loading{pointer-events:none;opacity:0.9}
.skeleton{display:block;border-radius:10px;background:linear-gradient(90deg,#f1f5f9 0%,#e2e8f0 50%,#f1f5f9 100%);background-size:200% 100%;animation:shimmer 1.1s linear infinite}
.skeleton.title{height:0.9rem;margin-bottom:0.6rem}
.skeleton.meta{height:0.7rem;width:70%;margin-bottom:0.7rem}
.skeleton.btn{height:2rem;border-radius:999px}
.empty-msg{margin-top:1rem;font-size:0.875rem;color:#64748b}
.dialog-overlay{
    position:fixed;
    inset:0;
    z-index:50;
    background:rgba(15,23,42,0.4);
    backdrop-filter:blur(4px);
    display:none;
    align-items:center;
    justify-content:center;
    padding:1rem;
}
.dialog-overlay.open{display:flex}
.dialog-box{
    background:#fff;
    border-radius:24px;
    padding:1.5rem;
    max-width:28rem;
    width:100%;
    box-shadow:0 24px 48px rgba(15,23,42,0.18);
    animation:dialog-in 200ms ease-out;
}
.dialog-box h4{font-size:1rem;font-weight:600;color:#0f172a;margin-bottom:0.5rem}
.dialog-box p{font-size:0.875rem;color:#64748b;line-height:1.5}
.dialog-close{
    float:right;
    margin:-0.5rem -0.5rem 0.5rem 0.5rem;
    padding:0.5rem;
    border:none;
    background:transparent;
    color:#64748b;
    cursor:pointer;
    border-radius:999px;
}
.dialog-close:hover{background:#f1f5f9;color:#0f172a}
@keyframes card-in{from{opacity:0;transform:translateY(8px) scale(.98)}to{opacity:1;transform:translateY(0) scale(1)}}
@keyframes result-in{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
@keyframes dialog-in{from{opacity:0;transform:scale(0.96)}to{opacity:1;transform:scale(1)}}
@keyframes shimmer{0%{background-position:200% 0}100%{background-position:-200% 0}}
@media (max-width:600px){
    .glass-card{padding:1.25rem}
    .search-row{flex-direction:column;align-items:stretch}
    .btn-search{width:100%}
}
</style>
</head>
<body>
<div class="gradient-hero">
  <header class="wrapper">
    <div class="brand">
      <small>Semantic search</small>
      <h1>Fashion search with vector + keyword ranking</h1>
    </div>
    <a class="live-btn" href="/" aria-label="Live">Live</a>
  </header>

  <section class="wrapper">
    <div class="glass-card">
      <div class="hero-inner">
        <div class="hero-text">
          <h2>Find the perfect pick.</h2>
          <p>Semantic search over Amazon Fashion. Results blend vector similarity (MiniLM) with keyword overlap and phrase boost.</p>
        </div>
        <div class="signal-blend">
          <div class="label">Signal blend</div>
          <div class="desc">MiniLM embeddings · OpenSearch KNN · BM25/phrase/fuzzy lexical re-rank</div>
          <div class="note">Only shipping working pieces: hybrid recall + explainable scoring.</div>
        </div>
      </div>

      <form class="search-row" onsubmit="return runSearch(event)">
        <div class="search-wrap">
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>
          <input id="query" type="text" placeholder="What are you looking for?" autofocus/>
        </div>
        <button type="submit" class="btn-search">Explore</button>
      </form>

      <div class="pills">
        <button type="button" class="pill active" data-q="">All</button>
        <button type="button" class="pill" data-q="jacket">Jackets</button>
        <button type="button" class="pill" data-q="dress">Dresses</button>
        <button type="button" class="pill" data-q="shoes">Shoes</button>
        <button type="button" class="pill" data-q="bag">Bags</button>
        <button type="button" class="pill" data-q="accessories">Accessories</button>
      </div>
    </div>
  </section>
</div>

<section class="wrapper results-section">
  <div class="results-header">
    <div>
      <h3 id="results-title">Recommended for you</h3>
      <p id="results-subtitle">Run a search or pick a category above.</p>
      <p id="results-hint" style="font-size:0.78rem;color:#94a3b8;margin-top:0.35rem;">Tip: click “Why this?” to see semantic vs lexical contributions.</p>
    </div>
    <span class="badge-mode" id="badge-mode">GKE · OpenSearch · MiniLM</span>
  </div>

  <div class="grid-results" id="results"></div>
  <p class="empty-msg" id="empty-msg" style="display:none">No results yet. Try a more specific query like &quot;black leather jacket&quot;.</p>
</section>

<div class="dialog-overlay" id="dialog" aria-hidden="true">
  <div class="dialog-box">
    <button type="button" class="dialog-close" onclick="closeDialog()" aria-label="Close">&times;</button>
    <h4 id="dialog-title">Why this result?</h4>
    <p id="dialog-body">Score blends vector similarity and keyword overlap.</p>
  </div>
</div>

<script>
var currentQuery = '';
function runSearch(e){ e.preventDefault(); search(); return false; }
function setPills(){
  document.querySelectorAll('.pills .pill').forEach(function(btn){
    btn.classList.remove('active');
    if ((btn.dataset.q || '') === currentQuery) btn.classList.add('active');
  });
}
function renderLoadingCards(container){
  container.innerHTML = "";
  for (var i=0; i<3; i++) {
    var card = document.createElement('div');
    card.className = "card-result loading";
    card.innerHTML = "<span class=\"skeleton title\"></span><span class=\"skeleton meta\"></span><span class=\"skeleton btn\"></span>";
    container.appendChild(card);
  }
}
function search(optionalQ){
  var q = (optionalQ !== undefined ? optionalQ : document.getElementById('query').value).trim();
  currentQuery = q;
  setPills();
  var resEl = document.getElementById('results');
  var titleEl = document.getElementById('results-title');
  var subEl = document.getElementById('results-subtitle');
  var emptyEl = document.getElementById('empty-msg');
  resEl.innerHTML = '';
  emptyEl.style.display = 'none';
  if (!q){ titleEl.textContent = 'Recommended for you'; subEl.textContent = 'Run a search or pick a category above.'; return; }
  titleEl.textContent = 'Searching…';
  subEl.textContent = 'Blending semantic + lexical signals';
  renderLoadingCards(resEl);
  fetch('/search?q=' + encodeURIComponent(q) + '&k=6')
    .then(function(r){ return r.json(); })
    .then(function(data){
      titleEl.textContent = data.length ? 'Top matches for "' + q + '"' : 'No results for "' + q + '"';
      subEl.textContent = data.length ? data.length + ' results' : 'Try a different query or category.';
      if (!Array.isArray(data) || !data.length){ emptyEl.style.display = 'block'; return; }
      data.forEach(function(item, i){
        var card = document.createElement('div');
        card.className = 'card-result';
        var t = (item.title || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
        var explain = item.explain || {};
        card.innerHTML = '<div class="cat">PRODUCT</div><div class="title">' + t + '</div><div class="meta"><span>Relevance</span><span class="score-tag">' + (item.score != null ? item.score.toFixed(4) : '—') + '</span></div><button type="button" class="btn-why" data-title="' + (item.title || '').replace(/"/g,'&quot;') + '" data-score="' + (item.score != null ? item.score : '') + '" data-semantic="' + (explain.semantic != null ? explain.semantic : '') + '" data-lexical="' + (explain.lexical != null ? explain.lexical : '') + '" data-phrase="' + (explain.phrase != null ? explain.phrase : '') + '" data-fuzzy="' + (explain.fuzzy != null ? explain.fuzzy : '') + '">Why this?</button>';
        card.querySelector('.btn-why').onclick = function(){ openDialog(this.dataset); };
        resEl.appendChild(card);
      });
    })
    .catch(function(){ titleEl.textContent = 'Something went wrong'; subEl.textContent = ''; emptyEl.style.display = 'block'; });
}
document.querySelectorAll('.pills .pill').forEach(function(btn){
  btn.addEventListener('click', function(){ var q = this.dataset.q || ''; document.getElementById('query').value = q; search(q); });
});
function openDialog(data){
  document.getElementById('dialog-title').textContent = 'Why this result?';
  var title = data.title || '';
  var safeTitle = title.replace(/</g,'&lt;').replace(/>/g,'&gt;').substring(0,100);
  if (title.length > 100) safeTitle += '…';
  var details = 'Semantic: <strong>' + (data.semantic || '0') + '</strong> · Lexical: <strong>' + (data.lexical || '0') + '</strong> · Phrase: <strong>' + (data.phrase || '0') + '</strong> · Fuzzy: <strong>' + (data.fuzzy || '0') + '</strong>';
  document.getElementById('dialog-body').innerHTML = (safeTitle ? '<strong style="color:#0f172a">' + safeTitle + '</strong><br><br>' : '') + 'This item ranked highly from a blend of <strong>vector similarity</strong> (semantic match) and <strong>lexical relevance</strong>. ' + details + '. Total: <strong>' + (data.score || '—') + '</strong>.';
  document.getElementById('dialog').classList.add('open'); document.getElementById('dialog').setAttribute('aria-hidden','false');
}
function closeDialog(){ document.getElementById('dialog').classList.remove('open'); document.getElementById('dialog').setAttribute('aria-hidden','true'); }
document.getElementById('dialog').addEventListener('click', function(e){ if (e.target === this) closeDialog(); });
</script>
</body>
</html>
"""


# ------------------------------------------------
# Cached Embedding
# ------------------------------------------------

@lru_cache(maxsize=1000)
def embed_cached(q: str):
    return get_model().encode(q).tolist()


def tokenize(text: str):
    return TOKEN_RE.findall((text or "").lower())


def normalized_knn_score(raw_score: float):
    # OpenSearch knn scores are not naturally bounded; this provides a stable [0,1) range.
    return raw_score / (1.0 + max(raw_score, 0.0))


def jaccard_similarity(a_tokens, b_tokens):
    a_set, b_set = set(a_tokens), set(b_tokens)
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def normalize_text_score(scores_by_id):
    if not scores_by_id:
        return {}
    max_score = max(scores_by_id.values())
    if max_score <= 0:
        return {doc_id: 0.0 for doc_id in scores_by_id}
    return {doc_id: score / max_score for doc_id, score in scores_by_id.items()}


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

    # Step 1: Vector recall
    vector_body = {
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

    # Step 2: Lexical recall (BM25 + phrase + fuzzy)
    text_body = {
        "size": 50,
        "query": {
            "bool": {
                "should": [
                    {"match": {"title": {"query": q, "boost": 1.0}}},
                    {"match_phrase": {"title": {"query": q, "boost": 2.0}}},
                    {"match": {"title": {"query": q, "fuzziness": "AUTO", "boost": 0.4}}}
                ],
                "minimum_should_match": 1
            }
        }
    }

    try:
        vector_response = requests.post(
            f"{OPENSEARCH_URL}/{INDEX_NAME}/_search",
            json=vector_body,
            timeout=10,
        )
        text_response = requests.post(
            f"{OPENSEARCH_URL}/{INDEX_NAME}/_search",
            json=text_body,
            timeout=10,
        )

        if vector_response.status_code >= 400:
            logger.error("OpenSearch vector error: %s", vector_response.text)
            raise HTTPException(status_code=500, detail=vector_response.text)
        if text_response.status_code >= 400:
            logger.error("OpenSearch lexical error: %s", text_response.text)
            raise HTTPException(status_code=500, detail=text_response.text)

        vector_hits = vector_response.json().get("hits", {}).get("hits", [])
        text_hits = text_response.json().get("hits", {}).get("hits", [])

        text_scores = normalize_text_score(
            {
                h.get("_id"): h.get("_score", 0)
                for h in text_hits
                if h.get("_id")
            }
        )
        vector_scores = {
            h.get("_id"): normalized_knn_score(h.get("_score", 0))
            for h in vector_hits
            if h.get("_id")
        }

        candidates = {}
        for h in vector_hits + text_hits:
            doc_id = h.get("_id")
            if doc_id and doc_id not in candidates:
                candidates[doc_id] = h

        query_tokens = tokenize(q)
        content_tokens = [t for t in query_tokens if t not in STOPWORDS]
        if not content_tokens:
            content_tokens = query_tokens

        results = []

        for doc_id, h in candidates.items():
            title = h["_source"].get("title", "")
            vector_score = vector_scores.get(doc_id, 0.0)
            lexical_score = text_scores.get(doc_id, 0.0)

            title_tokens = tokenize(title)
            title_lower = " ".join(title_tokens)

            # Content word overlap
            title_token_set = set(title_tokens)
            content_matches = [t for t in content_tokens if t in title_token_set]
            keyword_matches = len(content_matches)

            keyword_score = keyword_matches / max(len(content_tokens), 1)
            fuzzy_score = jaccard_similarity(content_tokens, title_tokens)

            # If there is no overlap on important words and the query is reasonably
            # descriptive, apply a penalty so obviously off-topic results are pushed down.
            if keyword_matches == 0 and len(content_tokens) >= 2:
                keyword_score -= 0.5

            # Phrase boost
            if q.lower() in title.lower():
                keyword_score += 1.0

            phrase_boost = 1.0 if q.lower() in title.lower() else 0.0

            semantic_component = VECTOR_WEIGHT * vector_score
            lexical_component = KEYWORD_WEIGHT * (0.65 * lexical_score + 0.35 * keyword_score)
            phrase_component = PHRASE_WEIGHT * phrase_boost
            fuzzy_component = FUZZY_WEIGHT * fuzzy_score

            # Final blended score
            final_score = semantic_component + lexical_component + phrase_component + fuzzy_component

            results.append({
                "id": doc_id,
                "title": title,
                "score": final_score,
                "explain": {
                    "semantic": round(semantic_component, 4),
                    "lexical": round(lexical_component, 4),
                    "phrase": round(phrase_component, 4),
                    "fuzzy": round(fuzzy_component, 4),
                }
            })

        # Manual sorting
        results = sorted(results, key=lambda x: x["score"], reverse=True)

        return results[:k]

    except requests.RequestException as e:
        logger.error("Request failure: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))
