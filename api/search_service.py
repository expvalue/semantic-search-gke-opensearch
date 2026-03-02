import math
import os
import re
import logging
from functools import lru_cache

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import requests
from sentence_transformers import SentenceTransformer


# =============================================================================
# config
# =============================================================================

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "http://localhost:9200")
INDEX_NAME = os.getenv("INDEX_NAME", "products")

# RRF: how much to smooth the two rank lists. 60 is a sane default.
RRF_K = 60
# how many neighbors to consider when searching the vector index. bump up if you want better recall and don't mind slower
KNN_EF_SEARCH = 200

# wrap the query in this so the embedding sits in the same "product description" space as the docs (ingest uses the same prefix)
QUERY_CONTEXT_PREFIX = "Clothing, fashion product or accessory: "
QUERY_CONTEXT_SUFFIX = ""

STOPWORDS = {
    "the", "a", "an", "of", "for", "and", "to", "in", "on", "with",
}

# so "men" doesn't surface women's stuff and vice versa
GENDER_MALE_TERMS = {"men", "mens", "man", "male", "boys", "boy"}
GENDER_FEMALE_TERMS = {"women", "womens", "woman", "female", "girls", "girl", "ladies", "lady"}
ATTRIBUTE_MISMATCH_PENALTY = 1.5
ATTRIBUTE_MATCH_BONUS = 0.6

model = None


# =============================================================================
# app lifecycle
# =============================================================================

@app.on_event("startup")
def load_model():
    """load the sentence-transformers model once at startup (used for query embedding)"""
    global model
    logger.info("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Model loaded.")


@app.get("/health")
def health():
    return {"ok": True}


# =============================================================================
# embedding
# =============================================================================

def _query_for_embedding(raw_query: str) -> str:
    """stick the prefix on so we're embedding in the same space as the product text"""
    q = (raw_query or "").strip()
    if not q:
        return q
    return QUERY_CONTEXT_PREFIX + q + QUERY_CONTEXT_SUFFIX


@lru_cache(maxsize=1000)
def embed_cached(query: str):
    """cache embeddings per query so we don't re-encode the same thing"""
    return model.encode(query).tolist()


# =============================================================================
# search ui (single-page html + js)
# =============================================================================

@app.get("/", response_class=HTMLResponse)
def home():
    """serve the search page: search bar, category pills, results grid, why-this dialog"""
    return r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>GKE Search · Semantic Apparel Intelligence</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet"/>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#06070A;
  --glass:rgba(255,255,255,0.06);
  --glass-strong:rgba(255,255,255,0.10);
  --text:#FFFFFF;
  --text-2:#A8AFBA;
  --text-3:#6D7582;
  --accent:#6366f1;
  --accent-glow:rgba(99,102,241,0.35);
  --accent-soft:rgba(99,102,241,0.18);
  --danger:#EF4444;
  --radius:14px;
  --radius-lg:18px;
  --shadow:0 24px 60px rgba(0,0,0,0.45);
  --shadow-inner:inset 0 1px 0 rgba(255,255,255,0.06);
}
body{font-family:'Inter',system-ui,sans-serif;min-height:100vh;background:var(--bg);color:var(--text);-webkit-font-smoothing:antialiased;line-height:1.5}
.dash{display:grid;grid-template-columns:260px 1fr;min-height:100vh}
@media (max-width:1024px){.dash{grid-template-columns:1fr}}
.sidebar{width:260px;background:var(--glass);backdrop-filter:blur(16px);border-right:1px solid rgba(255,255,255,0.06);padding:1.5rem 1.25rem;position:sticky;top:0;height:100vh}
@media (max-width:1024px){.sidebar{display:none}}
.sidebar .logo{display:flex;align-items:center;gap:0.5rem;margin-bottom:0.35rem}
.sidebar .logo-dot{width:8px;height:8px;border-radius:50%;background:var(--accent);box-shadow:0 0 12px var(--accent-glow)}
.sidebar .brand{font-size:1.125rem;font-weight:700;color:var(--text);letter-spacing:-0.02em}
.sidebar .tagline{font-size:0.7rem;color:var(--text-3);letter-spacing:0.05em}
.main{padding:1.5rem 1.75rem;overflow:auto}
.glass-card{background:var(--glass);backdrop-filter:blur(18px);border-radius:var(--radius-lg);padding:2rem 2.25rem;margin-bottom:1.5rem;box-shadow:var(--shadow);border:1px solid rgba(255,255,255,0.06);box-shadow:var(--shadow),var(--shadow-inner)}
.hero-title{font-size:1.5rem;font-weight:700;color:var(--text);letter-spacing:-0.03em;margin-bottom:0.35rem}
.hero-sub{font-size:0.875rem;color:var(--text-2);margin-bottom:1.5rem;line-height:1.5}
.search-row{display:flex;gap:0.75rem;align-items:stretch;margin-bottom:1rem}
.search-wrap{flex:1;min-width:200px;position:relative}
.search-wrap svg{position:absolute;left:1rem;top:50%;transform:translateY(-50%);width:1.125rem;height:1.125rem;color:var(--text-3);pointer-events:none}
.search-wrap input{width:100%;height:52px;padding:0 1rem 0 2.75rem;font-size:0.9375rem;background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.08);border-radius:16px;color:var(--text);outline:none;transition:border-color 0.2s,box-shadow 0.2s}
.search-wrap input:focus{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-soft)}
.search-wrap input::placeholder{color:var(--text-3)}
.btn-search{height:52px;padding:0 1.5rem;font-size:0.9375rem;font-weight:600;color:#fff;background:var(--accent);border:none;border-radius:999px;cursor:pointer;transition:all 0.2s;box-shadow:0 8px 24px var(--accent-glow)}
.btn-search:hover{filter:brightness(1.08);transform:translateY(-1px);box-shadow:0 12px 32px var(--accent-glow)}
.intent-strip{display:flex;flex-wrap:wrap;gap:0.5rem;align-items:center;margin-top:1rem}
.intent-tag{font-size:0.75rem;padding:0.35rem 0.7rem;background:var(--glass);border-radius:999px;color:var(--text-2);border:1px solid rgba(255,255,255,0.06)}
.confidence-bar{height:4px;background:rgba(255,255,255,0.08);border-radius:999px;margin-top:1rem;overflow:hidden}
.confidence-fill{height:100%;background:var(--accent);border-radius:999px;width:78%;box-shadow:0 0 12px var(--accent-glow);transition:width 0.3s}
.pills{display:flex;flex-wrap:wrap;gap:0.5rem;margin-top:1.25rem}
.pill{padding:0.45rem 1rem;font-size:0.8125rem;font-weight:500;border-radius:999px;background:var(--glass);border:1px solid rgba(255,255,255,0.08);color:var(--text-2);cursor:pointer;transition:all 0.2s}
.pill:hover{background:var(--glass-strong);color:var(--text)}
.pill.active{background:var(--accent-soft);color:var(--accent);border-color:rgba(99,102,241,0.4);box-shadow:0 0 0 1px rgba(99,102,241,0.25)}
.results-header{display:flex;flex-wrap:wrap;gap:1rem;align-items:center;justify-content:space-between;margin-bottom:1.25rem}
.results-header h3{font-size:1.125rem;font-weight:700;color:var(--text);letter-spacing:-0.02em}
.results-header p{font-size:0.8125rem;color:var(--text-3);margin-top:0.2rem}
.grid-results{display:grid;gap:1.25rem;grid-template-columns:repeat(2,1fr)}
@media (min-width:900px){.grid-results{grid-template-columns:repeat(2,1fr)}}
.card-result{background:var(--glass);backdrop-filter:blur(12px);border-radius:var(--radius-lg);padding:1.25rem;border:1px solid rgba(255,255,255,0.06);box-shadow:0 12px 40px rgba(0,0,0,0.35);transition:all 0.2s}
.card-result:hover{border-color:rgba(255,255,255,0.1);box-shadow:0 16px 48px rgba(0,0,0,0.4)}
.card-result .cat{font-size:0.65rem;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;color:var(--text-3);margin-bottom:0.4rem}
.card-result .title{font-size:0.9rem;font-weight:600;color:var(--text);line-height:1.4;margin-bottom:0.75rem}
.score-bar-wrap{margin-bottom:0.75rem}
.score-bar{height:6px;background:rgba(255,255,255,0.08);border-radius:999px;overflow:hidden}
.score-bar-fill{height:100%;background:linear-gradient(90deg,var(--accent),rgba(99,102,241,0.6));border-radius:999px;box-shadow:0 0 10px var(--accent-glow)}
.card-result .meta{font-size:0.75rem;color:var(--text-3);margin-bottom:0.75rem}
.btn-why{padding:0.5rem 0.9rem;font-size:0.75rem;font-weight:500;color:var(--accent);background:transparent;border:1px solid rgba(99,102,241,0.4);border-radius:999px;cursor:pointer;transition:all 0.2s;width:100%}
.btn-why:hover{background:var(--accent-soft);border-color:var(--accent)}
.empty-msg,.err-msg{font-size:0.875rem;color:var(--text-3);line-height:1.5;margin-top:0.5rem}
.err-msg{color:var(--danger)}
.dialog-overlay{position:fixed;inset:0;z-index:50;background:rgba(0,0,0,0.6);backdrop-filter:blur(8px);display:none;align-items:center;justify-content:center;padding:1.5rem}
.dialog-overlay.open{display:flex}
.dialog-box{background:var(--glass-strong);backdrop-filter:blur(20px);border-radius:var(--radius-lg);padding:1.75rem;max-width:24rem;width:100%;border:1px solid rgba(255,255,255,0.08);box-shadow:var(--shadow)}
.dialog-box h4{font-size:1rem;font-weight:700;color:var(--text);margin-bottom:0.6rem}
.dialog-box p{font-size:0.875rem;color:var(--text-2);line-height:1.55}
.dialog-close{float:right;margin:-0.25rem -0.25rem 0.5rem 0.5rem;padding:0.5rem;border:none;background:transparent;color:var(--text-3);cursor:pointer;border-radius:999px}
.dialog-close:hover{background:var(--glass);color:var(--text)}
</style>
</head>
<body>
<div class="dash">
<aside class="sidebar">
  <div class="logo"><span class="logo-dot"></span><span class="brand">GKE Search</span></div>
  <p class="tagline">Semantic Apparel Intelligence</p>
</aside>
<main class="main">
  <div class="glass-card">
    <h1 class="hero-title">Semantic Clothing Search</h1>
    <p class="hero-sub">Understands intent, vibe, seasonality, and style similarity.</p>
    <form class="search-row" onsubmit="return runSearch(event)">
      <div class="search-wrap">
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>
        <input id="query" type="text" placeholder="e.g. minimal black oversized tee for summer" autofocus/>
      </div>
      <button type="submit" class="btn-search">Search</button>
    </form>
    <div class="intent-strip" id="intent-strip">
      <span class="intent-tag">Style: —</span><span class="intent-tag">Color: —</span><span class="intent-tag">Fit: —</span><span class="intent-tag">Season: —</span>
    </div>
    <div class="confidence-bar"><div class="confidence-fill" id="confidence-fill"></div></div>
    <div class="pills">
      <button type="button" class="pill active" data-q="">All</button>
      <button type="button" class="pill" data-q="jacket">Jackets</button>
      <button type="button" class="pill" data-q="dress">Dresses</button>
      <button type="button" class="pill" data-q="shoes">Shoes</button>
      <button type="button" class="pill" data-q="bag">Bags</button>
    </div>
  </div>
  <div class="results-header">
    <div>
      <h3 id="results-title">Recommended for you</h3>
      <p id="results-subtitle">Run a search or pick a category above.</p>
    </div>
  </div>
  <div class="grid-results" id="results"></div>
  <p class="empty-msg" id="empty-msg" style="display:none">No results yet. Try a more specific query or check that OpenSearch is running.</p>
  <p class="err-msg" id="err-msg" style="display:none"></p>
</main>
</div>
<div class="dialog-overlay" id="dialog" aria-hidden="true">
  <div class="dialog-box">
    <button type="button" class="dialog-close" onclick="closeDialog()" aria-label="Close">&times;</button>
    <h4 id="dialog-title">Why this result?</h4>
    <p id="dialog-body">Relevance blend: vector similarity + keyword overlap.</p>
  </div>
</div>
<script>
function runSearch(e){ e.preventDefault(); search(); return false; }
function setPills(){
  var q = document.getElementById("query").value.trim().toLowerCase();
  document.querySelectorAll(".pills .pill").forEach(function(btn){
    var d = (btn.dataset.q || "").toLowerCase();
    btn.classList.toggle("active", d === q || (d === "" && !q));
  });
}
function updateIntentStrip(q){
  var strip = document.getElementById("intent-strip");
  if (!strip) return;
  var tokens = q ? q.split(/\s+/).filter(function(t){ return t.length > 1; }).slice(0,5) : [];
  if (tokens.length === 0) { strip.innerHTML = "<span class=\"intent-tag\">Style: —</span><span class=\"intent-tag\">Color: —</span><span class=\"intent-tag\">Fit: —</span>"; return; }
  strip.innerHTML = tokens.map(function(t){ return "<span class=\"intent-tag\">" + t + "</span>"; }).join("");
  var fill = document.getElementById("confidence-fill");
  if (fill) fill.style.width = Math.min(60 + tokens.length * 8, 95) + "%";
}
function search(optionalQ){
  var q = (optionalQ !== undefined ? optionalQ : document.getElementById("query").value.trim());
  document.getElementById("query").value = q;
  setPills();
  updateIntentStrip(q);
  var resEl = document.getElementById("results");
  var titleEl = document.getElementById("results-title");
  var subEl = document.getElementById("results-subtitle");
  var emptyEl = document.getElementById("empty-msg");
  var errEl = document.getElementById("err-msg");
  resEl.innerHTML = "";
  emptyEl.style.display = "none";
  errEl.style.display = "none";
  if (!q) { titleEl.textContent = "Recommended for you"; subEl.textContent = "Run a search or pick a category above."; return; }
  titleEl.textContent = "Searching…";
  subEl.textContent = "";
  fetch("/search?q=" + encodeURIComponent(q) + "&k=6").then(function(r){
   if (!r.ok) {
     return r.json().then(function(d){ throw new Error(d.detail || d.message || r.statusText); }).catch(function(){ throw new Error(r.statusText); });
   }
   return r.json();
 }).then(function(data){
    titleEl.textContent = (data && data.length) ? ("Top matches for \u201c" + q + "\u201d") : ("No results for \u201c" + q + "\u201d");
    subEl.textContent = (data && data.length) ? (data.length + " results") : "Try a different query.";
    if (!Array.isArray(data) || !data.length) { emptyEl.style.display = "block"; return; }
    var maxScore = Math.max.apply(null, data.map(function(x){ return x.score != null ? x.score : 0; }));
    if (maxScore <= 0) maxScore = 1;
    data.forEach(function(item){
      var title = (item.title || "").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
      var raw = item.score != null ? item.score : 0;
      var score = maxScore > 0 ? raw / maxScore : 0;
      var pct = Math.round(Math.min(100, Math.max(0, score * 100)));
      var card = document.createElement("div");
      card.className = "card-result";
      card.innerHTML = "<div class=\"cat\">PRODUCT</div><div class=\"title\">" + title + "</div><div class=\"score-bar-wrap\"><div class=\"score-bar\"><div class=\"score-bar-fill\" style=\"width:" + pct + "%\"></div></div></div><div class=\"meta\">Semantic match " + pct + "%</div><button type=\"button\" class=\"btn-why\">Why this?</button>";
      card.querySelector(".btn-why").onclick = function(){ openDialog(item.title, item.score); };
      resEl.appendChild(card);
    });
  }).catch(function(e){ titleEl.textContent = "Search failed"; subEl.textContent = ""; errEl.textContent = (e && e.message ? e.message : "Server or OpenSearch unavailable."); errEl.style.display = "block"; });
}
document.querySelectorAll(".pills .pill").forEach(function(btn){
  btn.addEventListener("click", function(){ var q = btn.dataset.q || ""; document.getElementById("query").value = q; search(q); });
});
function openDialog(title, score){
  document.getElementById("dialog-title").textContent = "Why this result?";
  var t = (title || "").replace(/</g,"&lt;").replace(/>/g,"&gt;").substring(0,100);
  if ((title || "").length > 100) t += "\u2026";
  document.getElementById("dialog-body").innerHTML = (t ? "<strong style=\"color:#fff\">" + t + "</strong><br><br>" : "") + "Embedding similarity + attribute alignment + keyword overlap. Final blended score: <strong style=\"color:#6366f1\">" + (score != null ? score.toFixed(4) : "\u2014") + "</strong>.";
  document.getElementById("dialog").classList.add("open");
  document.getElementById("dialog").setAttribute("aria-hidden","false");
}
function closeDialog(){ document.getElementById("dialog").classList.remove("open"); document.getElementById("dialog").setAttribute("aria-hidden","true"); }
document.getElementById("dialog").addEventListener("click", function(e){ if (e.target === this) closeDialog(); });
</script>
</body>
</html>
"""


# =============================================================================
# opensearch
# =============================================================================

def _run_search(url: str, body: dict, timeout: int = 10) -> requests.Response:
    """POST the search body to OpenSearch, return the response"""
    return requests.post(url, json=body, timeout=timeout)


# =============================================================================
# scoring helpers (cosine, scale, tokenize, attribute, keyword, rrf)
# =============================================================================

def _cosine_sim(query_vec: np.ndarray, doc_vec: np.ndarray) -> float:
    """cosine sim is in [-1,1]; squash to [0,1] so we can use it as a score"""
    a, b = np.asarray(query_vec, dtype=float), np.asarray(doc_vec, dtype=float)
    n = np.dot(a, b)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    if d <= 1e-9:
        return 0.0
    cos = n / d
    return (float(cos) + 1.0) / 2.0


def _min_max_scale(scores: list) -> list:
    """normalize so the best is 1 and worst is 0, rest spread in between"""
    if not scores:
        return scores
    lo, hi = min(scores), max(scores)
    span = hi - lo
    if span <= 1e-9:
        return [1.0] * len(scores)
    return [(s - lo) / span for s in scores]


def _tokenize_text(text: str) -> list:
    """chop into words for matching (lowercase, alphanumeric chunks only)"""
    return re.findall(r"[a-z0-9]+", text.lower())


def _attribute_adjustment(query_lower: str, title_lower: str) -> float:
    """if they asked for men and the doc says women (or the other way), knock it down. if it matches, bump it up"""
    q_words = set(_tokenize_text(query_lower))
    t_words = set(_tokenize_text(title_lower))
    q_male = q_words & GENDER_MALE_TERMS
    q_female = q_words & GENDER_FEMALE_TERMS
    t_male = t_words & GENDER_MALE_TERMS
    t_female = t_words & GENDER_FEMALE_TERMS
    if q_male and t_female:
        return -ATTRIBUTE_MISMATCH_PENALTY
    if q_female and t_male:
        return -ATTRIBUTE_MISMATCH_PENALTY
    if q_male and t_male:
        return ATTRIBUTE_MATCH_BONUS
    if q_female and t_female:
        return ATTRIBUTE_MATCH_BONUS
    return 0.0


def _keyword_score_bm25_style(title: str, content_tokens: list, q_lower: str) -> float:
    """keyword score: more mentions = higher, substring matches get some credit, exact phrase gets a boost"""
    if not content_tokens:
        return 0.0
    title_lower = title.lower()
    title_words = set(_tokenize_text(title))
    score = 0.0
    for t in content_tokens:
        tf = title_lower.count(t)
        if tf > 0:
            score += math.log1p(tf)
        else:
            for w in title_words:
                if t in w or w in t:
                    score += 0.5
                    break
    raw = score / len(content_tokens)
    if q_lower in title_lower:
        raw += 0.5
    return max(0.0, raw)


def _rrf_fuse(vector_scores: list, keyword_scores: list) -> list:
    """combine the two rank lists with RRF so we don't have to worry about one score dominating the other"""
    n = len(vector_scores)
    rank_v = _scores_to_rank(vector_scores, descending=True)
    rank_k = _scores_to_rank(keyword_scores, descending=True)
    return [1.0 / (RRF_K + rank_v[i]) + 1.0 / (RRF_K + rank_k[i]) for i in range(n)]


def _scores_to_rank(scores: list, descending: bool = True) -> list:
    """turn scores into ranks (1 = best, 2 = second, ...)"""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=descending)
    rank = [0] * len(scores)
    r = 1
    for i in order:
        rank[i] = r
        r += 1
    return rank


# --- re-rank pipeline: take knn hits, score with semantic + keyword + gender, fuse with RRF, return top k ---

def _hits_to_results(hits: list, q: str, k: int, query_embedding: list = None) -> list:
    """re-rank: semantic (cosine) + keyword (tf + substring + phrase + gender), then fuse with RRF and return top k"""
    query_tokens = q.lower().split()
    content_tokens = [t for t in query_tokens if t not in STOPWORDS]
    if not content_tokens:
        content_tokens = query_tokens
    q_lower = q.lower()

    has_embeddings = (
        query_embedding is not None
        and hits
        and isinstance(hits[0].get("_source", {}).get("embedding"), (list, np.ndarray))
    )
    query_vec = np.array(query_embedding, dtype=float) if query_embedding else None

    raw_vector_scores = []
    for h in hits:
        if has_embeddings and query_vec is not None:
            doc_emb = h["_source"].get("embedding")
            if doc_emb is not None:
                raw_vector_scores.append(_cosine_sim(query_vec, doc_emb))
            else:
                raw_vector_scores.append(0.0)
        else:
            raw_vector_scores.append(float(h.get("_score") or 0))

    vector_scores = _min_max_scale(raw_vector_scores)
    keyword_scores = []
    for h in hits:
        title = h["_source"].get("title", "")
        title_lower = title.lower()
        base = _keyword_score_bm25_style(title, content_tokens, q_lower)
        attr = _attribute_adjustment(q_lower, title_lower)
        keyword_scores.append(max(0.0, base + attr))
    keyword_scores = _min_max_scale(keyword_scores)

    rrf_scores = _rrf_fuse(vector_scores, keyword_scores)
    order = sorted(range(len(rrf_scores)), key=lambda i: rrf_scores[i], reverse=True)
    ordered_rrf = [rrf_scores[i] for i in order[:k]]
    # scale so the top hit shows as 100% in the UI
    display_scores = _min_max_scale(ordered_rrf) if ordered_rrf else []

    results = []
    for j, i in enumerate(order[:k]):
        title = hits[i]["_source"].get("title", "")
        results.append({"title": title, "score": display_scores[j] if j < len(display_scores) else rrf_scores[i]})

    return results


# =============================================================================
# api: search
# =============================================================================

def _opensearch_error_detail(status_code: int, text: str) -> str:
    """turn OpenSearch errors into a short message so the UI shows something useful (e.g. on GKE)"""
    if status_code == 404 or (text and "index_not_found" in text.lower()):
        return (
            "Index '%s' not found. Run the ingest script to load data: "
            "port-forward OpenSearch (kubectl port-forward svc/opensearch 9200:9200), "
            "then set OPENSEARCH_URL=http://localhost:9200 and run python ingest/ingest_fashion.py. See DEPLOY-GKE.md."
        ) % INDEX_NAME
    if status_code >= 500:
        return "OpenSearch is unavailable or overloaded. Try again in a moment."
    return (text[:300] + "…") if text and len(text) > 300 else (text or "Unknown error")


@app.get("/search")
def search(q: str, k: int = 5):
    """vector search first (knn), then re-rank with semantic + keyword + gender; if knn fails we fall back to plain match on title"""
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if k < 1 or k > 50:
        raise HTTPException(status_code=400, detail="k must be between 1 and 50")

    q = q.strip()
    logger.info("Search query received: %s", q)
    url = f"{OPENSEARCH_URL}/{INDEX_NAME}/_search"

    try:
        # try vector search first (we embed the query with the same prefix as the docs)
        try:
            embedding = embed_cached(_query_for_embedding(q))
            knn_body = {
                "size": 50,
                "knn": {
                    "field": "embedding",
                    "query_vector": embedding,
                    "k": 50,
                    "num_candidates": 50
                }
            }
            r = _run_search(url, knn_body)
            if r.status_code < 400:
                hits = r.json().get("hits", {}).get("hits", [])
                return _hits_to_results(hits, q, k, query_embedding=embedding)

            # knn failed, fall back to plain text match on title
            logger.warning("KNN search failed (%s), trying match fallback: %s", r.status_code, r.text[:200])
        except requests.RequestException as e:
            logger.warning("OpenSearch unreachable: %s", str(e))
            raise HTTPException(
                status_code=503,
                detail="OpenSearch is not running or not reachable. Check that the OpenSearch pod is up and OPENSEARCH_URL is correct. (%s)" % str(e),
            )

        # fallback: just match on title (no vectors)
        try:
            match_body = {
                "size": 50,
                "query": {"match": {"title": {"query": q, "fuzziness": "AUTO"}}},
            }
            r2 = _run_search(url, match_body)
            if r2.status_code >= 400:
                detail = _opensearch_error_detail(r2.status_code, r2.text or "")
                logger.error("Fallback match failed: %s", detail)
                raise HTTPException(status_code=503, detail=detail)
            hits = r2.json().get("hits", {}).get("hits", [])
            embedding = embed_cached(_query_for_embedding(q))
            return _hits_to_results(hits, q, k, query_embedding=embedding)
        except requests.RequestException as e:
            logger.warning("OpenSearch unreachable: %s", str(e))
            raise HTTPException(
                status_code=503,
                detail="OpenSearch is not running or not reachable. (%s)" % str(e),
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Search failed with unexpected error")
        raise HTTPException(
            status_code=500,
            detail="Search failed: %s. Check server logs." % str(e),
        ) 