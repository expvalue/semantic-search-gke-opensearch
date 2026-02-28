import logging
import os
import re
from functools import lru_cache

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from sentence_transformers import SentenceTransformer

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "http://opensearch-service:9200")
INDEX_NAME = os.getenv("INDEX_NAME", "products")

VECTOR_WEIGHT = 0.45
KEYWORD_WEIGHT = 0.55
PHRASE_WEIGHT = 0.20
FUZZY_WEIGHT = 0.05

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


def get_model():
    global model
    if model is None:
        logger.info("Loading embedding model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Model loaded successfully.")
    return model


@app.get("/health")
def health():
    return {"ok": True}


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
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet"/>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{
  font-family:'Inter',system-ui,-apple-system,sans-serif;
  background:radial-gradient(circle at 50% -20%, #ecebff 0%, #f6f6fb 42%, #f3f4f8 100%);
  color:#131728;
  min-height:100vh;
}
.container{max-width:1060px;margin:0 auto;padding:1.8rem 1.2rem 3rem}
.topbar{display:flex;justify-content:space-between;align-items:center;margin-bottom:1.2rem}
.brand{font-size:.78rem;letter-spacing:.42em;color:#5847cf;font-weight:700}
.live-pill{padding:.45rem .8rem;border-radius:999px;border:1px solid #e4e6f2;background:#fff;color:#4b5568;font-size:.78rem;text-decoration:none}
.panel{background:#fff;border:1px solid #eceef5;border-radius:28px;padding:2rem;box-shadow:0 12px 28px rgba(33,38,66,.06)}
.hero{display:flex;justify-content:space-between;gap:1rem;align-items:flex-start;flex-wrap:wrap}
.hero h1{font-size:2.1rem;line-height:1.15;font-weight:800;letter-spacing:-.025em;margin-bottom:.65rem}
.hero p{color:#65708a;max-width:600px;line-height:1.55}
.signal{min-width:280px;background:#f8f8fd;border:1px solid #eef0f8;border-radius:16px;padding:1rem}
.signal .t{font-size:.75rem;color:#78839d;font-weight:700;margin-bottom:.45rem;text-transform:uppercase;letter-spacing:.06em}
.signal .d{font-size:.95rem;color:#2b3148;line-height:1.45}
.signal .n{margin-top:.45rem;font-size:.75rem;color:#8d96ad}
.search-row{display:flex;gap:.8rem;align-items:center;margin-top:1.5rem}
.search-wrap{position:relative;flex:1}
.search-wrap svg{position:absolute;left:1rem;top:50%;transform:translateY(-50%);color:#9aa4bc;width:1rem;height:1rem}
.search-wrap input{height:3.2rem;width:100%;border:1px solid #dde2ef;border-radius:999px;padding:0 1rem 0 2.8rem;font-size:.98rem;outline:none;background:#fff}
.search-wrap input:focus{border-color:#7260ff;box-shadow:0 0 0 4px rgba(114,96,255,.15)}
.search-btn{height:3.2rem;border:none;border-radius:999px;padding:0 1.5rem;background:linear-gradient(135deg,#6f59ff,#4f46e5);color:#fff;font-weight:700;font-size:.95rem;cursor:pointer;box-shadow:0 8px 22px rgba(86,73,230,.34)}
.search-btn:hover{transform:translateY(-1px)}
.pills{display:flex;flex-wrap:wrap;gap:.55rem;margin-top:1.2rem}
.pill{border:1px solid #e2e6f2;background:#fff;border-radius:999px;padding:.45rem .85rem;font-size:.82rem;color:#4f5872;cursor:pointer}
.pill.active{background:#4f46e5;color:#fff;border-color:#4f46e5}
.results{margin-top:2rem}
.results-head{display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;flex-wrap:wrap}
.results-head h2{font-size:1.85rem;line-height:1.15;letter-spacing:-.02em}
.results-head p{color:#6d7690;margin-top:.3rem}
.mode{border:1px solid #e2e5f3;background:#fff;padding:.45rem .85rem;border-radius:999px;font-size:.78rem;color:#5f6780}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1rem;margin-top:1.2rem}
@media (max-width:980px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:690px){.grid{grid-template-columns:1fr}.panel{padding:1.25rem}.hero h1{font-size:1.6rem}.search-row{flex-direction:column;align-items:stretch}.search-btn{width:100%}}
.card{background:#fff;border:1px solid #eceff6;border-radius:18px;padding:1rem;box-shadow:0 4px 10px rgba(18,23,39,.04);animation:up .2s ease-out forwards;opacity:0}
.card:nth-child(1){animation-delay:0ms}.card:nth-child(2){animation-delay:30ms}.card:nth-child(3){animation-delay:60ms}.card:nth-child(4){animation-delay:90ms}
.card .k{font-size:.72rem;color:#97a0b7;font-weight:700;letter-spacing:.05em;margin-bottom:.35rem}
.card .tt{font-size:1.02rem;font-weight:700;line-height:1.35;min-height:2.7rem}
.card .meta{margin-top:.7rem;display:flex;justify-content:space-between;color:#66708b;font-size:.82rem}
.score{background:#eef0ff;color:#4f46e5;border-radius:999px;padding:.2rem .55rem;font-weight:700}
.why{margin-top:.9rem;width:100%;height:2.25rem;border-radius:999px;border:1px solid #dfe4f0;background:#fff;color:#3a4258;font-weight:600;cursor:pointer}
.empty{display:none;margin-top:1rem;color:#6f7892}
.loading{pointer-events:none}
.skeleton{display:block;border-radius:10px;background:linear-gradient(90deg,#eef1f8 0%,#e4e8f2 45%,#eef1f8 100%);background-size:180% 100%;animation:shine 1.2s linear infinite}
.skeleton.a{height:.76rem;width:40%;margin-bottom:.45rem}
.skeleton.b{height:1rem;margin:.2rem 0 .6rem}
.skeleton.c{height:.72rem;width:65%;margin-bottom:.8rem}
.skeleton.d{height:2.15rem;border-radius:999px}
.dialog{position:fixed;inset:0;background:rgba(17,20,37,.42);display:none;align-items:center;justify-content:center;padding:1rem;z-index:9}
.dialog.open{display:flex}
.dialog-box{width:min(520px,100%);background:#fff;border-radius:18px;padding:1rem 1.1rem;box-shadow:0 24px 60px rgba(18,20,38,.24)}
.dialog-box h4{font-size:1rem;margin-bottom:.6rem}
.dialog-box p{color:#5f6780;line-height:1.55;font-size:.9rem}
.close{float:right;border:none;background:transparent;font-size:1.4rem;cursor:pointer;color:#78839f}
@keyframes shine{0%{background-position:180% 0}100%{background-position:-180% 0}}
@keyframes up{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
</style>
</head>
<body>
<div class="container">
  <div class="topbar">
    <div class="brand">PHIA SEARCH DEMO</div>
    <a class="live-pill" href="/">Live Demo</a>
  </div>

  <section class="panel">
    <div class="hero">
      <div>
        <h1>Find the perfect fashion pick.</h1>
        <p>Semantic retrieval over Amazon Fashion using only shipped capabilities: vector KNN retrieval + lexical rerank with transparent score breakdown.</p>
      </div>
      <div class="signal">
        <div class="t">Signal blend</div>
        <div class="d">MiniLM embeddings · OpenSearch KNN · BM25 + phrase + fuzzy lexical scoring</div>
        <div class="n">No placeholder features shown.</div>
      </div>
    </div>

    <form class="search-row" onsubmit="return runSearch(event)">
      <div class="search-wrap">
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>
        <input id="query" type="text" placeholder="Search Amazon Fashion (e.g., black leather jacket)" autofocus/>
      </div>
      <button class="search-btn" type="submit">Explore feed</button>
    </form>

    <div class="pills">
      <button class="pill active" type="button" data-q="">All</button>
      <button class="pill" type="button" data-q="jacket">Jackets</button>
      <button class="pill" type="button" data-q="dress">Dresses</button>
      <button class="pill" type="button" data-q="heels">Heels</button>
      <button class="pill" type="button" data-q="bag">Bags</button>
      <button class="pill" type="button" data-q="sneakers">Sneakers</button>
    </div>
  </section>

  <section class="results">
    <div class="results-head">
      <div>
        <h2 id="results-title">Recommended for you</h2>
        <p id="results-sub">Run a search above to retrieve and rank products.</p>
      </div>
      <span class="mode" id="mode">Hybrid mode: MiniLM + BM25</span>
    </div>
    <div class="grid" id="results"></div>
    <p class="empty" id="empty">No results yet. Try a different query.</p>
  </section>
</div>

<div class="dialog" id="dialog" aria-hidden="true">
  <div class="dialog-box">
    <button class="close" type="button" aria-label="Close" onclick="closeDialog()">&times;</button>
    <h4 id="dialog-title">Why this result?</h4>
    <p id="dialog-body">Score details will appear here.</p>
  </div>
</div>

<script>
var currentQuery = '';
function runSearch(e){e.preventDefault();search();return false;}
function setPills(){
  document.querySelectorAll('.pill').forEach(function(btn){
    btn.classList.toggle('active', (btn.dataset.q||'')===currentQuery);
  });
}
function renderLoading(container){
  container.innerHTML = '';
  for(var i=0;i<3;i++){
    var c=document.createElement('div');
    c.className='card loading';
    c.innerHTML='<span class="skeleton a"></span><span class="skeleton b"></span><span class="skeleton c"></span><span class="skeleton d"></span>';
    container.appendChild(c);
  }
}
function escapeHtml(v){
  return (v||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function search(optionalQ){
  var q = (optionalQ !== undefined ? optionalQ : document.getElementById('query').value).trim();
  currentQuery = q;
  setPills();
  var resEl = document.getElementById('results');
  var titleEl = document.getElementById('results-title');
  var subEl = document.getElementById('results-sub');
  var emptyEl = document.getElementById('empty');

  emptyEl.style.display='none';
  if(!q){
    resEl.innerHTML='';
    titleEl.textContent='Recommended for you';
    subEl.textContent='Run a search above to retrieve and rank products.';
    return;
  }

  titleEl.textContent='Searching...';
  subEl.textContent='Blending semantic + lexical signals';
  renderLoading(resEl);

  fetch('/search?q=' + encodeURIComponent(q) + '&k=6')
    .then(function(r){return r.json();})
    .then(function(data){
      if(!Array.isArray(data)){ throw new Error('Invalid search response'); }
      titleEl.textContent = data.length ? 'Top matches for "' + q + '"' : 'No results for "' + q + '"';
      subEl.textContent = data.length ? data.length + ' ranked results' : 'Try a more specific fashion term.';
      resEl.innerHTML='';
      if(!data.length){ emptyEl.style.display='block'; return; }

      data.forEach(function(item){
        var card = document.createElement('div');
        card.className='card';
        var explain = item.explain || {};
        card.innerHTML = '<div class="k">PRODUCT</div>' +
          '<div class="tt">' + escapeHtml(item.title || 'Untitled') + '</div>' +
          '<div class="meta"><span>Relevance</span><span class="score">' + (item.score != null ? item.score.toFixed(4) : '—') + '</span></div>' +
          '<button type="button" class="why" data-title="' + escapeHtml(item.title || '') + '" data-score="' + (item.score || 0) + '" data-semantic="' + (explain.semantic || 0) + '" data-lexical="' + (explain.lexical || 0) + '" data-phrase="' + (explain.phrase || 0) + '" data-fuzzy="' + (explain.fuzzy || 0) + '">Why this?</button>';
        card.querySelector('.why').onclick=function(){openDialog(this.dataset);};
        resEl.appendChild(card);
      });
    })
    .catch(function(){
      titleEl.textContent='Search unavailable';
      subEl.textContent='Could not fetch results from backend';
      resEl.innerHTML='';
      emptyEl.style.display='block';
    });
}

document.querySelectorAll('.pill').forEach(function(btn){
  btn.addEventListener('click', function(){
    var q = this.dataset.q || '';
    document.getElementById('query').value = q;
    search(q);
  });
});

function openDialog(data){
  document.getElementById('dialog-title').textContent='Why this result?';
  var summary = 'Semantic: <strong>' + data.semantic + '</strong> · Lexical: <strong>' + data.lexical + '</strong> · Phrase: <strong>' + data.phrase + '</strong> · Fuzzy: <strong>' + data.fuzzy + '</strong>';
  document.getElementById('dialog-body').innerHTML = '<strong style="color:#11182a">' + (data.title || '').slice(0,120) + '</strong><br/><br/>Ranked by working hybrid scoring only. ' + summary + '. Total: <strong>' + data.score + '</strong>.';
  document.getElementById('dialog').classList.add('open');
  document.getElementById('dialog').setAttribute('aria-hidden','false');
}
function closeDialog(){
  document.getElementById('dialog').classList.remove('open');
  document.getElementById('dialog').setAttribute('aria-hidden','true');
}
document.getElementById('dialog').addEventListener('click', function(e){ if(e.target===this) closeDialog(); });
</script>
</body>
</html>
"""


@lru_cache(maxsize=1000)
def embed_cached(q: str):
    return get_model().encode(q).tolist()


def tokenize(text: str):
    return TOKEN_RE.findall((text or "").lower())


def normalized_knn_score(raw_score: float):
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


def _post_search(body):
    response = requests.post(
        f"{OPENSEARCH_URL}/{INDEX_NAME}/_search",
        json=body,
        timeout=10,
    )
    if response.status_code >= 400:
        raise HTTPException(status_code=500, detail=response.text)
    return response.json().get("hits", {}).get("hits", [])


@app.get("/search")
def search(q: str, k: int = 5):
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if k < 1 or k > 50:
        raise HTTPException(status_code=400, detail="k must be between 1 and 50")

    logger.info("Search query received: %s", q)

    text_body = {
        "size": 50,
        "query": {
            "bool": {
                "should": [
                    {"match": {"title": {"query": q, "boost": 1.0}}},
                    {"match_phrase": {"title": {"query": q, "boost": 2.0}}},
                    {"match": {"title": {"query": q, "fuzziness": "AUTO", "boost": 0.4}}},
                ],
                "minimum_should_match": 1,
            }
        },
    }

    vector_hits = []
    text_hits = []

    # Lexical retrieval should always be attempted.
    try:
        text_hits = _post_search(text_body)
    except (requests.RequestException, HTTPException) as e:
        logger.warning("Lexical recall unavailable: %s", str(e))

    # Vector retrieval is best-effort so search still works if model/network is unavailable.
    try:
        embedding = embed_cached(q)
        vector_body = {
            "size": 50,
            "query": {"knn": {"embedding": {"vector": embedding, "k": 50}}},
        }
        vector_hits = _post_search(vector_body)
    except Exception as e:  # keep broad for model download or runtime issues
        logger.warning("Vector recall unavailable; falling back to lexical-only: %s", str(e))

    if not text_hits and not vector_hits:
        raise HTTPException(status_code=502, detail="Search backend unavailable")

    text_scores = normalize_text_score(
        {h.get("_id"): h.get("_score", 0) for h in text_hits if h.get("_id")}
    )
    vector_scores = {
        h.get("_id"): normalized_knn_score(h.get("_score", 0))
        for h in vector_hits
        if h.get("_id")
    }

    candidates = {}
    for hit in vector_hits + text_hits:
        doc_id = hit.get("_id")
        if doc_id and doc_id not in candidates:
            candidates[doc_id] = hit

    query_tokens = tokenize(q)
    content_tokens = [t for t in query_tokens if t not in STOPWORDS] or query_tokens

    results = []
    for doc_id, hit in candidates.items():
        title = hit.get("_source", {}).get("title", "")
        title_tokens = tokenize(title)
        title_token_set = set(title_tokens)

        vector_score = vector_scores.get(doc_id, 0.0)
        lexical_score = text_scores.get(doc_id, 0.0)

        content_matches = [t for t in content_tokens if t in title_token_set]
        keyword_matches = len(content_matches)
        keyword_score = keyword_matches / max(len(content_tokens), 1)
        if keyword_matches == 0 and len(content_tokens) >= 2:
            keyword_score -= 0.5

        phrase_boost = 1.0 if q.lower() in title.lower() else 0.0
        fuzzy_score = jaccard_similarity(content_tokens, title_tokens)

        semantic_component = VECTOR_WEIGHT * vector_score
        lexical_component = KEYWORD_WEIGHT * (0.65 * lexical_score + 0.35 * keyword_score)
        phrase_component = PHRASE_WEIGHT * phrase_boost
        fuzzy_component = FUZZY_WEIGHT * fuzzy_score

        final_score = semantic_component + lexical_component + phrase_component + fuzzy_component

        results.append(
            {
                "id": doc_id,
                "title": title,
                "score": final_score,
                "explain": {
                    "semantic": round(semantic_component, 4),
                    "lexical": round(lexical_component, 4),
                    "phrase": round(phrase_component, 4),
                    "fuzzy": round(fuzzy_component, 4),
                },
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:k]
