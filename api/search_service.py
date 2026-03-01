@app.get("/search")
def search(q: str, k: int = 5):
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if k < 1 or k > 50:
        raise HTTPException(status_code=400, detail="k must be between 1 and 50")

    logger.info("Search query received: %s", q)

    vector_body = {
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

    text_body = vector_body  # same lexical structure

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
            raise HTTPException(status_code=500, detail=vector_response.text)
        if text_response.status_code >= 400:
            raise HTTPException(status_code=500, detail=text_response.text)

        vector_hits = vector_response.json().get("hits", {}).get("hits", [])
        text_hits = text_response.json().get("hits", {}).get("hits", [])

        vector_scores = {
            h["_id"]: normalized_knn_score(h.get("_score", 0))
            for h in vector_hits if h.get("_id")
        }

        text_scores = normalize_text_score({
            h["_id"]: h.get("_score", 0)
            for h in text_hits if h.get("_id")
        })

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
            title_tokens = tokenize(title)

            vector_score = vector_scores.get(doc_id, 0.0)
            lexical_score = text_scores.get(doc_id, 0.0)

            title_token_set = set(title_tokens)
            content_matches = [t for t in content_tokens if t in title_token_set]
            keyword_score = len(content_matches) / max(len(content_tokens), 1)

            phrase_boost = 1.0 if q.lower() in title.lower() else 0.0
            fuzzy_score = jaccard_similarity(content_tokens, title_tokens)

            semantic_component = VECTOR_WEIGHT * vector_score
            lexical_component = KEYWORD_WEIGHT * (0.65 * lexical_score + 0.35 * keyword_score)
            phrase_component = PHRASE_WEIGHT * phrase_boost
            fuzzy_component = FUZZY_WEIGHT * fuzzy_score

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

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    except requests.RequestException as e:
        logger.error("Request failure: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))