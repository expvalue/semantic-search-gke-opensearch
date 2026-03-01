"""minimal tests for the search API (homepage, search returns shape, empty query rejected)"""

from fastapi.testclient import TestClient

import search_service as svc


class DummyResponse:
    def __init__(self, status_code=200, hits=None, text=""):
        self.status_code = status_code
        self._hits = hits or []
        self.text = text

    def json(self):
        return {"hits": {"hits": self._hits}}


def test_homepage_renders():
    client = TestClient(svc.app)
    r = client.get("/")
    assert r.status_code == 200
    assert "GKE Search" in r.text or "Semantic" in r.text


def test_search_returns_list_with_title_and_score(monkeypatch):
    # mock OpenSearch: knn returns two hits with title + embedding so re-rank can run
    dummy_embedding = [0.1] * 384

    def fake_post(url, json, timeout):
        return DummyResponse(
            hits=[
                {"_source": {"title": "Black Leather Jacket for Men", "embedding": dummy_embedding}},
                {"_source": {"title": "Blue Summer Dress", "embedding": dummy_embedding}},
            ]
        )

    monkeypatch.setattr(svc, "embed_cached", lambda q: dummy_embedding)
    monkeypatch.setattr(svc, "_run_search", lambda url, body, timeout=10: fake_post(url, body, timeout))

    client = TestClient(svc.app)
    r = client.get("/search", params={"q": "black leather jacket", "k": 2})
    assert r.status_code == 200
    payload = r.json()
    assert isinstance(payload, list)
    assert len(payload) == 2
    assert "title" in payload[0] and "score" in payload[0]
    assert "black" in payload[0]["title"].lower() or "leather" in payload[0]["title"].lower()


def test_search_requires_query():
    client = TestClient(svc.app)
    r = client.get("/search", params={"q": "   "})
    assert r.status_code == 400
