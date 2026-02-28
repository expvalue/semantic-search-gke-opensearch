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
    assert "Semantic Fashion Search" in r.text


def test_search_lexical_fallback(monkeypatch):
    # Force model failure so vector side is unavailable.
    monkeypatch.setattr(svc, "embed_cached", lambda q: (_ for _ in ()).throw(RuntimeError("model unavailable")))

    def fake_post(url, json, timeout):
        assert timeout == 10
        if "bool" in json.get("query", {}):
            return DummyResponse(
                hits=[
                    {"_id": "1", "_score": 4.0, "_source": {"title": "Black Leather Jacket for Women"}},
                    {"_id": "2", "_score": 1.0, "_source": {"title": "Blue Summer Dress"}},
                ]
            )
        return DummyResponse(status_code=500, text="vector unavailable")

    monkeypatch.setattr(svc.requests, "post", fake_post)

    client = TestClient(svc.app)
    r = client.get("/search", params={"q": "black leather jacket", "k": 2})
    assert r.status_code == 200
    payload = r.json()
    assert len(payload) == 2
    assert payload[0]["title"].lower().startswith("black leather")
    assert "explain" in payload[0]


def test_search_requires_query():
    client = TestClient(svc.app)
    r = client.get("/search", params={"q": "   "})
    assert r.status_code == 400
