"""Tests for the Flask web app and JSON API."""
import pytest

pytest.importorskip("flask")

from webapp import create_app  # noqa: E402


@pytest.fixture
def no_llm(tmp_path):
    """Never pick up a locally trained model: tests must behave the same on every machine."""
    return str(tmp_path / "no-model")


@pytest.fixture
def client(history, no_llm):
    app = create_app({"TESTING": True, "HISTORY": history, "LLM_ADAPTER_DIR": no_llm})
    return app.test_client()


@pytest.fixture
def empty_client(no_llm):
    return create_app({"TESTING": True, "HISTORY": {}, "LLM_ADAPTER_DIR": no_llm}).test_client()


def test_index_page(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"LLAMAREC" in res.data and b"app.js" in res.data


def test_static_assets(client):
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/app.css").status_code == 200


def test_health(client):
    data = client.get("/api/health").get_json()
    assert data == {"status": "ok", "users": 8, "domains": ["Books", "Movies_and_TV"],
                    "engines": ["co-occurrence"]}


def test_health_without_data(empty_client):
    res = empty_client.get("/api/health")
    assert res.status_code == 503
    assert res.get_json()["status"] == "no-data"


def test_domains(client):
    data = client.get("/api/domains").get_json()
    assert data["Books"]["users"] == 8


def test_users_paging_and_random(client):
    data = client.get("/api/users?source=Books&target=Movies_and_TV&limit=2&offset=1").get_json()
    assert data["total"] == 7 and len(data["users"]) == 2
    rnd = client.get("/api/users?limit=3&random=1").get_json()
    assert len(rnd["users"]) == 3


def test_user_profile(client):
    assert client.get("/api/users/u1").get_json()["domain_count"] == 2
    assert client.get("/api/users/nobody").status_code == 404


def test_search(client):
    data = client.get("/api/items/search?domain=Books&q=hobbit").get_json()
    assert data["items"][0]["title"] == "The Hobbit"


def test_recommend_from_picked_items(client):
    res = client.post("/api/recommend", json={
        "source_domain": "Books", "target_domain": "Movies_and_TV", "k": 1, "liked_asins": ["B1"]})
    assert res.status_code == 200
    assert res.get_json()["recommendations"][0]["title"] == "The Lord of the Rings"


def test_recommend_for_user_and_random_user(client):
    data = client.post("/api/recommend", json={"user_id": "u5", "k": 2}).get_json()
    assert data["user_id"] == "u5" and len(data["recommendations"]) == 2
    data = client.post("/api/recommend", json={"k": 2}).get_json()
    assert data["user_id"] in {f"u{i}" for i in range(1, 8)}


def test_recommend_include_known(client):
    data = client.post("/api/recommend", json={"user_id": "u5", "k": 1, "include_known": True}).get_json()
    assert data["recommendations"][0]["liked_by_user"] is True


@pytest.mark.parametrize("body, status", [
    ({"source_domain": "Books", "target_domain": "Books"}, 400),
    ({"source_domain": "Music", "target_domain": "Books"}, 400),
    ({"k": "many"}, 400),
    ({"liked_asins": ["unknown"]}, 400),
    ({"user_id": "nobody"}, 404),
])
def test_recommend_errors(client, body, status):
    res = client.post("/api/recommend", json=body)
    assert res.status_code == status
    assert "error" in res.get_json()


def test_evaluate(client):
    data = client.get("/api/evaluate?source=Books&target=Movies_and_TV&k=1&n=10").get_json()
    assert set(data["methods"]) == {"co-occurrence", "popularity"}
    assert client.get("/api/evaluate?source=Books&target=Nothing").status_code == 400


# --- optional fine-tuned LLM engine ------------------------------------------
class FakeLLM:
    """Stands in for llm.recommend.LLMRecommender: always 'generates' the same text."""
    text = "The Lord of the Rings\nMade Up Film\nSe7en"

    def recommend(self, engine, source, target, source_titles, k=10, exclude_asins=(), matcher=None):
        from llm.recommend import CatalogMatcher, ground
        matcher = CatalogMatcher(engine.titles[target], engine.popularity[target])
        return ground(self.text, matcher, engine, source, target, source_titles, k, exclude_asins)


@pytest.fixture
def llm_client(history, tmp_path):
    app = create_app({"TESTING": True, "HISTORY": history, "LLM": FakeLLM()})
    return app.test_client()


def test_llm_unavailable_returns_503(history, tmp_path):
    app = create_app({"TESTING": True, "HISTORY": history, "LLM_ADAPTER_DIR": str(tmp_path / "missing")})
    res = app.test_client().post("/api/recommend", json={"engine": "llm", "user_id": "u1"})
    assert res.status_code == 503 and "llm-train" in res.get_json()["error"]


def test_health_lists_llm_when_available(llm_client):
    assert llm_client.get("/api/health").get_json()["engines"] == ["co-occurrence", "llm"]


def test_llm_recommend_from_picked_items(llm_client):
    data = llm_client.post("/api/recommend", json={"engine": "llm", "k": 5, "liked_asins": ["B1"]}).get_json()
    assert data["strategy"] == "fine-tuned LLM"
    assert [r["title"] for r in data["recommendations"]] == ["The Lord of the Rings", "Se7en"]
    assert data["invented"] == ["Made Up Film"]
    assert data["seeds"] == [{"asin": "B1", "title": "The Hobbit"}]


def test_llm_recommend_for_user_hides_and_flags_known(llm_client):
    data = llm_client.post("/api/recommend", json={"engine": "llm", "user_id": "u5", "k": 5}).get_json()
    assert "Se7en" not in [r["title"] for r in data["recommendations"]]       # u5 already liked it
    data = llm_client.post("/api/recommend", json={"engine": "llm", "user_id": "u5", "k": 5,
                                                   "include_known": True}).get_json()
    assert {r["title"]: r["liked_by_user"] for r in data["recommendations"]}["Se7en"] is True
