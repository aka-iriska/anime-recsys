import pytest
from fastapi.testclient import TestClient

from serving.artifacts import ALS_ARTIFACTS, RANKER_ARTIFACTS, RERANK_ARTIFACTS

# тесты API работают на настоящих артефактах: без export_models их пропускаем
ARTIFACTS = [ALS_ARTIFACTS / "item_factors.npy", RANKER_ARTIFACTS / "ranker.cbm", RERANK_ARTIFACTS / "settings.json"]
pytestmark = pytest.mark.skipif(not all(p.exists() for p in ARTIFACTS), reason="нет models/: uv run python -m training.pipelines.export_models")

HISTORY = [1, 205, 30, 47, 43]  # Cowboy Bebop, Samurai Champloo, Evangelion, Akira, Ghost in the Shell


@pytest.fixture(scope="module")
def client():
    """Клиент API с загруженными моделями (lifespan выполняется внутри with)."""
    from serving.api import app

    with TestClient(app) as test_client:
        yield test_client


def test_feed_is_personal_and_follows_rules(client):
    response = client.post("/feed", json={"history": HISTORY})
    assert response.status_code == 200
    body = response.json()
    assert body["strategy"] == "personal"
    assert body["history_used"] == len(HISTORY)
    ids = [item["anime"]["anime_id"] for item in body["items"]]
    # длина по настройкам правил, без повторов и без уже просмотренного
    assert len(ids) == 10
    assert len(set(ids)) == len(ids)
    assert not set(ids) & set(HISTORY)
    # у каждой позиции есть причина
    assert all(item["reason"] for item in body["items"])


def test_feed_respects_exclude_and_k(client):
    first = client.post("/feed", json={"history": HISTORY, "k": 5}).json()
    shown = [item["anime"]["anime_id"] for item in first["items"]]
    assert len(shown) == 5
    second = client.post("/feed", json={"history": HISTORY, "exclude": shown, "k": 5}).json()
    again = [item["anime"]["anime_id"] for item in second["items"]]
    assert not set(again) & set(shown)


def test_empty_history_gives_popular_feed(client):
    body = client.post("/feed", json={"history": []}).json()
    assert body["strategy"] == "popular"
    assert len(body["items"]) == 10
    assert body["items"][0]["reason_anime_id"] is None


def test_sequel_reason(client):
    # досмотрен Code Geass (1575): продолжение R2 (2904) – с причиной «Продолжение …»
    body = client.post("/feed", json={"history": [1575], "k": 20}).json()
    by_id = {item["anime"]["anime_id"]: item for item in body["items"]}
    if 2904 in by_id:
        assert by_id[2904]["reason"].startswith("Продолжение")
        assert by_id[2904]["reason_anime_id"] == 1575


def test_anime_card_and_404(client):
    card = client.get("/anime/1").json()
    assert card["name"] == "Cowboy Bebop"
    assert card["url"].endswith("/animes/1")
    assert client.get("/anime/999999999").status_code == 404


def test_search_prefers_title_start(client):
    results = client.get("/anime/search", params={"q": "евангелион"}).json()
    assert results
    assert results[0]["russian"].lower().startswith("евангелион")
    assert client.get("/anime/search", params={"q": ""}).status_code == 422


def test_info_has_manifests(client):
    body = client.get("/info").json()
    assert set(body["artifacts"]) == {"als_v0", "ranker_v0", "rerank_v0"}
    assert body["rerank"]["feed_length"] == 10
