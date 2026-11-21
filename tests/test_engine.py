"""Tests for core.engine.CrossDomainEngine (data-based recommender)."""
import pytest

from core.engine import CrossDomainEngine


@pytest.fixture
def engine(history):
    return CrossDomainEngine(history)


def titles(result):
    return [r["title"] for r in result["recommendations"]]


def test_index_and_stats(engine):
    assert engine.domains == ["Books", "Movies_and_TV"]
    assert engine.domain_stats()["Books"]["users"] == 8
    # u8 only liked placeholder films, so has no usable Movies_and_TV likes
    assert set(engine.users("Books", "Movies_and_TV")) == {f"u{i}" for i in range(1, 8)}


def test_placeholder_products_are_ignored(engine):
    assert "MX" not in engine.titles["Movies_and_TV"]
    assert "MY" not in engine.titles["Movies_and_TV"]
    recs = engine.recommend("Books", "Movies_and_TV", ["B1"], k=10)["recommendations"]
    assert not {"Test Big Data 1737", "Unknown"} & {r["title"] for r in recs}


def test_titles_are_cleaned(engine):
    assert engine.title("Movies_and_TV", "M5") == "Tom & Jerry"   # HTML entity unescaped
    assert engine.title("Books", "B5") == "B5"                     # "nan" title -> asin


def test_fantasy_reader_gets_fantasy_films_first(engine):
    result = engine.recommend("Books", "Movies_and_TV", ["B1"], k=2)
    assert titles(result)[0] == "The Lord of the Rings"
    assert result["recommendations"][0]["because"]["title"] == "The Hobbit"
    assert "The Hobbit" in result["recommendations"][0]["reason"]


def test_crime_reader_gets_crime_film(engine):
    assert titles(engine.recommend("Books", "Movies_and_TV", ["B3"], k=1)) == ["Se7en"]


def test_reverse_direction(engine):
    result = engine.recommend("Movies_and_TV", "Books", ["M3"], k=1)
    assert titles(result) == ["Gone Girl"]


def test_exclude_items_and_user(engine):
    result = engine.recommend("Books", "Movies_and_TV", ["B3"], k=5,
                              exclude_user="u4", exclude_items=["M3"])
    assert "Se7en" not in titles(result)


def test_unknown_seed_falls_back_to_popularity(engine):
    result = engine.recommend("Books", "Movies_and_TV", ["does-not-exist"], k=1)
    assert result["strategy"].startswith("popularity")
    assert titles(result) == ["Generic Blockbuster"]


def test_tops_up_with_popular_items_when_sparse(engine):
    result = engine.recommend("Books", "Movies_and_TV", ["B3"], k=4)
    assert len(result["recommendations"]) == 4
    assert "fallback" in result["strategy"]


def test_recommend_for_user_hides_own_data(engine):
    result = engine.recommend_for_user("u5", "Books", "Movies_and_TV", k=3)
    # u5 already liked Se7en, so it is not recommended again
    assert "Se7en" not in titles(result)
    assert result["already_liked_in_target"] == [{"asin": "M3", "title": "Se7en"}]


def test_include_known_flags_real_likes(engine):
    result = engine.recommend_for_user("u5", "Books", "Movies_and_TV", k=1, include_known=True)
    # u4 (not u5 itself) also liked Gone Girl + Se7en, so it is predicted and flagged
    assert result["recommendations"][0]["title"] == "Se7en"
    assert result["recommendations"][0]["liked_by_user"] is True


def test_recommend_for_user_errors(engine):
    with pytest.raises(KeyError):
        engine.recommend_for_user("nobody", "Books", "Movies_and_TV")
    with pytest.raises(ValueError):
        engine.recommend("Books", "Music", ["B1"])


def test_search(engine):
    hits = engine.search("Books", "harry")
    assert hits == [{"asin": "B2", "title": "Harry Potter", "likes": 2}]
    assert engine.search("Books", "") == []


def test_evaluate_beats_or_matches_popularity(engine):
    res = engine.evaluate("Books", "Movies_and_TV", k=1, n_users=50)
    assert res["users_evaluated"] == 7
    co, pop = res["methods"]["co-occurrence"], res["methods"]["popularity"]
    for metric in ("hit_rate", "precision", "recall", "ndcg", "diversity"):
        assert 0.0 <= co[metric] <= 1.0
    assert co["hit_rate"] >= pop["hit_rate"]
