"""Tests for the LLM data pipeline and grounding (no GPU, model or download needed)."""
import pytest

from core.engine import CrossDomainEngine
from llm.data import (MAX_SOURCE_ITEMS, MAX_TARGET_ITEMS, build_examples, example_messages, is_test_user,
                      liked_items, train_users)
from llm.recommend import CatalogMatcher, ground, normalise, parse_titles


def test_user_split_is_stable_and_roughly_80_20():
    ids = [f"user{i}" for i in range(5000)]
    first = [is_test_user(u) for u in ids]
    assert first == [is_test_user(u) for u in ids]          # deterministic
    assert 0.17 < sum(first) / len(ids) < 0.23


def test_liked_items_cleans_and_skips_placeholders():
    data = {"liked": [{"asin": "A", "title": "Tom &amp; Jerry"}, {"asin": "B", "title": "Test Big Data 1737"},
                      {"asin": "C", "title": "nan"}, {"asin": "D", "title": "tom & jerry"}]}
    assert liked_items(data) == [{"title": "Tom & Jerry", "asin": "A"}]


def test_examples_have_no_user_overlap_and_both_directions(history):
    train, test = build_examples(history, "train"), build_examples(history, "test")
    assert not {e["user_id"] for e in train} & {e["user_id"] for e in test}
    assert {(e["source"], e["target"]) for e in train + test} <= {("Books", "Movies_and_TV"),
                                                                  ("Movies_and_TV", "Books")}
    baseline_users = set(train_users(history))
    assert {e["user_id"] for e in train} <= baseline_users
    assert not {e["user_id"] for e in test} & baseline_users     # baselines never see test users


def test_example_messages_format():
    ex = {"source": "Books", "target": "Movies_and_TV",
          "source_items": [{"title": f"Book {i}", "asin": str(i)} for i in range(30)],
          "target_items": [{"title": f"Film {i}", "asin": f"m{i}"} for i in range(30)]}
    msgs = example_messages(ex)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[1]["content"].count("\n- ") == MAX_SOURCE_ITEMS
    assert msgs[2]["content"].splitlines() == [f"Film {i}" for i in range(MAX_TARGET_ITEMS)]
    assert len(example_messages(ex, with_answer=False)) == 2


def test_normalise_ignores_formats_and_punctuation():
    assert normalise("Holes VHS") == normalise("Holes (DVD)") == "holes"
    assert normalise("Up (Mandarin Chinese Edition)") == "up"
    assert normalise("Tom &amp; Jerry!") == "tom jerry"


def test_parse_titles():
    text = "1. The Matrix\n- Inception\n* \"Arrival\"\n\n2) The Matrix\nX"
    assert parse_titles(text) == ["The Matrix", "Inception", "Arrival"]


def test_catalog_matcher_exact_and_fuzzy():
    m = CatalogMatcher({"M1": "The Lord of the Rings VHS", "M2": "Se7en"}, {"M1": 5, "M2": 1})
    assert m.match("The Lord of the Rings") == "M1"
    assert m.match("A Completely Different Film") is None
    pytest.importorskip("rapidfuzz")
    assert m.match("Lord of the Rings, The") == "M1"


def test_ground_maps_skips_invented_and_excluded(history):
    engine = CrossDomainEngine(history)
    matcher = CatalogMatcher(engine.titles["Movies_and_TV"], engine.popularity["Movies_and_TV"])
    text = "The Lord of the Rings\nTotally Made Up Film\nSe7en\nHarry Potter Film"
    result = ground(text, matcher, engine, "Books", "Movies_and_TV", ["The Hobbit"], k=5, exclude_asins=["M3"])
    assert [r["title"] for r in result["recommendations"]] == ["The Lord of the Rings", "Harry Potter Film"]
    assert result["invented"] == ["Totally Made Up Film"]
    assert result["strategy"] == "fine-tuned LLM"
