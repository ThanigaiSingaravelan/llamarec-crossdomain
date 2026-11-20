"""Fast unit tests for LLAMAREC core pieces (no Ollama or dataset required)."""
import json

import pytest

from utils.metrics import jaccard_diversity, ndcg_at_k, precision_at_k, recall_at_k
from prompts import prompt_templates


# --- utils.metrics -----------------------------------------------------------
def test_precision_at_k_counts_case_insensitive_hits():
    preds = ["The Matrix", "Inception", "Up"]
    actual = ["the matrix", "up"]
    assert precision_at_k(preds, actual, k=3) == pytest.approx(2 / 3)


def test_recall_at_k_handles_empty_ground_truth():
    assert recall_at_k(["a"], [], k=3) == 0.0


def test_ndcg_is_one_for_perfect_ranking():
    assert ndcg_at_k(["a", "b"], ["a", "b"], k=2) == pytest.approx(1.0)


def test_ndcg_rewards_earlier_hits():
    early = ndcg_at_k(["a", "x", "y"], ["a"], k=3)
    late = ndcg_at_k(["x", "y", "a"], ["a"], k=3)
    assert early > late > 0


def test_jaccard_diversity_bounds():
    assert jaccard_diversity(["red apple", "red apple"]) == 0.0
    assert jaccard_diversity(["red apple", "blue sky"]) == 1.0
    assert jaccard_diversity(["only one"]) == 0.0


# --- prompts.prompt_templates -----------------------------------------------
def test_basic_template_formats_all_placeholders():
    text = prompt_templates.BASIC_TEMPLATE.format(
        source_domain="Books",
        target_domain="Movies_and_TV",
        user_history="Hatchet, Holes",
        num_recs=3,
    )
    assert "Books" in text and "Movies_and_TV" in text and "Hatchet" in text
    assert "{" not in text


# --- evaluations.quantitative_evaluator -------------------------------------
@pytest.fixture
def evaluator(tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")
    from evaluations.quantitative_evaluator import RecommendationEvaluator

    history = {
        "u1": {
            "Books": {"liked": [{"asin": "B1", "title": "Holes", "rating": 5.0}],
                      "disliked": [], "count": 1},
            "Movies_and_TV": {"liked": [{"asin": "M1", "title": "The Matrix", "rating": 5.0}],
                              "disliked": [], "count": 1},
        }
    }
    path = tmp_path / "user_history.json"
    path.write_text(json.dumps(history), encoding="utf-8")
    return RecommendationEvaluator(str(path))


def test_fuzzy_match(evaluator):
    assert evaluator.fuzzy_match("the matrix", "The Matrix")
    assert not evaluator.fuzzy_match("the matrix", "toy story")


def test_extract_recommended_items_from_numbered_list(evaluator):
    text = (
        "1. **The Matrix** - mind-bending sci-fi\n"
        "2. **Inception** - layered plot\n"
        "3. **Arrival** - thoughtful first contact"
    )
    items = evaluator.extract_recommended_items(text, k=3)
    assert len(items) == 3
    assert any("matrix" in i.lower() for i in items)
