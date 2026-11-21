"""Shared fixtures: a tiny hand-made user history (no real data needed)."""
import pytest


def _likes(*pairs):
    return {"liked": [{"asin": a, "title": t, "rating": 5.0} for a, t in pairs],
            "disliked": [], "count": len(pairs)}


@pytest.fixture
def history():
    """Fantasy readers like fantasy films; crime readers like crime films."""
    lotr_b, hp_b, gone_b = ("B1", "The Hobbit"), ("B2", "Harry Potter"), ("B3", "Gone Girl")
    lotr_m, hp_m, gone_m = ("M1", "The Lord of the Rings"), ("M2", "Harry Potter Film"), ("M3", "Se7en")
    blockbuster = ("M9", "Generic Blockbuster")
    return {
        "u1": {"Books": _likes(lotr_b, hp_b), "Movies_and_TV": _likes(lotr_m, hp_m)},
        "u2": {"Books": _likes(lotr_b), "Movies_and_TV": _likes(lotr_m, blockbuster)},
        "u3": {"Books": _likes(hp_b), "Movies_and_TV": _likes(hp_m, blockbuster)},
        "u4": {"Books": _likes(gone_b), "Movies_and_TV": _likes(gone_m, blockbuster)},
        "u5": {"Books": _likes(gone_b), "Movies_and_TV": _likes(gone_m)},
        "u6": {"Books": _likes(("B4", "Cookbook")), "Movies_and_TV": _likes(blockbuster)},
        "u7": {"Books": _likes(lotr_b, ("B5", "nan")), "Movies_and_TV": _likes(lotr_m, ("M5", "Tom &amp; Jerry"))},
    }
