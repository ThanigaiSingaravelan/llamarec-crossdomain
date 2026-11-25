"""Tests for the generative chat assistant: message understanding and the /api/chat stream."""
import json

import pytest

from core.engine import CrossDomainEngine
from llm.chat import MentionFinder, detect_target, understand

pytest.importorskip("flask")
from webapp import create_app  # noqa: E402


@pytest.fixture
def finders(history):
    engine = CrossDomainEngine(history)
    return {d: MentionFinder(engine.titles[d], engine.popularity[d]) for d in engine.domains}


# --- understanding messages ------------------------------------------------------
@pytest.mark.parametrize("message, target", [
    ("I loved The Hobbit, what should I watch?", "Movies_and_TV"),
    ("Recommend some books for me", "Books"),
    ("Any films for a fan of these books?", "Movies_and_TV"),
    ("I read a lot, suggest something to watch", "Movies_and_TV"),
    ("hello", "Books"),                                     # falls back to the default
])
def test_detect_target(message, target):
    assert detect_target(message, default_target="Books") == target


def test_finds_titles_longest_first(finders):
    found = finders["Books"].find("I loved The Hobbit and Harry Potter!")
    assert [f["title"] for f in found] == ["The Hobbit", "Harry Potter"]


def test_single_word_titles_need_capitals():
    finder = MentionFinder({"B9": "Holes", "B8": "Dune"})
    assert finder.find("there were holes in the plot") == []
    assert [f["title"] for f in finder.find("I loved Holes and Dune")] == ["Holes", "Dune"]


def test_understand_basic_request(finders):
    req = understand("I loved The Hobbit, what should I watch?", finders, "Books", "Movies_and_TV")
    assert (req["source"], req["target"]) == ("Books", "Movies_and_TV")
    assert [s["title"] for s in req["seeds"]] == ["The Hobbit"]


def test_understand_titles_from_the_other_domain_flip_direction(finders):
    req = understand("Se7en was great, anything similar?", finders, "Books", "Movies_and_TV")
    assert (req["source"], req["target"]) == ("Movies_and_TV", "Books")
    assert [s["title"] for s in req["seeds"]] == ["Se7en"]


def test_understand_follow_ups(finders):
    first = [{"asin": "B1", "title": "The Hobbit"}]
    more = understand("More like these", finders, "Books", "Movies_and_TV", first)
    assert more["more"] and more["seeds"] == first
    added = understand("Also add Gone Girl", finders, "Books", "Movies_and_TV", first)
    assert [s["title"] for s in added["seeds"]] == ["The Hobbit", "Gone Girl"]


# --- /api/chat stream -----------------------------------------------------------
class StreamingFakeLLM:
    def stream_recommendations(self, source, target, titles, n=10, sample=False):
        yield from ["The Lord ", "of the Rings\n", "Made Up Film\n", "Se7", "en"]

    def stream_explanation(self, source, target, liked, picks):
        yield from ["You like ", "epic adventures."]


def events(response):
    return [json.loads(line[6:]) for line in response.get_data(as_text=True).splitlines()
            if line.startswith("data: ")]


@pytest.fixture
def chat_client(history):
    return create_app({"TESTING": True, "HISTORY": history, "LLM": StreamingFakeLLM()}).test_client()


def test_chat_streams_understanding_cards_and_explanation(chat_client):
    res = chat_client.post("/api/chat", json={"message": "I loved The Hobbit, what should I watch?", "k": 5})
    assert res.mimetype == "text/event-stream"
    evs = events(res)
    assert evs[0]["type"] == "understood" and evs[0]["seeds"][0]["title"] == "The Hobbit"
    assert "".join(e["text"] for e in evs if e["type"] == "token") == "The Lord of the Rings\nMade Up Film\nSe7en"
    assert [e["rec"]["title"] for e in evs if e["type"] == "rec"] == ["The Lord of the Rings", "Se7en"]
    assert [e["title"] for e in evs if e["type"] == "invented"] == ["Made Up Film"]
    assert "".join(e["text"] for e in evs if e["type"] == "explain") == "You like epic adventures."
    assert evs[-1]["type"] == "done" and evs[-1]["picked"] == ["M1", "M3"]


def test_chat_more_skips_already_recommended(chat_client):
    res = chat_client.post("/api/chat", json={"message": "More like these", "k": 5,
                                              "seeds": [{"asin": "B1", "title": "The Hobbit"}], "exclude": ["M1"]})
    assert [e["rec"]["title"] for e in events(res) if e["type"] == "rec"] == ["Se7en"]


def test_chat_without_known_titles_explains_what_to_do(chat_client):
    evs = events(chat_client.post("/api/chat", json={"message": "hello there"}))
    assert [e["type"] for e in evs] == ["understood", "error", "done"]


def test_chat_validation(chat_client, history, tmp_path):
    assert chat_client.post("/api/chat", json={}).status_code == 400
    app = create_app({"TESTING": True, "HISTORY": history, "LLM_ADAPTER_DIR": str(tmp_path / "none")})
    assert app.test_client().post("/api/chat", json={"message": "hi"}).status_code == 503
