"""Understand a chat message: which titles does the person mention, and what do they want?

    "I loved The Hobbit and Holes, what should I watch?"
        -> source Books, target Movies_and_TV, seeds [The Hobbit, Holes]

Titles are found by matching word sequences in the message against the real catalogue
(longest match wins). One-word titles only count when written with a capital letter, so
"holes in the plot" is not mistaken for the book *Holes*.
"""
import re
from typing import Dict, List, Optional, Tuple

from llm.recommend import normalise

_TARGET_WORDS = {
    "Movies_and_TV": r"\b(watch|watching|film|films|movie|movies|show|shows|series|tv|cinema)\b",
    "Books": r"\b(read|reading|book|books|novel|novels|author|authors)\b",
}
_MORE = re.compile(r"^\s*(more|others?|next|again|another|different ones?|show me more)\b", re.I)
_STOP_SINGLE = {"the", "a", "an", "it", "up", "us", "me", "you", "i", "he", "she", "they", "we", "and", "or",
                "but", "if", "so", "love", "liked", "like", "watch", "read", "more", "books", "movies", "films"}


class MentionFinder:
    """Finds catalogue titles of one domain inside free text."""

    def __init__(self, titles: Dict[str, str], popularity: Optional[Dict[str, int]] = None, max_words: int = 10):
        popularity = popularity or {}
        self.index: Dict[str, List[Tuple[List[str], str, str]]] = {}
        best: Dict[str, Tuple[str, str]] = {}
        for asin, title in sorted(titles.items(), key=lambda at: -popularity.get(at[0], 0)):
            key = normalise(title)
            if key and key not in best and len(key.split()) <= max_words:
                best[key] = (asin, title)
        for key, (asin, title) in best.items():
            words = key.split()
            if len(words) == 1 and (words[0] in _STOP_SINGLE or len(words[0]) < 3):
                continue
            self.index.setdefault(words[0], []).append((words, asin, title))
        for cands in self.index.values():
            cands.sort(key=lambda c: -len(c[0]))           # longest first

    def find(self, message: str) -> List[Dict]:
        raw_words = re.findall(r"[\w']+", message)
        words = [normalise(w) for w in raw_words]
        found, i = [], 0
        while i < len(words):
            match = None
            for cand_words, asin, title in self.index.get(words[i], []):
                n = len(cand_words)
                if words[i:i + n] == cand_words:
                    if n == 1 and not raw_words[i][:1].isupper():
                        continue                            # "holes in the plot" is not *Holes*
                    match = (n, asin, title)
                    break
            if match:
                n, asin, title = match
                if asin not in {f["asin"] for f in found}:
                    found.append({"asin": asin, "title": title})
                i += n
            else:
                i += 1
        return found


# Words just before a domain word that mean "what I already liked": "these books", "my favourite book", "I read".
_LIKED_CONTEXT = re.compile(r"(\b(liked|loved|enjoyed|read|watched|fan of|these|those|my|favourite|favorite|"
                            r"same)\s+(the\s+)?|\bi\s+)$")
_ASKING_I = re.compile(r"\b(should|can|could|would|shall|do|to|might)\s+i\s+$")


def detect_target(message: str, default_target: str) -> str:
    """Which domain does the person want recommendations in?

    Each domain word scores +1 when it describes what they want ("to watch", "some books",
    "should I read") and -1 when it describes what they already liked ("these books",
    "I read", "my favourite book").
    """
    text = message.lower()
    scores = {}
    for domain, pattern in _TARGET_WORDS.items():
        score = 0
        for m in re.finditer(pattern, text):
            before = text[max(0, m.start() - 30):m.start()]
            liked = _LIKED_CONTEXT.search(before) and not _ASKING_I.search(before)
            score += -1 if liked else 1
        scores[domain] = score
    books, films = scores["Books"], scores["Movies_and_TV"]
    if books == films:
        return default_target
    return "Books" if books > films else "Movies_and_TV"


def understand(message: str, finders: Dict[str, MentionFinder], default_source: str, default_target: str,
               previous_seeds: Optional[List[Dict]] = None) -> Dict:
    """Turn a chat message into a recommendation request.

    Returns {source, target, seeds: [{asin, title}], more: bool}. "more"/"others" reuses the
    previous seeds; new titles are added to the previous ones (same direction).
    """
    previous_seeds = previous_seeds or []
    if _MORE.match(message) and previous_seeds:
        return {"source": default_source, "target": default_target, "seeds": previous_seeds, "more": True}

    target = detect_target(message, default_target)
    source = next(d for d in finders if d != target)
    seeds = finders[source].find(message)
    if not seeds:                        # maybe they named titles from the other domain
        other = finders[target].find(message)
        if other:
            source, target, seeds = target, source, other
    if previous_seeds and source == default_source:
        if seeds:                        # "also add Dune" -> keep earlier likes, add the new ones
            seen = {p["asin"] for p in previous_seeds}
            seeds = previous_seeds + [s for s in seeds if s["asin"] not in seen]
        else:                            # "any other ideas?" -> same likes again
            seeds = previous_seeds
    return {"source": source, "target": target, "seeds": seeds, "more": False}
