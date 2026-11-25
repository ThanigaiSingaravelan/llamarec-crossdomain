"""Turn user histories into chat-style training and test examples.

Every user who liked items in both domains gives two examples:
    "liked these Books"         -> "also liked these Movies_and_TV"
    "liked these Movies_and_TV" -> "also liked these Books"

Users are split 80/20 by a stable hash of their ID, so the split never changes
between runs and test users are never seen in training.
"""
import hashlib
import json
import random
from typing import Dict, Iterable, List

from core.engine import _clean_title, _is_placeholder

DOMAINS = ("Books", "Movies_and_TV")
PRETTY = {"Books": "Books", "Movies_and_TV": "Movies & TV"}
MAX_SOURCE_ITEMS = 15
MAX_TARGET_ITEMS = 10
TEST_FRACTION = 0.2

SYSTEM_PROMPT = ("You are LLAMAREC, a cross-domain recommender. Given titles a person liked in one "
                 "domain, list titles from the other domain they are likely to enjoy. "
                 "Answer with one title per line, most likely first, and nothing else.")


def is_test_user(user_id: str, test_fraction: float = TEST_FRACTION) -> bool:
    bucket = int(hashlib.md5(user_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return bucket < test_fraction


def liked_items(domain_data: Dict) -> List[Dict]:
    """Liked items with clean titles: [{"title", "asin"}]. Placeholders, untitled and duplicate titles removed."""
    out, seen = [], set()
    for item in domain_data.get("liked", []):
        if not isinstance(item, dict):
            continue
        title, asin = item.get("title"), item.get("asin", "")
        if _is_placeholder(title):
            continue
        clean = _clean_title(title, asin)
        if clean == asin or clean.lower() in seen:
            continue  # no usable title
        seen.add(clean.lower())
        out.append({"title": clean, "asin": asin})
    return out


def user_prompt(source: str, target: str, source_titles: Iterable[str], n: int = MAX_TARGET_ITEMS) -> str:
    lines = "\n".join(f"- {t}" for t in list(source_titles)[:MAX_SOURCE_ITEMS])
    return (f"{PRETTY[source]} this person liked:\n{lines}\n\n"
            f"Recommend {n} {PRETTY[target]} titles for them.")


def messages(source: str, target: str, source_titles: List[str], target_titles: List[str] = None) -> List[Dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(source, target, source_titles)}]
    if target_titles is not None:
        msgs.append({"role": "assistant", "content": "\n".join(target_titles[:MAX_TARGET_ITEMS])})
    return msgs


def build_examples(history: Dict, split: str, seed: int = 42) -> List[Dict]:
    """split = "train" or "test".

    source_items: up to MAX_SOURCE_ITEMS liked items given to every method as input.
    target_items: all liked items in the other domain (the ground truth; the first
                  MAX_TARGET_ITEMS titles are the training answer).
    """
    rng = random.Random(seed)
    examples = []
    for uid in sorted(history):
        if is_test_user(uid) != (split == "test"):
            continue
        items = {d: liked_items(history[uid].get(d, {})) for d in DOMAINS}
        for source, target in ((DOMAINS[0], DOMAINS[1]), (DOMAINS[1], DOMAINS[0])):
            src, tgt = items[source][:], items[target][:]
            if not src or not tgt:
                continue
            rng.shuffle(src)
            rng.shuffle(tgt)
            examples.append({"user_id": uid, "source": source, "target": target,
                             "source_items": src[:MAX_SOURCE_ITEMS], "target_items": tgt})
    return examples


def example_messages(ex: Dict, with_answer: bool = True) -> List[Dict]:
    return messages(ex["source"], ex["target"], [i["title"] for i in ex["source_items"]],
                    [i["title"] for i in ex["target_items"]] if with_answer else None)


def train_users(history: Dict) -> Dict:
    """History restricted to training users (used to build fair baselines)."""
    return {u: d for u, d in history.items() if not is_test_user(u)}


def save_jsonl(examples: List[Dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
