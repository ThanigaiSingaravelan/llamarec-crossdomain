"""User-history helpers shared by runners, testers and the API."""
import json
import logging
import os
from functools import lru_cache
from typing import Dict, Iterable, List

logger = logging.getLogger(__name__)


def load_user_history(path: str) -> Dict:
    """Load user_history.json. Returns {} (and logs) if missing or unreadable."""
    if not os.path.exists(path):
        logger.error("User history not found: %s", path)
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.error("Error loading user history %s: %s", path, exc)
        return {}


@lru_cache(maxsize=4)
def _cached(path: str, mtime: float) -> Dict:
    return load_user_history(path)


def load_user_history_cached(path: str) -> Dict:
    """Like load_user_history, but cached until the file changes (used by the API)."""
    mtime = os.path.getmtime(path) if os.path.exists(path) else 0.0
    return _cached(os.path.abspath(path), mtime)


def item_title(item) -> str:
    if isinstance(item, dict):
        return str(item.get("title") or item.get("asin") or item)
    return str(item)


def format_items(items: Iterable, max_items: int = 5, max_len: int = 100) -> str:
    """Format liked items for a prompt: ["Title A", "Title B", ...]."""
    formatted = []
    for item in list(items or [])[:max_items]:
        title = item_title(item).replace('"', "'").strip()
        if len(title) > max_len:
            title = title[: max_len - 3] + "..."
        formatted.append(f'"{title}"')
    return f"[{', '.join(formatted)}]"


def available_domains(history: Dict) -> List[str]:
    domains = set()
    for user_data in history.values():
        domains.update(user_data.keys())
    return sorted(domains)


def eligible_users(history: Dict, source: str, target: str) -> List[str]:
    """Users with liked items in `source` and any data in `target`."""
    return [uid for uid, d in history.items()
            if source in d and target in d and d[source].get("liked")]


def user_profile(history: Dict, user_id: str, sample_size: int = 3) -> Dict:
    """Per-domain summary of one user ({} if unknown)."""
    if user_id not in history:
        return {}
    profile = {"user_id": user_id, "domains": {}, "total_interactions": 0, "domain_count": 0}
    for domain, data in history[user_id].items():
        count = data.get("count", 0)
        if count <= 0:
            continue
        profile["domains"][domain] = {
            "liked_count": len(data.get("liked", [])),
            "disliked_count": len(data.get("disliked", [])),
            "total_count": count,
            "sample_liked": [item_title(i) for i in data.get("liked", [])[:sample_size]],
        }
        profile["total_interactions"] += count
        profile["domain_count"] += 1
    return profile
