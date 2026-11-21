"""Data-based cross-domain recommender (no LLM, no Ollama).

Idea: "people who liked these Books also liked these Movies".
For each item the user liked in the source domain, find the other users who
liked the same item, and count what *they* liked in the target domain.
Scores are damped by target-item popularity so the list is not just blockbusters,
and every recommendation carries the source item that contributed most to it,
which gives a human-readable explanation.

Built entirely from data/splits/user_history.json.
"""
import html
import random
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Set

from core.metrics import jaccard_diversity, ndcg_at_k, precision_at_k, recall_at_k


class CrossDomainEngine:
    def __init__(self, history: Dict, popularity_damping: float = 0.1, smoothing: float = 10.0):
        """
        Args:
            history: {user_id: {domain: {"liked": [{"asin", "title", "rating"}], ...}}}
            popularity_damping: 0 = raw co-occurrence counts, 1 = fully popularity-normalised.
            smoothing: added to popularity before damping, so items liked by only one or
                two users cannot outrank well-supported ones by chance.
        """
        self.history = history
        self.alpha = popularity_damping
        self.smoothing = smoothing
        self.titles: Dict[str, Dict[str, str]] = defaultdict(dict)
        self.liked_by: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        self.user_liked: Dict[str, Dict[str, List[str]]] = defaultdict(dict)

        for uid, domains in history.items():
            for domain, data in domains.items():
                asins = []
                for item in data.get("liked", []):
                    asin = item.get("asin") if isinstance(item, dict) else str(item)
                    if not asin:
                        continue
                    title = item.get("title") if isinstance(item, dict) else None
                    self.titles[domain].setdefault(asin, _clean_title(title, asin))
                    self.liked_by[domain][asin].add(uid)
                    asins.append(asin)
                if asins:
                    self.user_liked[uid][domain] = list(dict.fromkeys(asins))

        self.popularity: Dict[str, Counter] = {
            d: Counter({a: len(u) for a, u in items.items()}) for d, items in self.liked_by.items()
        }

    # ------------------------------------------------------------------ info
    @property
    def domains(self) -> List[str]:
        return sorted(self.liked_by)

    def domain_stats(self) -> Dict[str, Dict[str, int]]:
        return {d: {"items": len(self.liked_by[d]),
                    "users": sum(1 for u in self.user_liked.values() if d in u)}
                for d in self.domains}

    def title(self, domain: str, asin: str) -> str:
        return self.titles.get(domain, {}).get(asin, asin)

    def users(self, source: str, target: str) -> List[str]:
        """Users with liked items in both domains."""
        return [u for u, d in self.user_liked.items() if source in d and target in d]

    def search(self, domain: str, query: str, limit: int = 20) -> List[Dict]:
        q = (query or "").strip().lower()
        if not q:
            return []
        hits = [(a, t) for a, t in self.titles.get(domain, {}).items() if q in t.lower()]
        hits.sort(key=lambda at: (-self.popularity[domain][at[0]], at[1]))
        return [{"asin": a, "title": t, "likes": self.popularity[domain][a]} for a, t in hits[:limit]]

    # --------------------------------------------------------- recommending
    def popular(self, target: str, k: int = 10, exclude: Iterable[str] = ()) -> List[Dict]:
        exclude = set(exclude)
        out = []
        for asin, likes in self.popularity.get(target, Counter()).most_common():
            if asin in exclude:
                continue
            out.append({"asin": asin, "title": self.title(target, asin), "score": float(likes),
                        "support": likes, "because": None,
                        "reason": f"Popular in {target}: liked by {likes} users"})
            if len(out) >= k:
                break
        return out

    def recommend(self, source: str, target: str, seed_asins: Iterable[str], k: int = 10,
                  exclude_user: Optional[str] = None, exclude_items: Iterable[str] = ()) -> Dict:
        """Recommend `target` items for someone who liked `seed_asins` in `source`.

        exclude_user: ignore this user's own row (needed for honest evaluation).
        exclude_items: target items not to recommend (e.g. ones already liked).
        """
        if source not in self.liked_by or target not in self.liked_by:
            raise ValueError(f"Unknown domain. Available: {self.domains}")
        seeds = [a for a in dict.fromkeys(seed_asins) if a in self.liked_by[source]]
        exclude_items = set(exclude_items)

        scores: Counter = Counter()
        support: Counter = Counter()
        best_seed: Dict[str, Counter] = defaultdict(Counter)
        for seed in seeds:
            for neighbour in self.liked_by[source][seed]:
                if neighbour == exclude_user:
                    continue
                for t in self.user_liked[neighbour].get(target, ()):
                    if t in exclude_items:
                        continue
                    support[t] += 1
                    best_seed[t][seed] += 1

        pop = self.popularity[target]
        for t, count in support.items():
            scores[t] = count / ((pop[t] + self.smoothing) ** self.alpha)

        recs = []
        for asin, score in scores.most_common(k):
            seed, n = best_seed[asin].most_common(1)[0]
            seed_title = self.title(source, seed)
            recs.append({
                "asin": asin,
                "title": self.title(target, asin),
                "score": round(score, 4),
                "support": support[asin],
                "because": {"asin": seed, "title": seed_title},
                "reason": f"{n} user{'s' if n != 1 else ''} who liked '{seed_title}' also liked this",
            })

        strategy = "co-occurrence"
        if len(recs) < k:  # cold start / sparse: top up with popular items
            have = {r["asin"] for r in recs} | exclude_items
            recs += self.popular(target, k - len(recs), exclude=have)
            strategy = "co-occurrence + popularity fallback" if scores else "popularity (no overlap found)"

        return {
            "source_domain": source,
            "target_domain": target,
            "seeds": [{"asin": s, "title": self.title(source, s)} for s in seeds],
            "strategy": strategy,
            "recommendations": recs,
        }

    def recommend_for_user(self, user_id: str, source: str, target: str, k: int = 10,
                           include_known: bool = False) -> Dict:
        """Recommend for an existing user. Their own row is always ignored.

        include_known=False: skip target items they already liked (normal use).
        include_known=True:  allow them, and flag matches - an honest "would we have
                             predicted what they actually liked?" check.
        """
        if user_id not in self.history:
            raise KeyError(f"User {user_id} not found")
        seeds = self.user_liked.get(user_id, {}).get(source, [])
        if not seeds:
            raise ValueError(f"User {user_id} has no liked items in {source}")
        already = self.user_liked[user_id].get(target, [])
        result = self.recommend(source, target, seeds, k, exclude_user=user_id,
                                exclude_items=() if include_known else already)
        known = set(already)
        for rec in result["recommendations"]:
            rec["liked_by_user"] = rec["asin"] in known
        result["user_id"] = user_id
        result["already_liked_in_target"] = [{"asin": a, "title": self.title(target, a)} for a in already]
        return result

    # ------------------------------------------------------------ evaluation
    def evaluate(self, source: str, target: str, k: int = 10, n_users: int = 200,
                 seed: int = 42) -> Dict:
        """Leave-one-user-out evaluation.

        For each sampled user, recommend from their source likes while ignoring their
        own row, then check the list against what they really liked in the target domain.
        The same users are scored with a popularity baseline for comparison.
        """
        users = self.users(source, target)
        if not users:
            raise ValueError(f"No users with likes in both {source} and {target}")
        rng = random.Random(seed)
        sample = rng.sample(users, min(n_users, len(users)))

        def score(method: str) -> Dict:
            p, r, n, hits, div = [], [], [], 0, []
            for u in sample:
                truth = self.user_liked[u][target]
                if method == "popularity":
                    # leave-one-user-out popularity
                    pop = self.popularity[target].copy()
                    pop.subtract({a: 1 for a in truth})
                    preds = [a for a, _ in pop.most_common(k)]
                else:
                    preds = [x["asin"] for x in self.recommend(
                        source, target, self.user_liked[u][source], k, exclude_user=u)["recommendations"]]
                p.append(precision_at_k(preds, truth, k))
                r.append(recall_at_k(preds, truth, k))
                n.append(ndcg_at_k(preds, truth, k))
                hits += any(a in set(truth) for a in preds)
                div.append(jaccard_diversity([self.title(target, a) for a in preds]))
            m = len(sample)
            return {name: round(float(v), 4) for name, v in {
                "precision": sum(p) / m, "recall": sum(r) / m, "ndcg": sum(n) / m,
                "hit_rate": hits / m, "diversity": sum(div) / m}.items()}

        return {"source_domain": source, "target_domain": target, "k": k,
                "users_evaluated": len(sample), "users_available": len(users),
                "methods": {"co-occurrence": score("co-occurrence"),
                            "popularity": score("popularity")},
                "note": "Leave-one-user-out: each user's own data is excluded when scoring them."}


def _clean_title(title, asin: str) -> str:
    """Titles in the dataset can be missing (NaN) or contain HTML entities."""
    if not isinstance(title, str) or not title.strip() or title.strip().lower() == "nan":
        return asin
    return html.unescape(title).strip()
