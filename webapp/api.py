"""JSON API. All endpoints live under /api."""
import random
import threading

from flask import Blueprint, jsonify, request

from core.data import user_profile
from webapp import get_engine, get_llm, llm_available

api = Blueprint("api", __name__)
_gpu_lock = threading.Lock()   # one generation at a time on the GPU


def _error(message: str, status: int = 400):
    return jsonify({"error": message}), status


def _int_arg(name: str, default: int, lo: int, hi: int) -> int:
    try:
        value = int(request.args.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(lo, min(hi, value))


@api.get("/health")
def health():
    engine = get_engine()
    ready = bool(engine.domains)
    engines = ["co-occurrence"] + (["llm"] if llm_available() else [])
    return jsonify({"status": "ok" if ready else "no-data", "users": len(engine.history),
                    "domains": engine.domains, "engines": engines}), (200 if ready else 503)


@api.get("/domains")
def domains():
    return jsonify(get_engine().domain_stats())


@api.get("/users")
def users():
    engine = get_engine()
    source = request.args.get("source", "Books")
    target = request.args.get("target", "Movies_and_TV")
    limit = _int_arg("limit", 20, 1, 200)
    offset = _int_arg("offset", 0, 0, 10**9)
    ids = engine.users(source, target)
    if request.args.get("random") in ("1", "true", "yes"):
        ids = random.sample(ids, min(limit, len(ids)))
        page = ids
    else:
        page = ids[offset: offset + limit]
    return jsonify({"total": len(ids), "users": [
        {"user_id": u,
         "source_likes": len(engine.user_liked[u].get(source, [])),
         "target_likes": len(engine.user_liked[u].get(target, []))} for u in page]})


@api.get("/users/<user_id>")
def user(user_id):
    profile = user_profile(get_engine().history, user_id)
    return jsonify(profile) if profile else _error(f"User {user_id} not found", 404)


@api.get("/items/search")
def search():
    domain = request.args.get("domain", "Books")
    q = request.args.get("q", "")
    return jsonify({"domain": domain, "query": q,
                    "items": get_engine().search(domain, q, _int_arg("limit", 20, 1, 100))})


@api.post("/recommend")
def recommend():
    """Body: {source_domain, target_domain, k, engine ("co-occurrence" | "llm"),
    and either user_id (+ include_known) or liked_asins: [...]}"""
    body = request.get_json(silent=True) or {}
    engine = get_engine()
    source = body.get("source_domain", "Books")
    target = body.get("target_domain", "Movies_and_TV")
    if source == target:
        return _error("source_domain and target_domain must differ")
    if source not in engine.domains or target not in engine.domains:
        return _error(f"Unknown domain. Available: {engine.domains}")
    try:
        k = max(1, min(50, int(body.get("k", 10))))
    except (TypeError, ValueError):
        return _error("k must be an integer")
    use_llm = body.get("engine", "co-occurrence") == "llm"
    if use_llm and not llm_available():
        return _error("The fine-tuned LLM is not available. Train it with: python app.py llm-train", 503)

    try:
        if body.get("liked_asins"):
            if use_llm:
                seeds = [a for a in body["liked_asins"] if a in engine.titles.get(source, {})]
                if not seeds:
                    return _error(f"None of liked_asins are known {source} items")
                return jsonify(_llm_recommend(engine, source, target, seeds, k, exclude=()))
            result = engine.recommend(source, target, body["liked_asins"], k)
            if not result["seeds"]:
                return _error(f"None of liked_asins are known {source} items")
            return jsonify(result)
        user_id = body.get("user_id")
        if not user_id:
            candidates = engine.users(source, target)
            if not candidates:
                return _error(f"No users with data in both {source} and {target}", 404)
            user_id = random.choice(candidates)
        include_known = bool(body.get("include_known"))
        if use_llm:
            if user_id not in engine.history:
                return _error(f"User {user_id} not found", 404)
            seeds = engine.user_liked.get(user_id, {}).get(source, [])
            if not seeds:
                return _error(f"User {user_id} has no liked items in {source}")
            already = engine.user_liked[user_id].get(target, [])
            result = _llm_recommend(engine, source, target, seeds, k, exclude=() if include_known else already)
            known = set(already)
            for rec in result["recommendations"]:
                rec["liked_by_user"] = rec["asin"] in known
            result["user_id"] = user_id
            result["already_liked_in_target"] = [{"asin": a, "title": engine.title(target, a)} for a in already]
            return jsonify(result)
        return jsonify(engine.recommend_for_user(user_id, source, target, k, include_known=include_known))
    except KeyError as exc:
        return _error(str(exc).strip("'"), 404)
    except ValueError as exc:
        return _error(str(exc))


def _llm_recommend(engine, source, target, seed_asins, k, exclude):
    from llm.data import MAX_SOURCE_ITEMS
    seeds = list(dict.fromkeys(seed_asins))[:MAX_SOURCE_ITEMS]
    titles = [engine.title(source, a) for a in seeds]
    with _gpu_lock:
        result = get_llm().recommend(engine, source, target, titles, k, exclude_asins=exclude)
    result["seeds"] = [{"asin": a, "title": t} for a, t in zip(seeds, titles)]
    return result


@api.get("/evaluate")
def evaluate():
    engine = get_engine()
    source = request.args.get("source", "Books")
    target = request.args.get("target", "Movies_and_TV")
    try:
        return jsonify(engine.evaluate(source, target, k=_int_arg("k", 10, 1, 50),
                                       n_users=_int_arg("n", 200, 10, 2000)))
    except ValueError as exc:
        return _error(str(exc))
