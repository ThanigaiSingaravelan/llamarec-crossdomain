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


@api.post("/chat")
def chat():
    """Generative chat assistant (Server-Sent Events).

    Body: {message, source_domain, target_domain, k, seeds: [{asin, title}] from earlier turns,
           exclude: [asin] already recommended}
    Events: understood, token, rec, invented, explain, done, error
    """
    import json

    from flask import Response, current_app, stream_with_context

    from llm.chat import MentionFinder, understand
    from llm.recommend import CatalogMatcher, parse_titles

    body = request.get_json(silent=True) or {}
    message = str(body.get("message") or "").strip()[:1000]
    if not message:
        return _error("message is required")
    if not llm_available():
        return _error("The chat assistant needs the fine-tuned LLM. Train it with: python app.py llm-train", 503)
    engine = get_engine()
    source = body.get("source_domain", "Books")
    target = body.get("target_domain", "Movies_and_TV")
    if source not in engine.domains or target not in engine.domains or source == target:
        return _error(f"Unknown or identical domains. Available: {engine.domains}")
    try:
        k = max(1, min(15, int(body.get("k", 8))))
    except (TypeError, ValueError):
        return _error("k must be an integer")
    previous = [s for s in (body.get("seeds") or [])
                if isinstance(s, dict) and s.get("asin") in engine.titles.get(source, {})]
    exclude = {a for a in (body.get("exclude") or []) if isinstance(a, str)}

    ext = current_app.extensions
    if "llamarec_finders" not in ext:
        ext["llamarec_finders"] = {d: MentionFinder(engine.titles[d], engine.popularity[d]) for d in engine.domains}
        ext["llamarec_matchers"] = {d: CatalogMatcher(engine.titles[d], engine.popularity[d]) for d in engine.domains}
    req = understand(message, ext["llamarec_finders"], source, target, previous)
    matcher = ext["llamarec_matchers"][req["target"]]
    llm = get_llm()

    def sse(event):
        return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    def events():
        src, tgt, seeds = req["source"], req["target"], req["seeds"]
        yield sse({"type": "understood", "source": src, "target": tgt, "seeds": seeds, "more": req["more"]})
        if not seeds:
            yield sse({"type": "error", "message": "I couldn't spot any titles I know in that message. "
                                                   "Try naming a few books or films, e.g. "
                                                   "“I loved The Hobbit and Holes, what should I watch?”"})
            yield sse({"type": "done", "seeds": [], "picked": []})
            return
        titles = [s["title"] for s in seeds][:15]
        picked, skip = [], set(exclude) if req["more"] or tgt == target else set()
        try:
            with _gpu_lock:
                buffer = ""
                n = min(30, k + (len(skip) if req["more"] else 0) + 2)
                stream = llm.stream_recommendations(src, tgt, titles, n=n, sample=req["more"])

                def handle(line):
                    for title in parse_titles(line, limit=1):
                        asin = matcher.match(title)
                        if asin is None:
                            return {"type": "invented", "title": title}
                        if asin in skip or len(picked) >= k:
                            return None
                        skip.add(asin)
                        picked.append(asin)
                        return {"type": "rec", "rec": {
                            "asin": asin, "title": engine.title(tgt, asin), "generated_title": title,
                            "support": engine.popularity[tgt][asin], "reason": "Suggested by the fine-tuned LLM"}}
                    return None

                for piece in stream:
                    yield sse({"type": "token", "text": piece})
                    buffer += piece
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        event = handle(line)
                        if event:
                            yield sse(event)
                event = handle(buffer)
                if event:
                    yield sse(event)
                if picked:
                    pick_titles = [engine.title(tgt, a) for a in picked]
                    for piece in llm.stream_explanation(src, tgt, titles, pick_titles):
                        yield sse({"type": "explain", "text": piece})
        except Exception as exc:
            yield sse({"type": "error", "message": f"Generation failed: {exc}"})
        yield sse({"type": "done", "seeds": seeds, "picked": picked, "source": src, "target": tgt})

    return Response(stream_with_context(events()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


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
