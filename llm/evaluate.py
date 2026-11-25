"""Compare the fine-tuned LLM with the other methods on held-out users.

    python -m llm.evaluate --n 300

All methods get exactly the same input (the same <= 15 liked source items of a test
user) and are scored the same way: a recommendation is a hit if its normalised title
matches a title the user really liked in the target domain. The co-occurrence engine
and the popularity baseline are rebuilt from the training users only, so no method
has seen the test users.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.data import load_user_history  # noqa: E402
from core.engine import CrossDomainEngine  # noqa: E402
from core.metrics import jaccard_diversity  # noqa: E402
from llm import DEFAULT_ADAPTER_DIR  # noqa: E402
from llm.data import DOMAINS, build_examples, train_users  # noqa: E402
from llm.recommend import CatalogMatcher, LLMRecommender, ground, normalise  # noqa: E402


def score(pred_titles, truth_titles, k):
    truth = {normalise(t) for t in truth_titles} - {""}
    preds = [normalise(t) for t in pred_titles[:k]]
    hits = [p in truth for p in preds]
    dcg = sum(1 / math.log2(i + 2) for i, h in enumerate(hits) if h)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(truth), k)))
    return {"hit_rate": float(any(hits)), "precision": sum(hits) / k,
            "recall": sum(hits) / len(truth) if truth else 0.0, "ndcg": dcg / idcg if idcg else 0.0,
            "diversity": jaccard_diversity(pred_titles[:k])}


def average(rows):
    return {m: round(sum(r[m] for r in rows) / len(rows), 4) for m in rows[0]} if rows else {}


def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluate the fine-tuned LLM against the other methods")
    p.add_argument("--adapter", default=DEFAULT_ADAPTER_DIR)
    p.add_argument("--user-history", default="data/splits/user_history.json")
    p.add_argument("--n", type=int, default=300, help="test examples per direction")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--skip-base", action="store_true", help="don't evaluate the untrained base model")
    p.add_argument("--out", default="results")
    args = p.parse_args(argv)

    history = load_user_history(os.path.join(ROOT, args.user_history))
    engine = CrossDomainEngine(train_users(history))       # baselines see training users only
    tests = build_examples(history, "test")
    rng = random.Random(7)
    llm = LLMRecommender(os.path.join(ROOT, args.adapter)).load()

    report = {"date": datetime.now().isoformat(timespec="seconds"), "k": args.k, "adapter": args.adapter,
              "note": "Held-out users; every method gets the same <=15 liked source items.", "directions": {}}
    for source, target in ((DOMAINS[0], DOMAINS[1]), (DOMAINS[1], DOMAINS[0])):
        pool = [e for e in tests if e["source"] == source]
        sample = rng.sample(pool, min(args.n, len(pool)))
        matcher = CatalogMatcher(engine.titles[target], engine.popularity[target])
        popular = [engine.title(target, a) for a, _ in engine.popularity[target].most_common(args.k)]
        rows = {"popularity": [], "co-occurrence": [], "fine-tuned LLM": []}
        if not args.skip_base:
            rows["base LLM (untrained)"] = []
        invented = {name: [0, 0] for name in rows if "LLM" in name}
        start = time.time()

        for i in range(0, len(sample), args.batch_size):
            chunk = sample[i:i + args.batch_size]
            reqs = [{"source": source, "target": target, "source_titles": [s["title"] for s in ex["source_items"]]}
                    for ex in chunk]
            texts = {"fine-tuned LLM": llm.generate(reqs, use_adapter=True)}
            if not args.skip_base:
                texts["base LLM (untrained)"] = llm.generate(reqs, use_adapter=False)
            for j, ex in enumerate(chunk):
                truth = [t["title"] for t in ex["target_items"]]
                rows["popularity"].append(score(popular, truth, args.k))
                co = engine.recommend(source, target, [s["asin"] for s in ex["source_items"]], args.k)
                rows["co-occurrence"].append(score([r["title"] for r in co["recommendations"]], truth, args.k))
                for name, outs in texts.items():
                    g = ground(outs[j], matcher, engine, source, target, reqs[j]["source_titles"], args.k)
                    rows[name].append(score([r["title"] for r in g["recommendations"]], truth, args.k))
                    invented[name][0] += len(g["invented"])
                    invented[name][1] += len(g["generated"])
            done = min(i + args.batch_size, len(sample))
            print(f"{source} -> {target}: {done}/{len(sample)}  {time.time() - start:.0f}s", flush=True)

        result = {name: average(r) for name, r in rows.items()}
        for name, (bad, total) in invented.items():
            result[name]["invented_titles"] = round(bad / total, 4) if total else 0.0
        report["directions"][f"{source} -> {target}"] = {"examples": len(sample), "methods": result}

    os.makedirs(os.path.join(ROOT, args.out), exist_ok=True)
    path = os.path.join(ROOT, args.out, f"llm_evaluation_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    for direction, d in report["directions"].items():
        print(f"\n{direction}  ({d['examples']} held-out examples, k={args.k})")
        names = list(d["methods"])
        print(f"  {'metric':<16}" + "".join(f"{n:>24}" for n in names))
        for metric in ("hit_rate", "precision", "recall", "ndcg", "diversity", "invented_titles"):
            cells = [d["methods"][n].get(metric) for n in names]
            print(f"  {metric:<16}" + "".join(f"{c:>24.4f}" if c is not None else f"{'-':>24}" for c in cells))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
