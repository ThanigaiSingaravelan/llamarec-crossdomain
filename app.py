#!/usr/bin/env python
"""
LLAMAREC - single entry point.

Examples:
    python app.py serve                      # web UI + JSON API at http://127.0.0.1:5000
    python app.py check                      # packages and data files present?
    python app.py recommend --source Books --target Movies_and_TV
    python app.py evaluate-engine --k 10 --n 300
    python app.py preprocess all             # raw Amazon files -> data/splits
    python app.py ncf --domain Books
    python app.py llm-train                  # fine-tune the LLM (GPU, requirements-llm.txt)
    python app.py llm-eval --n 300           # LLM vs co-occurrence vs popularity
    python app.py test

Extra flags after `preprocess`, `ncf`, `evaluate`, `baselines` and `test` are
passed straight through to the underlying script or pytest.
"""

import argparse
import json
import os
import random
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
USER_HISTORY = os.path.join("data", "splits", "user_history.json")
# Domains present in data/splits/user_history.json (others need re-preprocessing).
DOMAINS = ["Books", "Movies_and_TV"]

# Some scripts print emoji; make sure Windows consoles don't crash on them.
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def run_script(rel_path, args):
    """Run a project script from the repo root with PYTHONPATH set."""
    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    print(f"> python {rel_path} {' '.join(args)}\n", flush=True)
    return subprocess.call([sys.executable, os.path.join(ROOT, rel_path), *args], cwd=ROOT, env=env)


def load_engine(path):
    from core.data import load_user_history
    from core.engine import CrossDomainEngine

    history = load_user_history(os.path.join(ROOT, path))
    if not history:
        print(f"No user history at {path}. Run: python app.py preprocess all")
        return None
    return CrossDomainEngine(history)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def cmd_serve(args, extra):
    from webapp import create_app

    os.chdir(ROOT)
    app = create_app({"USER_HISTORY_PATH": os.path.join(ROOT, args.user_history)})
    print(f"LLAMAREC running at http://{args.host}:{args.port}  (Ctrl+C to stop)")
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


def cmd_check(args, extra):
    ok = True
    print("LLAMAREC environment check\n" + "-" * 40)
    for module in ("flask", "numpy", "pandas"):
        try:
            __import__(module)
            print(f"[ OK ] Python package: {module}")
        except ImportError:
            ok = False
            print(f"[FAIL] Python package missing: {module}  (run: pip install -r requirements.txt)")

    for label, p in (("Processed data", os.path.join("data", "processed")),
                     ("Splits", os.path.join("data", "splits")),
                     ("User history", USER_HISTORY)):
        if os.path.exists(os.path.join(ROOT, p)):
            print(f"[ OK ] {label}: {p}")
        else:
            ok = False
            print(f"[MISS] {label}: {p}  (run: python app.py preprocess all)")

    print("-" * 40)
    print("Ready. Start the app with: python app.py serve" if ok else "Some checks failed - see above.")
    return 0 if ok else 1


def cmd_recommend(args, extra):
    engine = load_engine(args.user_history)
    if engine is None:
        return 1
    try:
        user_id = args.user
        if not user_id:
            users = engine.users(args.source, args.target)
            if not users:
                print(f"No users with likes in both {args.source} and {args.target}.")
                return 1
            user_id = random.choice(users)
        result = engine.recommend_for_user(user_id, args.source, args.target, args.num)
    except (KeyError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    print(f"\nUser {result['user_id']} liked in {args.source}:")
    for s in result["seeds"][:10]:
        print(f"  - {s['title']}")
    print(f"\nRecommended {args.target} ({result['strategy']}):")
    for i, r in enumerate(result["recommendations"], 1):
        print(f"  {i:>2}. {r['title']}\n      {r['reason']}")

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {args.save}")
    return 0


def cmd_evaluate_engine(args, extra):
    engine = load_engine(args.user_history)
    if engine is None:
        return 1
    for src, tgt in ((args.source, args.target), (args.target, args.source)):
        res = engine.evaluate(src, tgt, k=args.k, n_users=args.n)
        print(f"\n{src} -> {tgt}  (k={args.k}, {res['users_evaluated']} users, leave-one-user-out)")
        print(f"  {'metric':<10}" + "".join(f"{m:>16}" for m in res["methods"]))
        for metric in ("hit_rate", "precision", "recall", "ndcg", "diversity"):
            print(f"  {metric:<10}" + "".join(f"{res['methods'][m][metric]:>16.4f}" for m in res["methods"]))
    return 0


def cmd_preprocess(args, extra):
    raw, processed, splits = args.raw_dir, args.processed_dir, args.splits_dir
    steps = {
        "collect": ("preprocessing/amazon_collector.py", ["--base_path", raw, "--output_dir", processed]),
        "profile": ("preprocessing/domain_preprocessor.py", ["--base_path", processed, "--output_path", splits]),
        "split": ("preprocessing/data_splitter.py", ["--input_path", splits, "--output_path", splits]),
    }
    for name in (["collect", "profile", "split"] if args.step == "all" else [args.step]):
        script, step_args = steps[name]
        print(f"=== Step: {name} ===")
        code = run_script(script, step_args + extra)
        if code != 0:
            print(f"Step '{name}' failed with exit code {code}.")
            return code
    return 0


def cmd_ncf(args, extra):
    dataset = args.dataset or os.path.join("data", "splits", f"{args.domain}_filtered.csv")
    return run_script("evaluation/ncf_evaluation_runner.py",
                      ["--dataset", dataset, "--user_history", args.user_history,
                       "--domain", args.domain, *extra])


def cmd_test(args, extra):
    return subprocess.call([sys.executable, "-m", "pytest", *extra], cwd=ROOT)


# Commands that hand off to an existing script.
PASSTHROUGH = {
    "evaluate": ("evaluation/evaluation_runner.py",
                 "Score saved result files in results/ (Precision/Recall/NDCG)"),
    "baselines": ("evaluation/baseline_evaluation.py",
                  "Compare saved result files against baselines"),
    "llm-train": ("llm/train.py",
                  "Fine-tune the LLM recommender with LoRA (needs requirements-llm.txt + GPU)"),
    "llm-eval": ("llm/evaluate.py",
                 "Compare the fine-tuned LLM with co-occurrence and popularity on held-out users"),
}
FORWARDS_EXTRA = {"preprocess", "ncf", "test", *PASSTHROUGH}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        prog="app.py",
        description="LLAMAREC: cross-domain recommendations (web app, CLI, evaluation).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples:", 1)[1],
    )
    sub = p.add_subparsers(dest="command", metavar="<command>")

    c = sub.add_parser("serve", help="Start the web UI + JSON API (Flask)")
    c.add_argument("--host", default="127.0.0.1")
    c.add_argument("--port", type=int, default=5000)
    c.add_argument("--debug", action="store_true")
    c.add_argument("--user-history", default=USER_HISTORY)
    c.set_defaults(func=cmd_serve)

    c = sub.add_parser("check", help="Check packages and data files")
    c.set_defaults(func=cmd_check)

    c = sub.add_parser("recommend", help="Recommendations for one user (terminal)")
    c.add_argument("--user", help="User ID (random eligible user if omitted)")
    c.add_argument("--source", default=DOMAINS[0], help="e.g. Books")
    c.add_argument("--target", default=DOMAINS[1], help="e.g. Movies_and_TV")
    c.add_argument("--num", type=int, default=10)
    c.add_argument("--user-history", default=USER_HISTORY)
    c.add_argument("--save", metavar="FILE", help="Save result as JSON")
    c.set_defaults(func=cmd_recommend)

    c = sub.add_parser("evaluate-engine", help="Leave-one-user-out evaluation (both directions)")
    c.add_argument("--source", default=DOMAINS[0])
    c.add_argument("--target", default=DOMAINS[1])
    c.add_argument("--k", type=int, default=10)
    c.add_argument("--n", type=int, default=300, help="users to sample")
    c.add_argument("--user-history", default=USER_HISTORY)
    c.set_defaults(func=cmd_evaluate_engine)

    c = sub.add_parser("preprocess", help="Build processed data and splits")
    c.add_argument("step", nargs="?", default="all", choices=["all", "collect", "profile", "split"])
    c.add_argument("--raw-dir", default=os.path.join("data", "raw"))
    c.add_argument("--processed-dir", default=os.path.join("data", "processed"))
    c.add_argument("--splits-dir", default=os.path.join("data", "splits"))
    c.set_defaults(func=cmd_preprocess)

    c = sub.add_parser("ncf", help="Train and evaluate the NCF baseline")
    c.add_argument("--domain", default=DOMAINS[0])
    c.add_argument("--dataset", help="CSV path (default: data/splits/<domain>_filtered.csv)")
    c.add_argument("--user-history", default=USER_HISTORY)
    c.set_defaults(func=cmd_ncf)

    c = sub.add_parser("test", help="Run the unit tests (pytest)", add_help=False)
    c.set_defaults(func=cmd_test)

    for name, (script, help_text) in PASSTHROUGH.items():
        c = sub.add_parser(name, help=help_text, add_help=False)
        c.set_defaults(func=lambda a, extra, s=script: run_script(s, extra))

    return p


def main(argv=None):
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    if extra and args.command not in FORWARDS_EXTRA:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    try:
        return args.func(args, extra) or 0
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
