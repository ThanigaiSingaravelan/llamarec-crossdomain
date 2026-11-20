#!/usr/bin/env python
"""
LLAMAREC - single entry point.

Runs every part of the project from the repo root, so imports between
packages (prompts, evaluations, testers, ...) resolve correctly.

Examples:
    python app.py check
    python app.py preprocess all
    python app.py recommend --source Books --target Movies_and_TV
    python app.py interactive
    python app.py coldwarm --mode comparative --max-users 10
    python app.py evaluate --batch --results-dir results
    python app.py ncf --domain Books

Any extra flags after a script-backed command are passed straight through
to the underlying script, e.g. `python app.py coldwarm --help`.
"""

import argparse
import json
import os
import random
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("LLAMAREC_MODEL", "llama3:8b")
USER_HISTORY = os.path.join("data", "splits", "user_history.json")
# Domains present in data/splits/user_history.json (others need re-preprocessing).
DOMAINS = ["Books", "Movies_and_TV"]

# Scripts print emoji; make sure Windows consoles don't crash on them.
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
    cmd = [sys.executable, os.path.join(ROOT, rel_path), *args]
    print(f"> python {rel_path} {' '.join(args)}\n", flush=True)
    return subprocess.call(cmd, cwd=ROOT, env=env)


def ollama_models():
    """Return list of installed Ollama models, or None if unreachable."""
    import requests

    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return None


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def cmd_check(args, extra):
    ok = True
    print("LLAMAREC environment check\n" + "-" * 40)

    models = ollama_models()
    if models is None:
        ok = False
        print(f"[FAIL] Ollama not reachable at {OLLAMA_URL} (run: ollama serve)")
    else:
        print(f"[ OK ] Ollama reachable at {OLLAMA_URL}")
        if args.model in models:
            print(f"[ OK ] Model '{args.model}' installed")
        else:
            ok = False
            print(f"[FAIL] Model '{args.model}' missing (run: ollama pull {args.model})")
            print(f"       Installed: {', '.join(models) or 'none'}")

    paths = {
        "Processed data": os.path.join("data", "processed"),
        "Splits": os.path.join("data", "splits"),
        "User history": USER_HISTORY,
    }
    for label, p in paths.items():
        if os.path.exists(os.path.join(ROOT, p)):
            print(f"[ OK ] {label}: {p}")
        else:
            ok = False
            print(f"[MISS] {label}: {p}  (run: python app.py preprocess all)")

    print("-" * 40)
    print("Ready." if ok else "Some checks failed - see above.")
    return 0 if ok else 1


def cmd_preprocess(args, extra):
    raw, processed, splits = args.raw_dir, args.processed_dir, args.splits_dir
    steps = {
        "collect": ("preprocessing/amazon_collector.py",
                    ["--base_path", raw, "--output_dir", processed]),
        "profile": ("preprocessing/domain_preprocessor.py",
                    ["--base_path", processed, "--output_path", splits]),
        "split": ("preprocessing/data_splitter.py",
                  ["--input_path", splits, "--output_path", splits]),
    }
    order = ["collect", "profile", "split"] if args.step == "all" else [args.step]
    for name in order:
        script, step_args = steps[name]
        print(f"=== Step: {name} ===")
        code = run_script(script, step_args + extra)
        if code != 0:
            print(f"Step '{name}' failed with exit code {code}.")
            return code
    return 0


def cmd_recommend(args, extra):
    from testers.interactive_llamarec_tester import InteractiveLlamaRecTester

    os.chdir(ROOT)
    tester = InteractiveLlamaRecTester(
        user_history_path=args.user_history,
        ollama_model=args.model,
        ollama_url=OLLAMA_URL,
    )
    if not tester.user_history:
        print("No user history loaded. Run: python app.py preprocess all")
        return 1

    user_id = args.user
    if not user_id:
        eligible = [
            uid for uid, d in tester.user_history.items()
            if args.source in d and args.target in d and d[args.source].get("liked")
        ]
        if not eligible:
            print(f"No users have liked items in {args.source} and data in {args.target}.")
            return 1
        user_id = random.choice(eligible)
        print(f"\nPicked random user: {user_id}")

    rec = tester.generate_recommendation(
        user_id=user_id,
        source_domain=args.source,
        target_domain=args.target,
        num_recs=args.num,
        temperature=args.temperature,
    )
    if "error" in rec:
        print(f"Error: {rec['error']}")
        return 1

    print(f"\n{args.source} history: {rec['source_items']}")
    print(f"\n--- {args.target} recommendations ({rec['response_time']:.1f}s) ---\n")
    print(rec["recommendations"])

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {args.save}")
    return 0


def cmd_ncf(args, extra):
    dataset = args.dataset or os.path.join("data", "splits", f"{args.domain}_filtered.csv")
    return run_script(
        "baselines/ncf_evaluation_runner.py",
        ["--dataset", dataset, "--user_history", args.user_history,
         "--domain", args.domain, *extra],
    )


# Commands that just hand off to an existing script.
PASSTHROUGH = {
    "interactive": ("testers/interactive_llamarec_tester.py",
                    "Interactive menu-driven tester"),
    "run": ("runners/run_ollama_llamarec.py",
            "Main LLAMAREC runner (sample evaluation / demo)"),
    "coldwarm": ("runners/run_cold_warm_scenarios.py",
                 "Cold vs warm user study (use --mode, --max-users, ...)"),
    "enhanced": ("runners/run_enhanced_llamarec.py",
                 "Multi-template / multi-model comparison study"),
    "evaluate": ("runners/evaluation_runner.py",
                 "Score result files (Precision/Recall/NDCG/diversity)"),
    "baselines": ("runners/baseline_evaluation.py",
                  "Compare LLAMAREC against baseline result files"),
    "chat": ("scripts/ollama_chat.py", "Plain chat with the local Ollama model"),
}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        prog="app.py",
        description="LLAMAREC: cross-domain recommendations with local LLMs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples:", 1)[1] if "Examples:" in __doc__ else "",
    )
    sub = p.add_subparsers(dest="command", metavar="<command>")

    c = sub.add_parser("check", help="Check Ollama, model and data files")
    c.add_argument("--model", default=DEFAULT_MODEL)
    c.set_defaults(func=cmd_check)

    c = sub.add_parser("preprocess", help="Build processed data and splits")
    c.add_argument("step", nargs="?", default="all",
                   choices=["all", "collect", "profile", "split"])
    c.add_argument("--raw-dir", default=os.path.join("data", "raw"))
    c.add_argument("--processed-dir", default=os.path.join("data", "processed"))
    c.add_argument("--splits-dir", default=os.path.join("data", "splits"))
    c.set_defaults(func=cmd_preprocess)

    c = sub.add_parser("recommend", help="One cross-domain recommendation")
    c.add_argument("--user", help="User ID (random eligible user if omitted)")
    c.add_argument("--source", default=DOMAINS[0], help="e.g. Books")
    c.add_argument("--target", default=DOMAINS[1], help="e.g. Movies_and_TV")
    c.add_argument("--num", type=int, default=3)
    c.add_argument("--temperature", type=float, default=0.7)
    c.add_argument("--model", default=DEFAULT_MODEL)
    c.add_argument("--user-history", default=USER_HISTORY)
    c.add_argument("--save", metavar="FILE", help="Save result as JSON")
    c.set_defaults(func=cmd_recommend)

    c = sub.add_parser("ncf", help="Train and evaluate the NCF baseline")
    c.add_argument("--domain", default=DOMAINS[0])
    c.add_argument("--dataset", help="CSV path (default: data/splits/<domain>_filtered.csv)")
    c.add_argument("--user-history", default=USER_HISTORY)
    c.set_defaults(func=cmd_ncf)

    c = sub.add_parser("test", help="Run the unit tests (pytest)", add_help=False)
    c.set_defaults(func=lambda a, extra: subprocess.call(
        [sys.executable, "-m", "pytest", *extra], cwd=ROOT))

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
    if extra and args.command not in PASSTHROUGH and args.command not in ("preprocess", "ncf", "test"):
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    try:
        return args.func(args, extra) or 0
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
