"""LLAMAREC Flask app: web UI + JSON API for data-based cross-domain recommendations.

Run:  python app.py serve        (or: flask --app webapp run)
"""
import os
import threading

from flask import Flask, current_app

from core.data import load_user_history
from core.engine import CrossDomainEngine

_lock = threading.Lock()


def create_app(config: dict = None) -> Flask:
    app = Flask(__name__)
    app.config.update(
        USER_HISTORY_PATH=os.environ.get("LLAMAREC_USER_HISTORY", "data/splits/user_history.json"),
        HISTORY=None,           # tests can inject a dict instead of a file path
        JSON_SORT_KEYS=False,
        LLM_ADAPTER_DIR=os.environ.get("LLAMAREC_LLM_ADAPTER", "models/llamarec-qwen2.5-1.5b-lora"),
        LLM=None,               # tests can inject a stand-in LLM recommender
    )
    if config:
        app.config.update(config)

    from webapp.api import api
    from webapp.views import views
    app.register_blueprint(api, url_prefix="/api")
    app.register_blueprint(views)
    return app


def get_engine() -> CrossDomainEngine:
    """Build the engine once per app (lazily, so the server starts instantly)."""
    ext = current_app.extensions
    if "llamarec_engine" not in ext:
        with _lock:
            if "llamarec_engine" not in ext:
                history = current_app.config["HISTORY"]
                if history is None:
                    history = load_user_history(current_app.config["USER_HISTORY_PATH"])
                ext["llamarec_engine"] = CrossDomainEngine(history)
    return ext["llamarec_engine"]


def llm_available() -> bool:
    """True if a trained adapter exists and the LLM packages are installed (does not load the model)."""
    if current_app.config["LLM"] is not None:
        return True
    if not os.path.isdir(current_app.config["LLM_ADAPTER_DIR"]):
        return False
    import importlib.util
    return all(importlib.util.find_spec(m) for m in ("torch", "transformers", "peft", "rapidfuzz"))


def get_llm():
    """The fine-tuned LLM recommender, created once (the model itself loads on first use)."""
    if current_app.config["LLM"] is not None:
        return current_app.config["LLM"]
    ext = current_app.extensions
    if "llamarec_llm" not in ext:
        with _lock:
            if "llamarec_llm" not in ext:
                from llm.recommend import LLMRecommender
                ext["llamarec_llm"] = LLMRecommender(current_app.config["LLM_ADAPTER_DIR"])
    return ext["llamarec_llm"]
