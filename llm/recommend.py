"""Generate recommendations with the fine-tuned LLM and ground them in the real catalogue.

The model writes free-text titles. Each one is matched to a real product (exact match on
a normalised title, then a strict fuzzy match). Titles that match nothing are counted as
"invented" and never shown.
"""
import html
import re
from typing import Dict, List, Optional

from llm import DEFAULT_ADAPTER_DIR
from llm.data import MAX_TARGET_ITEMS, messages

_FORMAT_WORDS = re.compile(r"\b(vhs|dvd|blu[- ]?ray|hd dvd|4k|uhd|widescreen|full ?screen|paperback|hardcover|"
                           r"edition|unabridged|audio ?cd|import)\b", re.I)


def normalise(title: str) -> str:
    """Lower-case title without format words, bracketed notes or punctuation."""
    t = html.unescape(str(title)).lower()
    t = re.sub(r"[\(\[][^\)\]]*[\)\]]", " ", t)       # (Mandarin Chinese Edition), [Paperback]
    t = _FORMAT_WORDS.sub(" ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def parse_titles(text: str, limit: int = MAX_TARGET_ITEMS) -> List[str]:
    """One title per line; strip bullets, numbering and quotes."""
    out, seen = [], set()
    for line in text.splitlines():
        t = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"\'*').strip()
        if len(t) < 2 or t.lower() in seen:
            continue
        seen.add(t.lower())
        out.append(t)
        if len(out) >= limit:
            break
    return out


class CatalogMatcher:
    """Map free-text titles to real catalogue items of one domain."""

    def __init__(self, titles: Dict[str, str], popularity: Optional[Dict[str, int]] = None, cutoff: int = 90):
        popularity = popularity or {}
        self.cutoff = cutoff
        self.by_norm: Dict[str, str] = {}
        for asin, title in sorted(titles.items(), key=lambda at: -popularity.get(at[0], 0)):
            self.by_norm.setdefault(normalise(title), asin)   # most-liked product wins
        self.by_norm.pop("", None)
        self.keys = list(self.by_norm)

    def match(self, title: str) -> Optional[str]:
        key = normalise(title)
        if not key:
            return None
        if key in self.by_norm:
            return self.by_norm[key]
        from rapidfuzz import fuzz, process
        hit = process.extractOne(key, self.keys, scorer=fuzz.token_sort_ratio, score_cutoff=self.cutoff)
        return self.by_norm[hit[0]] if hit else None


class LLMRecommender:
    """Fine-tuned Qwen + LoRA. Loads lazily on first use (needs torch, transformers, peft)."""

    def __init__(self, adapter_dir: str = DEFAULT_ADAPTER_DIR, base_model: Optional[str] = None,
                 device: Optional[str] = None):
        self.adapter_dir = adapter_dir
        self.base_model = base_model
        self.device = device
        self.model = None
        self.tokenizer = None

    def load(self):
        if self.model is not None:
            return self
        import json
        import os

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base = self.base_model
        meta_path = os.path.join(self.adapter_dir, "llamarec_training.json")
        if base is None and os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                base = json.load(f)["base_model"]
        if base is None:
            from llm import DEFAULT_BASE_MODEL
            base = DEFAULT_BASE_MODEL
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(base)
        self.tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype).to(self.device)
        self.model = PeftModel.from_pretrained(model, self.adapter_dir).eval()
        return self

    def generate(self, requests: List[Dict], use_adapter: bool = True, max_new_tokens: int = 220) -> List[str]:
        """requests: [{"source", "target", "source_titles"}] -> raw generated texts (greedy, batched)."""
        import contextlib

        import torch

        self.load()
        prompts = [self.tokenizer.apply_chat_template(messages(r["source"], r["target"], r["source_titles"]),
                                                      tokenize=False, add_generation_prompt=True) for r in requests]
        batch = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        ctx = contextlib.nullcontext() if use_adapter else self.model.disable_adapter()
        with ctx, torch.no_grad():
            out = self.model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                                      repetition_penalty=1.1, pad_token_id=self.tokenizer.pad_token_id)
        return self.tokenizer.batch_decode(out[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)

    def stream(self, chat_messages: List[Dict], use_adapter: bool = True, max_new_tokens: int = 220,
               sample: bool = False):
        """Yield generated text pieces as they are produced (for the chat assistant)."""
        import contextlib
        import threading

        import torch
        from transformers import TextIteratorStreamer

        self.load()
        prompt = self.tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=120)
        options = dict(max_new_tokens=max_new_tokens, repetition_penalty=1.1,
                       pad_token_id=self.tokenizer.pad_token_id, streamer=streamer)
        options.update(do_sample=True, temperature=0.8, top_p=0.9) if sample else options.update(do_sample=False)
        errors = []

        def run():
            try:
                ctx = contextlib.nullcontext() if use_adapter else self.model.disable_adapter()
                with ctx, torch.no_grad():
                    self.model.generate(**inputs, **options)
            except Exception as exc:  # surface generation errors to the caller
                errors.append(exc)
                streamer.end()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        for piece in streamer:
            yield piece
        thread.join()
        if errors:
            raise errors[0]

    def stream_recommendations(self, source: str, target: str, source_titles: List[str], n: int = MAX_TARGET_ITEMS,
                               sample: bool = False):
        msgs = messages(source, target, source_titles)
        msgs[1]["content"] = msgs[1]["content"].replace(f"Recommend {MAX_TARGET_ITEMS} ", f"Recommend {n} ")
        return self.stream(msgs, use_adapter=True, max_new_tokens=22 * n + 20, sample=sample)

    def stream_explanation(self, source: str, target: str, liked: List[str], picks: List[str]):
        """Short natural-language reason, written by the general instruct model (adapter off)."""
        from llm.data import PRETTY
        msgs = [{"role": "system", "content": "You are LLAMAREC, a friendly recommendation assistant. "
                                              "Be concise and specific. Never invent plot details you are unsure of."},
                {"role": "user", "content":
                    f"Someone liked these {PRETTY[source]}: {', '.join(liked[:8])}.\n"
                    f"We recommended these {PRETTY[target]}: {', '.join(picks[:6])}.\n"
                    "In 2-3 sentences, explain what their taste has in common with these picks. "
                    "Speak to them directly ('you')."}]
        return self.stream(msgs, use_adapter=False, max_new_tokens=140)

    def recommend(self, engine, source: str, target: str, source_titles: List[str], k: int = 10,
                  exclude_asins=(), matcher: Optional[CatalogMatcher] = None) -> Dict:
        """Recommend `target` items for someone who liked `source_titles`, in the engine's output format."""
        matcher = matcher or CatalogMatcher(engine.titles[target], engine.popularity[target])
        text = self.generate([{"source": source, "target": target, "source_titles": source_titles}])[0]
        return ground(text, matcher, engine, source, target, source_titles, k, exclude_asins)


def ground(text: str, matcher: CatalogMatcher, engine, source: str, target: str, source_titles: List[str],
           k: int = 10, exclude_asins=()) -> Dict:
    generated = parse_titles(text, limit=max(k, MAX_TARGET_ITEMS))
    exclude, recs, invented = set(exclude_asins), [], []
    for title in generated:
        asin = matcher.match(title)
        if asin is None:
            invented.append(title)
            continue
        if asin in exclude:
            continue
        exclude.add(asin)
        recs.append({"asin": asin, "title": engine.title(target, asin), "generated_title": title,
                     "score": None, "support": engine.popularity[target][asin], "because": None,
                     "reason": "Suggested by the fine-tuned LLM"})
        if len(recs) >= k:
            break
    return {"source_domain": source, "target_domain": target,
            "seeds": [{"asin": None, "title": t} for t in source_titles],
            "strategy": "fine-tuned LLM", "recommendations": recs,
            "generated": generated, "invented": invented}
