"""LoRA fine-tuning of a small instruction model on the cross-domain examples.

    python -m llm.train                       # full run (defaults fit a 6 GB GPU)
    python -m llm.train --limit 64 --epochs 1 # quick smoke test

Only the answer tokens (the recommended titles) are trained on; the system and user
prompt are masked out of the loss.
"""
import argparse
import json
import math
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from core.data import load_user_history  # noqa: E402
from llm import DEFAULT_ADAPTER_DIR, DEFAULT_BASE_MODEL  # noqa: E402
from llm.data import build_examples, example_messages  # noqa: E402


def encode(tokenizer, ex, max_len):
    """Token ids + labels with the prompt masked (-100)."""
    full = tokenizer.apply_chat_template(example_messages(ex), tokenize=False)
    prompt = tokenizer.apply_chat_template(example_messages(ex, with_answer=False), tokenize=False,
                                           add_generation_prompt=True)
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    full_ids = full_ids[:max_len]
    labels = [-100] * min(prompt_len, len(full_ids)) + full_ids[prompt_len:]
    return full_ids, labels


def batches(encoded, batch_size, pad_id, rng):
    """Length-bucketed, shuffled batches (less padding = faster on a small GPU)."""
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i][0]))
    chunks = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    rng.shuffle(chunks)
    for chunk in chunks:
        width = max(len(encoded[i][0]) for i in chunk)
        ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        labels = torch.full((len(chunk), width), -100, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for row, i in enumerate(chunk):
            x, y = encoded[i]
            ids[row, :len(x)] = torch.tensor(x)
            labels[row, :len(y)] = torch.tensor(y)
            mask[row, :len(x)] = 1
        yield ids, labels, mask


def main(argv=None):
    p = argparse.ArgumentParser(description="Fine-tune an LLM recommender with LoRA")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--out", default=DEFAULT_ADAPTER_DIR)
    p.add_argument("--user-history", default="data/splits/user_history.json")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-len", type=int, default=640)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--limit", type=int, default=0, help="use only N training examples (smoke test)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    if not torch.cuda.is_available():
        sys.exit("A CUDA GPU is required for training (torch.cuda.is_available() is False).")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print(f"Loading {args.base_model} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype).to("cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    history = load_user_history(os.path.join(ROOT, args.user_history))
    examples = build_examples(history, "train", seed=args.seed)
    if args.limit:
        examples = examples[: args.limit]
    encoded = [encode(tokenizer, ex, args.max_len) for ex in examples]
    print(f"{len(encoded)} training examples, avg {sum(len(e[0]) for e in encoded) / len(encoded):.0f} tokens",
          flush=True)

    steps_per_epoch = math.ceil(len(encoded) / args.batch_size / args.grad_accum)
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    warmup = max(1, int(0.03 * total_steps))
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: min(1.0, (s + 1) / warmup) * max(0.0, (total_steps - s) / max(1, total_steps - warmup)))

    os.makedirs(args.out, exist_ok=True)
    model.train()
    step, micro, running, start = 0, 0, [], time.time()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    epoch = 0
    while step < total_steps:
        epoch += 1
        for ids, labels, mask in batches(encoded, args.batch_size, pad_id, rng):
            out = model(input_ids=ids.cuda(), attention_mask=mask.cuda(), labels=labels.cuda())
            (out.loss / args.grad_accum).backward()
            running.append(out.loss.item())
            micro += 1
            if micro % args.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or step == total_steps:
                elapsed = time.time() - start
                eta = elapsed / step * (total_steps - step)
                print(f"step {step}/{total_steps}  loss {sum(running) / len(running):.4f}  "
                      f"lr {sched.get_last_lr()[0]:.2e}  {elapsed / 60:.1f} min  eta {eta / 60:.1f} min  "
                      f"gpu {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
                running = []
            if step % 300 == 0:
                model.save_pretrained(args.out)
            if step >= total_steps:
                break

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    meta = {"base_model": args.base_model, "examples": len(encoded), "epochs": args.epochs,
            "steps": total_steps, "lr": args.lr, "batch_size": args.batch_size, "grad_accum": args.grad_accum,
            "lora_r": args.lora_r, "max_len": args.max_len, "minutes": round((time.time() - start) / 60, 1)}
    with open(os.path.join(args.out, "llamarec_training.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved LoRA adapter to {args.out}  ({meta['minutes']} min)", flush=True)


if __name__ == "__main__":
    main()
