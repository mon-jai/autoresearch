"""
Stage 2-017: Curriculum learning on gold SciERC data.

E2H (ACL Findings 2023) + CSJE (KBS 2025) show curriculum learning
helps joint NER+RE, especially in low-resource settings. We implement
a simple difficulty-based curriculum:

Difficulty score per sentence = n_entities * max(n_relations, 1)
  - Easy: 1-2 entities, 0-1 relations
  - Medium: 2-4 entities, 1-3 relations
  - Hard: 4+ entities, 3+ relations

Training phases:
  Phase 1 (0 → phase1_steps): easy + medium sentences only
  Phase 2 (phase1_steps → end): all sentences (standard)

The hypothesis: easier examples build robust representations first,
then harder examples fine-tune for complex extractions.

Usage:
    # Curriculum
    uv run python train_stage2_curriculum.py --max-steps 1500 --phase1-frac 0.4

    # Baseline (no curriculum, all data from start)
    uv run python train_stage2_curriculum.py --max-steps 1500 --phase1-frac 0.0
"""
import argparse
import random
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data.scierc import (
    SciERCSentenceDataset, collate_fn, build_dataloaders,
)
from models.bert_kg_encoder import BertKGExtractor, compute_loss
from eval.triple_f1 import evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="allenai/scibert_scivocab_uncased")
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--warmup-steps", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--re-weight", type=float, default=1.0)
    p.add_argument("--phase1-frac", type=float, default=0.4,
                   help="Fraction of steps for easy-only phase. 0=no curriculum.")
    p.add_argument("--hard-threshold", type=float, default=0.5,
                   help="Top fraction of difficulty scores to exclude in phase 1.")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    # CAST pseudo-label support (optional, combinable with curriculum)
    p.add_argument("--synth-jsonl", default="",
                   help="Optional pseudo-label jsonl (CAST or flat). Empty=gold only.")
    p.add_argument("--synth-weight", type=float, default=0.1)
    p.add_argument("--synth-start-frac", type=float, default=0.4,
                   help="Start mixing pseudo-labels at this fraction of total steps.")
    return p.parse_args()


def compute_difficulty(example):
    """Score sentence difficulty: n_entities * max(n_relations, 1)."""
    n_ent = len(example["ner"])
    n_rel = len(example["relations"])
    return n_ent * max(n_rel, 1)


def cycle(loader):
    while True:
        yield from loader


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_curriculum = args.phase1_frac > 0
    use_synth = bool(args.synth_jsonl)

    print(f"=== Stage 2-017: Curriculum learning ===")
    print(f"  curriculum: {'ON' if use_curriculum else 'OFF (baseline)'}")
    if use_curriculum:
        print(f"  phase1 frac: {args.phase1_frac} ({int(args.phase1_frac * args.max_steps)} steps)")
        print(f"  hard threshold: {args.hard_threshold}")
    if use_synth:
        print(f"  synth: {args.synth_jsonl} (w={args.synth_weight})")
    print(f"  steps: {args.max_steps}, seed: {args.seed}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    data_dir = Path(args.data_dir) if args.data_dir else None

    # Build full dataset
    full_train, dev_loader, _ = build_dataloaders(
        tokenizer, data_dir=data_dir,
        batch_size=args.batch_size, max_length=args.max_length,
    )
    train_dataset = full_train.dataset
    print(f"  train sentences: {len(train_dataset)}")
    print(f"  dev sentences: {len(dev_loader.dataset)}")

    # Compute difficulty scores
    difficulties = []
    for i in range(len(train_dataset)):
        ex = train_dataset.examples[i]
        d = compute_difficulty(ex)
        difficulties.append((i, d))

    difficulties.sort(key=lambda x: x[1])
    n_easy = int(len(difficulties) * (1 - args.hard_threshold))

    easy_indices = [idx for idx, d in difficulties[:n_easy]]
    all_indices = list(range(len(train_dataset)))

    if use_curriculum:
        print(f"  easy sentences: {len(easy_indices)} (difficulty ≤ {difficulties[n_easy-1][1] if n_easy > 0 else 'N/A'})")
        print(f"  hard sentences: {len(train_dataset) - len(easy_indices)}")

        # Difficulty distribution
        from collections import Counter
        d_counts = Counter(d for _, d in difficulties)
        print(f"  difficulty distribution: {dict(sorted(d_counts.items())[:8])}...")

    # Build loaders
    if use_curriculum:
        easy_loader = DataLoader(
            Subset(train_dataset, easy_indices),
            batch_size=args.batch_size, shuffle=True,
            collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        )
    full_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    synth_loader = None
    if use_synth:
        from data.synth_loader import build_synth_loader
        synth_loader = build_synth_loader(
            tokenizer, args.synth_jsonl,
            batch_size=args.batch_size, max_length=args.max_length,
            min_containment=0.0,
        )
        print(f"  synth sentences: {len(synth_loader.dataset)}")

    # Model
    model = BertKGExtractor(args.model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    phase1_steps = int(args.phase1_frac * args.max_steps) if use_curriculum else 0
    synth_start = int(args.synth_start_frac * args.max_steps) if use_synth else args.max_steps

    easy_iter = cycle(easy_loader) if use_curriculum else None
    full_iter = cycle(full_loader)
    synth_iter = cycle(synth_loader) if synth_loader else None

    best_metrics = {"triple_f1": -1.0}
    best_step = -1

    model.train()
    t0 = time.time()
    for step in range(args.max_steps):
        optimizer.zero_grad()

        # Select data source based on phase
        if use_curriculum and step < phase1_steps:
            batch = next(easy_iter)
            phase = "easy"
        else:
            batch = next(full_iter)
            phase = "all"

        gold_loss, ner_loss, re_loss, _ = compute_loss(
            model, batch, device, re_weight=args.re_weight,
        )

        # Optional synth
        synth_loss_val = 0.0
        if use_synth and step >= synth_start and synth_iter:
            synth_batch = next(synth_iter)
            try:
                s_loss, _, _, _ = compute_loss(
                    model, synth_batch, device, re_weight=args.re_weight,
                )
                synth_loss_val = s_loss.item()
                total = gold_loss + args.synth_weight * s_loss
            except Exception:
                total = gold_loss
        else:
            total = gold_loss

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 50 == 0:
            print(f"  [Step {step:04d} {phase}] L_gold={gold_loss.item():.4f} "
                  f"L_ner={ner_loss.item():.4f} L_re={re_loss.item():.4f}")

        if step > 0 and step % args.eval_every == 0:
            model.eval()
            metrics = evaluate(model, dev_loader, device)
            star = ""
            if metrics["triple_f1"] > best_metrics["triple_f1"]:
                best_metrics = dict(metrics)
                best_step = step
                star = " ★"
            print(f"  [Eval {step}] NER={metrics['ner_f1']:.4f} RE={metrics['re_f1']:.4f} "
                  f"Triple={metrics['triple_f1']:.4f}{star}")
            model.train()

    # Final eval
    model.eval()
    metrics = evaluate(model, dev_loader, device)
    if metrics["triple_f1"] > best_metrics["triple_f1"]:
        best_metrics = dict(metrics)
        best_step = args.max_steps

    print(f"\n=== FINAL (step {args.max_steps}) ===")
    print(f"  NER F1    = {metrics['ner_f1']:.4f}")
    print(f"  RE F1     = {metrics['re_f1']:.4f}")
    print(f"  Triple F1 = {metrics['triple_f1']:.4f}")
    print(f"=== BEST DEV (step {best_step}, by Triple F1) ===")
    print(f"  NER F1    = {best_metrics['ner_f1']:.4f}")
    print(f"  RE F1     = {best_metrics['re_f1']:.4f}")
    print(f"  Triple F1 = {best_metrics['triple_f1']:.4f}")
    print(f"  Total time = {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
