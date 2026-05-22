"""
Stage 2a training: BERT-base on SciERC with NER + RE supervision.

This is the simplest possible "real" baseline:
- Real text (SciERC abstracts, sentence-level)
- Real labels (NER + RE)
- Standard supervised training (no adversarial loop yet)
- Triple F1 as the headline metric

Run:
    cd autoresearch
    uv sync                                    # install transformers, datasets
    python data/download_scierc.py             # one-time data download
    python train_stage2.py                     # smoke test
    python train_stage2.py --max-steps 1000    # short training
"""
import argparse
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data.scierc import build_dataloaders, NUM_BIO_TAGS, NUM_RELATIONS
from models.bert_kg_encoder import BertKGExtractor, compute_loss
from eval.triple_f1 import evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="bert-base-uncased")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--warmup-steps", type=int, default=0,
                   help="Linear warmup from 0 to lr over this many steps. "
                        "Stage 2-005 default: 100. Stage 2-001..004 used 0.")
    p.add_argument("--re-weight", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--data-dir", default=None, help="Path to SciERC json files")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-best-to", default=None,
                   help="Path to save the best-dev checkpoint. "
                        "Saved as {'encoder': state_dict, 'step': int, 'metrics': dict}.")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Stage 2a — BERT KG extraction on SciERC ===")
    print(f"  device: {device}")
    print(f"  model:  {args.model_name}")
    print(f"  bs:     {args.batch_size}")
    print(f"  lr:     {args.lr}")
    print(f"  steps:  {args.max_steps}")

    # ── Data ─────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    data_dir = Path(args.data_dir) if args.data_dir else None
    train_loader, dev_loader, _ = build_dataloaders(
        tokenizer, data_dir=data_dir,
        batch_size=args.batch_size, max_length=args.max_length,
    )
    print(f"  train sentences: {len(train_loader.dataset)}")
    print(f"  dev   sentences: {len(dev_loader.dataset)}")
    print(f"  NUM_BIO_TAGS={NUM_BIO_TAGS}  NUM_RELATIONS={NUM_RELATIONS}")

    # ── Model ────────────────────────────────────────────────────────
    model = BertKGExtractor(args.model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Stage 2-005: linear warmup → linear decay scheduler.
    # If --warmup-steps == 0, the schedule is just linear decay (still helpful).
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )
    print(f"  warmup: {args.warmup_steps} | scheduler: linear decay to 0")

    # ── Train loop ───────────────────────────────────────────────────
    # Stage 2-005: track best dev metric so we can report best-checkpoint
    # alongside final-checkpoint, like every published SciERC baseline does.
    best_metrics = {"triple_f1": -1.0}
    best_step = -1

    model.train()
    step = 0
    train_iter = iter(train_loader)
    t0 = time.time()

    while step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        optimizer.zero_grad()
        loss, ner_loss, re_loss, _ = compute_loss(model, batch, device, re_weight=args.re_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 10 == 0:
            dt = (time.time() - t0) * 1000 / max(step, 1)
            cur_lr = scheduler.get_last_lr()[0]
            print(f"[Step {step:04d}] L_total={loss.item():.4f} | L_ner={ner_loss.item():.4f} | L_re={re_loss.item():.4f} | lr={cur_lr:.2e} | {dt:.1f}ms/step")

        if step > 0 and step % args.eval_every == 0:
            model.eval()
            metrics = evaluate(model, dev_loader, device)
            star = ""
            if metrics["triple_f1"] > best_metrics["triple_f1"]:
                best_metrics = dict(metrics)
                best_step = step
                star = " ★ NEW BEST"
                if args.save_best_to:
                    save_path = Path(args.save_best_to)
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"encoder": model.state_dict(), "step": step,
                                "metrics": metrics}, save_path)
            print(f"[Eval @ step {step}] "
                  f"NER F1={metrics['ner_f1']:.4f} | "
                  f"RE F1={metrics['re_f1']:.4f} | "
                  f"Triple F1={metrics['triple_f1']:.4f} | "
                  f"({metrics['n_gold_triples']} gold triples on {metrics['n_examples']} sentences){star}")
            model.train()

        step += 1

    # Final eval
    model.eval()
    metrics = evaluate(model, dev_loader, device)
    if metrics["triple_f1"] > best_metrics["triple_f1"]:
        best_metrics = dict(metrics)
        best_step = step
        if args.save_best_to:
            save_path = Path(args.save_best_to)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"encoder": model.state_dict(), "step": step,
                        "metrics": metrics}, save_path)
    print(f"\n=== FINAL (step {step}) ===")
    print(f"  NER F1     = {metrics['ner_f1']:.4f}")
    print(f"  RE F1      = {metrics['re_f1']:.4f}")
    print(f"  Triple F1  = {metrics['triple_f1']:.4f}")
    print(f"=== BEST DEV (step {best_step}, by Triple F1) ===")
    print(f"  NER F1     = {best_metrics['ner_f1']:.4f}")
    print(f"  RE F1      = {best_metrics['re_f1']:.4f}")
    print(f"  Triple F1  = {best_metrics['triple_f1']:.4f}")
    print(f"  Total time = {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
