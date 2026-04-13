"""
Multi-dataset training with span-based NER head.

The span NER head classifies all candidate (start, end) word spans as
entity_type or NONE. This replaces the BIO token-classification approach
(stage2-001 through stage2-023) with a span enumeration approach similar
to SpERT (Eberts & Ulges, 2020).

The RE head is unchanged — it still takes pairs of predicted entity
spans and classifies relations.

Usage:
    uv run python train_span.py --dataset scierc --max-steps 1500
    uv run python train_span.py --dataset conll04 --max-steps 1500
"""
import argparse
import importlib
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from models.bert_kg_encoder import BertKGExtractor


DATASET_REGISTRY = {
    "scierc": "data.scierc",
    "conll04": "data.conll04",
    "ade": "data.ade",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="scierc", choices=list(DATASET_REGISTRY.keys()))
    p.add_argument("--model-name", default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--warmup-steps", type=int, default=250)
    p.add_argument("--max-span-width", type=int, default=8)
    p.add_argument("--re-weight", type=float, default=1.0)
    p.add_argument("--neg-sample-ratio", type=float, default=0.5,
                   help="Ratio of negative spans to positive spans for NER training. "
                        "0.5 = half as many negatives as positives.")
    p.add_argument("--focal-gamma", type=float, default=2.0,
                   help="Focal loss gamma. 0 = standard CE. 2.0 = recommended for imbalance.")
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-best-to", default=None)
    p.add_argument("--synth-jsonl", default="",
                   help="Path to CAST pseudo-label jsonl. Empty = gold only.")
    p.add_argument("--synth-weight", type=float, default=0.3)
    p.add_argument("--gold-only-steps", type=int, default=500,
                   help="Train on gold-only for this many steps before mixing synth.")
    return p.parse_args()


def focal_loss(logits, targets, gamma=2.0):
    """Focal loss for class-imbalanced classification."""
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def _build_span_labels(gold_entities, num_words, max_span_width, entity_type2id):
    """
    Build a dict: (start, end_inclusive) -> entity_type_id (1-indexed).
    Spans not in gold get label 0 (NONE).
    """
    gold_span_labels = {}
    for (s, e, etype) in gold_entities:
        if e - s + 1 <= max_span_width:
            etype_id = entity_type2id.get(etype, 0)
            if etype_id > 0:
                gold_span_labels[(s, e)] = etype_id
    return gold_span_labels


def compute_span_loss(model, batch, device, ds_mod, entity_type2id,
                      re_weight=1.0, neg_sample_ratio=0.5, max_span_width=8,
                      focal_gamma=2.0):
    """Compute span NER loss + RE loss."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    word_ids_list = batch["word_ids"]
    gold_entities_list = batch["gold_entities"]
    gold_relations_list = batch["gold_relations"]
    num_words_list = batch["num_words"]

    hidden = model.encode(modality="text", input_ids=input_ids, attention_mask=attention_mask)

    span_losses = []
    re_losses = []
    NO_REL = ds_mod.NO_REL_ID

    for b_idx in range(input_ids.size(0)):
        n_words = num_words_list[b_idx]
        gold_ents = gold_entities_list[b_idx]
        gold_rels = gold_relations_list[b_idx]

        # Span NER
        span_logits, candidates = model.forward_span_ner(
            hidden[b_idx], word_ids_list[b_idx], n_words, max_span_width,
        )
        if not candidates:
            continue

        gold_labels = _build_span_labels(gold_ents, n_words, max_span_width, entity_type2id)

        # Build target tensor
        targets = []
        for (s, e) in candidates:
            targets.append(gold_labels.get((s, e), 0))
        targets = torch.tensor(targets, device=device, dtype=torch.long)

        # Negative sampling: keep all positives + sample negatives
        pos_mask = targets > 0
        neg_mask = targets == 0
        n_pos = pos_mask.sum().item()
        n_neg_keep = max(int(n_pos * neg_sample_ratio), 1)
        neg_indices = neg_mask.nonzero(as_tuple=True)[0]
        if len(neg_indices) > n_neg_keep:
            perm = torch.randperm(len(neg_indices), device=device)[:n_neg_keep]
            neg_indices = neg_indices[perm]
        keep_indices = torch.cat([pos_mask.nonzero(as_tuple=True)[0], neg_indices])

        if len(keep_indices) > 0:
            span_loss = focal_loss(span_logits[keep_indices], targets[keep_indices], gamma=focal_gamma)
            span_losses.append(span_loss)

        # RE loss — same as train_multi.py but using gold entity spans
        if len(gold_ents) >= 2:
            rel_lookup = {(h, t): rid for (h, t, rid) in gold_rels}
            spans = [(s, e) for (s, e, _) in gold_ents]
            pairs = [(h, t) for h in spans for t in spans if h != t]
            if pairs:
                pair_targets = [rel_lookup.get((h, t), NO_REL) for (h, t) in pairs]
                pair_targets_t = torch.tensor(pair_targets, device=device, dtype=torch.long)
                re_logits = model.forward_re(hidden[b_idx], word_ids_list[b_idx], pairs)
                re_losses.append(F.cross_entropy(re_logits, pair_targets_t))

    ner_loss = torch.stack(span_losses).mean() if span_losses else hidden.new_tensor(0.0)
    re_loss = torch.stack(re_losses).mean() if re_losses else hidden.new_tensor(0.0)
    total = ner_loss + re_weight * re_loss
    return total, ner_loss.detach(), re_loss.detach()


def evaluate_span(model, dataloader, device, ds_mod, entity_type2id, id2entity_type,
                  max_span_width=8, span_threshold=0.5):
    """Evaluate with span-based NER predictions feeding into RE."""
    from eval.triple_f1 import _prf
    model.eval()
    NO_REL = ds_mod.NO_REL_ID

    ner_tp = ner_fp = ner_fn = 0
    triple_tp = triple_fp = triple_fn = 0
    n_examples = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        word_ids_list = batch["word_ids"]
        gold_entities_list = batch["gold_entities"]
        gold_relations_list = batch["gold_relations"]
        num_words_list = batch["num_words"]

        hidden = model.encode(modality="text", input_ids=input_ids, attention_mask=attention_mask)

        for b_idx in range(input_ids.size(0)):
            n_examples += 1
            n_words = num_words_list[b_idx]
            gold_ents = gold_entities_list[b_idx]
            gold_rels = gold_relations_list[b_idx]

            with torch.no_grad():
                span_logits, candidates = model.forward_span_ner(
                    hidden[b_idx], word_ids_list[b_idx], n_words, max_span_width,
                )

            if not candidates:
                ner_fn += len(gold_ents)
                continue

            # Predict spans: take argmax, keep those != NONE (0)
            span_probs = torch.softmax(span_logits, dim=-1)
            pred_types = span_logits.argmax(dim=-1).tolist()
            pred_confs = span_probs.max(dim=-1).values.tolist()

            pred_spans = []
            for (s, e), etype_id, conf in zip(candidates, pred_types, pred_confs):
                if etype_id > 0 and conf >= span_threshold:
                    etype = id2entity_type.get(etype_id, "Unknown")
                    pred_spans.append((s, e, etype))

            # Remove overlapping spans: keep highest confidence
            # (greedy non-overlapping: sort by confidence, skip overlaps)
            scored = sorted(
                [(s, e, t, pred_confs[candidates.index((s, e))]) for (s, e, t) in pred_spans],
                key=lambda x: -x[3],
            )
            taken = set()
            filtered = []
            for (s, e, t, c) in scored:
                overlap = any(
                    not (e < ts or te < s)
                    for (ts, te) in taken
                )
                if not overlap:
                    filtered.append((s, e, t))
                    taken.add((s, e))
            pred_spans = filtered

            # NER F1
            pred_ent_set = {(s, e, t) for (s, e, t) in pred_spans}
            gold_ent_set = {(s, e, t) for (s, e, t) in gold_ents}
            ner_tp += len(pred_ent_set & gold_ent_set)
            ner_fp += len(pred_ent_set - gold_ent_set)
            ner_fn += len(gold_ent_set - pred_ent_set)

            # Triple F1 (full pipeline)
            pred_span_list = [(s, e) for (s, e, _) in pred_spans]
            pred_pairs = [(a, b) for a in pred_span_list for b in pred_span_list if a != b]
            if pred_pairs:
                with torch.no_grad():
                    pred_re_logits = model.forward_re(hidden[b_idx], word_ids_list[b_idx], pred_pairs)
                pred_re_ids = pred_re_logits.argmax(dim=-1).tolist()
                pred_full = {(h, t, p) for (h, t), p in zip(pred_pairs, pred_re_ids) if p != NO_REL}
            else:
                pred_full = set()
            gold_full = {(h, t, r) for (h, t, r) in gold_rels}
            triple_tp += len(pred_full & gold_full)
            triple_fp += len(pred_full - gold_full)
            triple_fn += len(gold_full - pred_full)

    _, _, nf = _prf(ner_tp, ner_fp, ner_fn)
    _, _, tf = _prf(triple_tp, triple_fp, triple_fn)
    return {"ner_f1": nf, "triple_f1": tf, "n_examples": n_examples}


def cycle(loader):
    while True:
        yield from loader


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    ds_mod = importlib.import_module(DATASET_REGISTRY[args.dataset])
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_name is None:
        args.model_name = {
            "scierc": "allenai/scibert_scivocab_uncased",
            "conll04": "bert-base-uncased",
            "ade": "allenai/scibert_scivocab_uncased",
        }.get(args.dataset, "bert-base-uncased")

    print(f"=== Span-based NER training ({args.dataset}) ===")
    print(f"  encoder:        {args.model_name}")
    print(f"  max_span_width: {args.max_span_width}")
    print(f"  neg_sample:     {args.neg_sample_ratio}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_loader, dev_loader, test_loader = ds_mod.build_dataloaders(
        tokenizer, batch_size=args.batch_size, max_length=args.max_length,
    )
    print(f"  train: {len(train_loader.dataset)} | dev: {len(dev_loader.dataset)}")

    # Entity type mapping: type_name -> id (1-indexed, 0 = NONE)
    entity_types = ds_mod.ENTITY_TYPES
    entity_type2id = {t: i + 1 for i, t in enumerate(entity_types)}
    id2entity_type = {i + 1: t for i, t in enumerate(entity_types)}
    num_entity_types = len(entity_types)
    print(f"  entity types:   {entity_types} ({num_entity_types})")

    # Patch scierc dicts only if running a different dataset
    import data.scierc as scierc_mod
    if args.dataset != "scierc":
        scierc_mod.ID2BIO.clear()
        scierc_mod.ID2BIO.update(ds_mod.ID2BIO)
        scierc_mod.BIO_TAG2ID.clear()
        scierc_mod.BIO_TAG2ID.update(ds_mod.BIO_TAG2ID)
        scierc_mod.NO_REL_ID = ds_mod.NO_REL_ID

    model = BertKGExtractor(
        args.model_name,
        num_bio_tags=ds_mod.NUM_BIO_TAGS,
        num_relations=ds_mod.NUM_RELATIONS,
        num_entity_types=num_entity_types,
        use_span_ner=True,
        max_span_width=args.max_span_width,
    ).to(device)
    print(f"  span_ner_head:  {model.span_ner_head}")
    print(f"  re_head out:    {model.re_head[-1].out_features}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps,
    )

    use_synth = bool(args.synth_jsonl)
    synth_loader = None
    if use_synth:
        from data.synth_loader import build_synth_loader
        synth_loader = build_synth_loader(
            tokenizer, args.synth_jsonl,
            batch_size=args.batch_size, max_length=args.max_length,
        )
        print(f"  synth: {len(synth_loader.dataset)}")

    gold_iter = cycle(train_loader)
    synth_iter = cycle(synth_loader) if synth_loader else None
    best_metrics = {"triple_f1": -1.0}
    best_step = -1

    model.train()
    t0 = time.time()
    step = 0
    while step < args.max_steps:
        optimizer.zero_grad()
        batch = next(gold_iter)
        gold_loss, ner_loss, re_loss = compute_span_loss(
            model, batch, device, ds_mod, entity_type2id,
            re_weight=args.re_weight, neg_sample_ratio=args.neg_sample_ratio,
            max_span_width=args.max_span_width, focal_gamma=args.focal_gamma,
        )

        synth_loss_val = 0.0
        if use_synth and step >= args.gold_only_steps and synth_iter:
            synth_batch = next(synth_iter)
            try:
                s_loss, _, _ = compute_span_loss(
                    model, synth_batch, device, ds_mod, entity_type2id,
                    re_weight=args.re_weight, neg_sample_ratio=args.neg_sample_ratio,
                    max_span_width=args.max_span_width, focal_gamma=args.focal_gamma,
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

        if step % 10 == 0:
            dt = (time.time() - t0) * 1000 / max(step, 1)
            cur_lr = scheduler.get_last_lr()[0]
            print(f"[Step {step:04d}] L={gold_loss.item():.4f} NER={ner_loss.item():.4f} "
                  f"RE={re_loss.item():.4f} synth={synth_loss_val:.4f} lr={cur_lr:.2e} | {dt:.0f}ms/step")

        if step > 0 and step % args.eval_every == 0:
            metrics = evaluate_span(
                model, dev_loader, device, ds_mod,
                entity_type2id, id2entity_type,
                max_span_width=args.max_span_width,
            )
            star = ""
            if metrics["triple_f1"] > best_metrics["triple_f1"]:
                best_metrics = dict(metrics)
                best_step = step
                star = " *"
                if args.save_best_to:
                    save_path = Path(args.save_best_to)
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"encoder": model.state_dict(), "step": step,
                                "metrics": metrics}, save_path)
            print(f"[Eval @ {step}] NER={metrics['ner_f1']:.4f} "
                  f"Triple={metrics['triple_f1']:.4f}{star}")
            model.train()

        step += 1

    for split_name, loader in [("dev", dev_loader), ("test", test_loader)]:
        metrics = evaluate_span(
            model, loader, device, ds_mod,
            entity_type2id, id2entity_type,
            max_span_width=args.max_span_width,
        )
        if split_name == "dev" and metrics["triple_f1"] > best_metrics["triple_f1"]:
            best_metrics = dict(metrics)
            best_step = step
        print(f"\n=== {split_name.upper()} (step {step}) ===")
        print(f"  NER={metrics['ner_f1']:.4f} Triple={metrics['triple_f1']:.4f}")
    print(f"=== BEST DEV (step {best_step}) ===")
    print(f"  NER={best_metrics['ner_f1']:.4f} Triple={best_metrics['triple_f1']:.4f}")
    print(f"  time={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
