"""
Stage 2-013: Multi-round self-training (CSJE-inspired).

CSJE (KBS 2025) shows +1-5% F1 with iterative pseudo-labeling where each
round's student becomes the next round's teacher. Our one-round attempt
(stage2-012) failed at -0.015 because the teacher (Triple F1 0.353) generated
too many false positives at τ_ner=0.7, τ_re=0.5.

Strategy:
  Round 0: Gold-only baseline (existing stage2_007_best.pt, Triple F1 0.3573).
  Round 1: Teacher=round0, τ_ner=0.90, τ_re=0.70 → very strict pseudo-labels.
           Train gold + pseudo (synth_weight=0.1). Save best checkpoint.
  Round 2: Teacher=round1, τ_ner=0.85, τ_re=0.60 → slightly relaxed.
           Train gold + pseudo. Save best.
  Round 3: Teacher=round2, τ_ner=0.80, τ_re=0.55 → further relaxed.
           Train gold + pseudo. Save best.

Key vs. one-round:
  - Much stricter initial thresholds (0.9/0.7 vs 0.7/0.5)
  - Progressive relaxation as teacher quality improves
  - Each round starts from fresh SciBERT (no catastrophic forgetting)
  - Better text cleaning (skip arXiv header noise)

Usage:
    uv run python train_stage2_multiround.py --n-rounds 3 --max-steps 1500
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from data.scierc import build_dataloaders, NUM_BIO_TAGS, NUM_RELATIONS, NO_REL_ID, ID2REL
from models.bert_kg_encoder import BertKGExtractor, compute_loss
from eval.triple_f1 import evaluate, _bio_to_spans, _word_level_bio_from_token_logits


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="allenai/scibert_scivocab_uncased")
    p.add_argument("--teacher-ckpt", default="checkpoints/stage2_007_best.pt",
                   help="Round 0 teacher checkpoint (gold-only baseline).")
    p.add_argument("--n-rounds", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--warmup-steps", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--re-weight", type=float, default=1.0)
    # Threshold schedule: start strict, relax each round
    p.add_argument("--tau-ner-start", type=float, default=0.90)
    p.add_argument("--tau-ner-end", type=float, default=0.80)
    p.add_argument("--tau-re-start", type=float, default=0.70)
    p.add_argument("--tau-re-end", type=float, default=0.55)
    p.add_argument("--synth-weight", type=float, default=0.1,
                   help="Weight on pseudo-label loss (gold-dominant).")
    p.add_argument("--gold-only-steps", type=int, default=300,
                   help="Train gold-only before mixing pseudo-labels.")
    p.add_argument("--arxiv-jsonl", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ─── Pseudo-label generation (in-process, no separate script) ──────────

def _clean_arxiv_text(text):
    """Strip arXiv headers, emails, URLs from abstract text."""
    clean = text.replace("\n", " ")
    # Find "Abstract" marker and skip everything before it
    abs_idx = clean.lower().find("abstract")
    if abs_idx >= 0:
        clean = clean[abs_idx + len("abstract"):].strip()
        clean = clean.lstrip("0123456789 .")
    # Remove arXiv IDs, emails, URLs, author-like lines
    clean = re.sub(r'arXiv:\S+', '', clean)
    clean = re.sub(r'\S+@\S+\.\S+', '', clean)
    clean = re.sub(r'https?://\S+', '', clean)
    clean = re.sub(r'\b[A-Z][A-Z\s,\.]+\d+\b', '', clean)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean


def _is_clean_sentence(sent):
    """Filter out sentences that still have arXiv noise."""
    words = sent.split()
    if len(words) < 5 or len(words) > 80:
        return False
    # Skip sentences with arXiv IDs, version markers, emails
    if re.search(r'arXiv:', sent, re.IGNORECASE):
        return False
    if re.search(r'\bv\d+\b.*\[cs\.', sent):
        return False
    if re.search(r'\S+@\S+\.\S+', sent):
        return False
    # Skip sentences that are mostly numbers/punctuation
    alpha_ratio = sum(c.isalpha() for c in sent) / max(len(sent), 1)
    if alpha_ratio < 0.5:
        return False
    return True


def generate_pseudo_labels(model, tokenizer, arxiv_jsonl, tau_ner, tau_re,
                           max_length, device, out_jsonl):
    """Generate pseudo-labels from arXiv using the teacher model."""
    model.eval()

    # Load arXiv text
    if arxiv_jsonl is None:
        arxiv_jsonl = str(Path(__file__).parent / "data" / "arxiv_real" / "cs_validation.jsonl")
    raw_texts = []
    with open(arxiv_jsonl) as f:
        for line in f:
            rec = json.loads(line)
            raw_texts.append(rec.get("text", ""))
    print(f"  arXiv docs: {len(raw_texts)}")

    out_path = Path(out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_sentences = 0
    n_with_entities = 0
    n_relations = 0

    with out_path.open("w") as fout:
        for doc_idx, text in enumerate(raw_texts):
            clean = _clean_arxiv_text(text)
            sentences = [s.strip() + "." for s in clean.split(". ")
                         if _is_clean_sentence(s.strip())]

            for sent in sentences:
                words = sent.split()
                n_sentences += 1

                # Tokenize
                enc = tokenizer(
                    words, is_split_into_words=True,
                    return_tensors="pt", padding=True,
                    truncation=True, max_length=max_length,
                )
                word_ids = enc.word_ids(batch_index=0)
                enc = {k: v.to(device) for k, v in enc.items()}

                with torch.no_grad():
                    hidden = model.encode(
                        modality="text",
                        input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"],
                    )
                    ner_logits = model.forward_ner(hidden)

                # Extract confident entities
                logits = ner_logits[0]
                probs = torch.softmax(logits, dim=-1)
                word_bio = _word_level_bio_from_token_logits(logits, word_ids)
                spans = _bio_to_spans(word_bio)

                # Per-word confidence
                word_conf = {}
                seen = set()
                for tok_i, wid in enumerate(word_ids):
                    if wid is None or wid in seen:
                        continue
                    if wid < len(word_bio):
                        word_conf[wid] = probs[tok_i, word_bio[wid]].item()
                    seen.add(wid)

                entities = []
                for (s, e, etype) in spans:
                    span_confs = [word_conf.get(w, 0.0) for w in range(s, e + 1)]
                    if span_confs and min(span_confs) >= tau_ner:
                        entities.append((s, e, etype, min(span_confs)))

                if len(entities) < 2:
                    continue
                n_with_entities += 1

                # Extract confident relations
                span_pairs = [((h[0], h[1]), (t[0], t[1]))
                              for h in entities for t in entities if h != t]
                if not span_pairs:
                    continue

                pairs_for_re = [(h, t) for h, t in span_pairs]
                with torch.no_grad():
                    re_logits = model.forward_re(hidden[0], word_ids, pairs_for_re)
                re_probs = torch.softmax(re_logits, dim=-1)
                pred_rels = re_logits.argmax(dim=-1).tolist()
                pred_confs = re_probs.max(dim=-1).values.tolist()

                type_by_span = {(s, e): t for (s, e, t, _) in entities}
                for (h_span, t_span), rel_id, conf in zip(pairs_for_re, pred_rels, pred_confs):
                    if rel_id == NO_REL_ID or conf < tau_re:
                        continue
                    hs, he = h_span
                    ts, te = t_span
                    record = {
                        "synth_sentence": sent,
                        "source_sentence": sent,
                        "head": " ".join(words[hs:he + 1]),
                        "rel": ID2REL[rel_id],
                        "tail": " ".join(words[ts:te + 1]),
                        "rel_id": int(rel_id),
                        "entity_type": type_by_span.get(h_span, "Method"),
                        "tail_entity_type": type_by_span.get(t_span, "Method"),
                        "containment": 1.0,
                        "pseudo_label": True,
                        "ner_confidence": float(min(
                            c for (s, e, t, c) in entities
                            if (s, e) in (h_span, t_span)
                        )) if any((s, e) in (h_span, t_span) for (s, e, t, c) in entities) else 0.0,
                        "re_confidence": float(conf),
                    }
                    fout.write(json.dumps(record) + "\n")
                    n_relations += 1

    print(f"  pseudo-labels: {n_sentences} sents → {n_with_entities} with entities → {n_relations} relations")
    return n_relations


# ─── Training loop (adapted from train_stage2e.py) ────────────────────

def cycle(loader):
    while True:
        yield from loader


def train_one_round(model_name, synth_jsonl, max_steps, warmup_steps,
                    batch_size, lr, max_length, eval_every, re_weight,
                    synth_weight, gold_only_steps, data_dir, device, seed,
                    save_path):
    """Train one round of self-training. Returns best metrics dict."""
    import random
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    torch.manual_seed(seed)
    random.seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    _data_dir = Path(data_dir) if data_dir else None
    train_loader, dev_loader, _ = build_dataloaders(
        tokenizer, data_dir=_data_dir,
        batch_size=batch_size, max_length=max_length,
    )

    use_synth = bool(synth_jsonl) and Path(synth_jsonl).exists()
    synth_loader = None
    if use_synth:
        from data.synth_loader import build_synth_loader
        synth_loader = build_synth_loader(
            tokenizer, synth_jsonl,
            batch_size=batch_size, max_length=max_length,
            min_containment=0.0,  # pseudo-labels always have containment=1.0
        )
        print(f"  synth sentences: {len(synth_loader.dataset)}")

    model = BertKGExtractor(model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps,
    )

    gold_iter = cycle(train_loader)
    synth_iter = cycle(synth_loader) if synth_loader else None
    best_metrics = {"triple_f1": -1.0}
    best_step = -1

    model.train()
    t0 = time.time()
    for step in range(max_steps):
        optimizer.zero_grad()

        gold_batch = next(gold_iter)
        gold_loss, ner_loss, re_loss, _ = compute_loss(
            model, gold_batch, device, re_weight=re_weight,
        )

        synth_loss_val = 0.0
        if use_synth and step >= gold_only_steps and synth_iter:
            synth_batch = next(synth_iter)
            try:
                s_loss, _, _, _ = compute_loss(
                    model, synth_batch, device, re_weight=re_weight,
                )
                synth_loss_val = s_loss.item()
                total = gold_loss + synth_weight * s_loss
            except Exception:
                total = gold_loss
        else:
            total = gold_loss

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 50 == 0:
            phase = "gold+pseudo" if (use_synth and step >= gold_only_steps) else "gold"
            print(f"  [Step {step:04d} {phase}] L_gold={gold_loss.item():.4f} "
                  f"L_synth={synth_loss_val:.4f} L_ner={ner_loss.item():.4f} "
                  f"L_re={re_loss.item():.4f}")

        if step > 0 and step % eval_every == 0:
            model.eval()
            metrics = evaluate(model, dev_loader, device)
            star = ""
            if metrics["triple_f1"] > best_metrics["triple_f1"]:
                best_metrics = dict(metrics)
                best_step = step
                star = " ★"
                if save_path:
                    sp = Path(save_path)
                    sp.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "encoder": model.state_dict(),
                        "step": step,
                        "metrics": metrics,
                    }, sp)
            print(f"  [Eval {step}] NER={metrics['ner_f1']:.4f} RE={metrics['re_f1']:.4f} "
                  f"Triple={metrics['triple_f1']:.4f}{star}")
            model.train()

    # Final eval
    model.eval()
    metrics = evaluate(model, dev_loader, device)
    if metrics["triple_f1"] > best_metrics["triple_f1"]:
        best_metrics = dict(metrics)
        best_step = max_steps
        if save_path:
            torch.save({
                "encoder": model.state_dict(),
                "step": max_steps,
                "metrics": metrics,
            }, save_path)

    print(f"  BEST @ step {best_step}: NER={best_metrics['ner_f1']:.4f} "
          f"RE={best_metrics['re_f1']:.4f} Triple={best_metrics['triple_f1']:.4f}")
    print(f"  Time: {time.time() - t0:.1f}s")

    return best_metrics, save_path


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{'='*60}")
    print(f"Stage 2-013: Multi-round self-training (CSJE-inspired)")
    print(f"  rounds:     {args.n_rounds}")
    print(f"  steps/round: {args.max_steps}")
    print(f"  τ_ner:      {args.tau_ner_start} → {args.tau_ner_end}")
    print(f"  τ_re:       {args.tau_re_start} → {args.tau_re_end}")
    print(f"  synth_w:    {args.synth_weight}")
    print(f"  teacher:    {args.teacher_ckpt}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    teacher_ckpt = args.teacher_ckpt
    all_results = []

    # ── Gold-only baseline (round 0) ──────────────────────────
    print(f"\n{'─'*40}")
    print(f"Round 0: Gold-only baseline")
    print(f"{'─'*40}")
    r0_save = "checkpoints/multiround_r0_best.pt"
    r0_metrics, _ = train_one_round(
        model_name=args.model_name,
        synth_jsonl="",
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        eval_every=args.eval_every,
        re_weight=args.re_weight,
        synth_weight=0.0,
        gold_only_steps=0,
        data_dir=args.data_dir,
        device=device,
        seed=args.seed,
        save_path=r0_save,
    )
    all_results.append(("round0_gold", r0_metrics))
    # Use round 0 as first teacher (or provided checkpoint if it's better)
    teacher_ckpt_for_round = r0_save

    for round_idx in range(1, args.n_rounds + 1):
        # Interpolate thresholds
        t = (round_idx - 1) / max(args.n_rounds - 1, 1)
        tau_ner = args.tau_ner_start + t * (args.tau_ner_end - args.tau_ner_start)
        tau_re = args.tau_re_start + t * (args.tau_re_end - args.tau_re_start)

        print(f"\n{'─'*40}")
        print(f"Round {round_idx}/{args.n_rounds}: τ_ner={tau_ner:.2f}, τ_re={tau_re:.2f}")
        print(f"  teacher: {teacher_ckpt_for_round}")
        print(f"{'─'*40}")

        # ── Step A: Generate pseudo-labels ────────────────────
        teacher = BertKGExtractor(args.model_name).to(device)
        ckpt = torch.load(teacher_ckpt_for_round, map_location="cpu")
        teacher.load_state_dict(ckpt["encoder"])
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        teacher_f1 = ckpt.get("metrics", {}).get("triple_f1", 0.0)
        print(f"  teacher Triple F1: {teacher_f1:.4f}")

        pseudo_jsonl = f"data/multiround_pseudo_r{round_idx}.jsonl"
        n_rels = generate_pseudo_labels(
            teacher, tokenizer, args.arxiv_jsonl,
            tau_ner=tau_ner, tau_re=tau_re,
            max_length=args.max_length, device=device,
            out_jsonl=pseudo_jsonl,
        )
        del teacher
        torch.cuda.empty_cache() if device == "cuda" else None

        if n_rels == 0:
            print(f"  ⚠ No pseudo-labels generated! Thresholds too strict.")
            print(f"  Skipping round {round_idx}.")
            all_results.append((f"round{round_idx}", {"triple_f1": -1, "note": "no pseudo-labels"}))
            continue

        # ── Step B: Train student ─────────────────────────────
        save_path = f"checkpoints/multiround_r{round_idx}_best.pt"
        metrics, _ = train_one_round(
            model_name=args.model_name,
            synth_jsonl=pseudo_jsonl,
            max_steps=args.max_steps,
            warmup_steps=args.warmup_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            max_length=args.max_length,
            eval_every=args.eval_every,
            re_weight=args.re_weight,
            synth_weight=args.synth_weight,
            gold_only_steps=args.gold_only_steps,
            data_dir=args.data_dir,
            device=device,
            seed=args.seed + round_idx,  # different seed per round
            save_path=save_path,
        )
        all_results.append((f"round{round_idx}", metrics))

        # Student becomes next teacher
        teacher_ckpt_for_round = save_path

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"MULTI-ROUND SELF-TRAINING SUMMARY")
    print(f"{'='*60}")
    for name, m in all_results:
        if "triple_f1" in m and m["triple_f1"] >= 0:
            print(f"  {name:15s}: NER={m.get('ner_f1', 0):.4f} "
                  f"RE={m.get('re_f1', 0):.4f} Triple={m['triple_f1']:.4f}")
        else:
            print(f"  {name:15s}: SKIPPED ({m.get('note', 'unknown')})")

    # Compare to known baseline
    baseline_f1 = 0.3573
    best_round = max(all_results, key=lambda x: x[1].get("triple_f1", -1))
    best_f1 = best_round[1].get("triple_f1", 0)
    delta = best_f1 - baseline_f1
    print(f"\n  Baseline:   {baseline_f1:.4f}")
    print(f"  Best round: {best_round[0]} = {best_f1:.4f} (Δ = {delta:+.4f})")
    if delta > 0:
        print(f"  ✅ NEW BEST — multi-round self-training improved over baseline!")
    else:
        print(f"  ❌ No improvement over baseline.")


if __name__ == "__main__":
    main()
