"""
Multi-dataset CAST pseudo-label generator.

In-domain self-training: teacher predicts on its OWN train data,
filters by per-class confidence thresholds, produces pseudo-labeled
jsonl for student training.

Why in-domain: CoNLL04 (newswire) and ADE (biomedical) don't have
domain-matched unlabeled pools like SciERC has arXiv. Instead we
use the gold train sentences themselves — the teacher's confident
predictions ADD new entity/relation annotations that the gold labels
missed (gold annotations are often incomplete).

Usage:
    uv run python generate_pseudo_labels_multi.py \
        --dataset conll04 \
        --teacher-ckpt checkpoints/conll04_baseline_best.pt \
        --out-jsonl data/conll04_pseudo_cast.jsonl
"""
import argparse
import importlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from eval.triple_f1 import _bio_to_spans, _word_level_bio_from_token_logits


DATASET_REGISTRY = {
    "scierc": "data.scierc",
    "conll04": "data.conll04",
    "ade": "data.ade",
}

# Per-dataset CAST thresholds: {rel_type: tau_re}
# Overrepresented relations get stricter thresholds (CAST principle)
CAST_THRESHOLDS = {
    "scierc": {
        "USED-FOR": 0.82,      # 72.7% of gold → strict
        "CONJUNCTION": 0.70,
        "FEATURE-OF": 0.50,    # rare → relaxed
        "PART-OF": 0.50,
        "HYPONYM-OF": 0.50,
        "EVALUATE-FOR": 0.50,
        "COMPARE": 0.50,
    },
    "conll04": {
        "Work_For": 0.80,      # most frequent
        "Located_In": 0.75,
        "OrgBased_In": 0.60,
        "Live_In": 0.50,       # rare
        "Kill": 0.50,          # rare
    },
    "ade": {
        "Adverse-Effect": 0.70,  # only 1 relation type
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(DATASET_REGISTRY.keys()))
    p.add_argument("--model-name", default=None)
    p.add_argument("--teacher-ckpt", required=True)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--tau-ner", type=float, default=0.7)
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds_mod = importlib.import_module(DATASET_REGISTRY[args.dataset])

    if args.model_name is None:
        args.model_name = {
            "scierc": "allenai/scibert_scivocab_uncased",
            "conll04": "bert-base-uncased",
            "ade": "allenai/scibert_scivocab_uncased",
        }[args.dataset]

    print(f"=== CAST pseudo-label generation ({args.dataset}) ===")
    print(f"  teacher: {args.teacher_ckpt}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    data_dir = None
    train_loader, _, _ = ds_mod.build_dataloaders(
        tokenizer, data_dir=data_dir, batch_size=1, max_length=args.max_length,
    )
    print(f"  train sentences: {len(train_loader.dataset)}")

    # Load teacher with correct head dimensions
    import data.scierc as scierc_mod
    scierc_mod.ID2BIO.clear()
    scierc_mod.ID2BIO.update(ds_mod.ID2BIO)
    scierc_mod.BIO_TAG2ID.clear()
    scierc_mod.BIO_TAG2ID.update(ds_mod.BIO_TAG2ID)
    scierc_mod.NO_REL_ID = ds_mod.NO_REL_ID
    from models.bert_kg_encoder import BertKGExtractor
    model = BertKGExtractor(
        args.model_name,
        num_bio_tags=ds_mod.NUM_BIO_TAGS,
        num_relations=ds_mod.NUM_RELATIONS,
    ).to(device)

    ckpt = torch.load(args.teacher_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["encoder"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"  teacher loaded: step={ckpt['step']} triple_f1={ckpt['metrics']['triple_f1']:.4f}")

    # Monkey-patch eval ID2BIO
    import eval.triple_f1 as eval_mod
    orig_id2bio = eval_mod.ID2BIO
    eval_mod.ID2BIO = ds_mod.ID2BIO

    cast_thresholds = CAST_THRESHOLDS.get(args.dataset, {})
    default_tau = 0.70

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_sentences = 0
    n_relations = 0
    rel_counts = {}

    with out_path.open("w") as fout:
        for batch in train_loader:
            words = batch["words"][0]
            n_sentences += 1

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            word_ids = batch["word_ids"][0]

            with torch.no_grad():
                hidden = model.encode(modality="text", input_ids=input_ids, attention_mask=attention_mask)
                ner_logits = model.forward_ner(hidden)

            # Extract entities with confidence
            logits_0 = ner_logits[0]
            probs = torch.softmax(logits_0, dim=-1)
            word_bio = _word_level_bio_from_token_logits(logits_0, word_ids)
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

            confident_spans = []
            for (s, e, etype) in spans:
                span_confs = [word_conf.get(w, 0.0) for w in range(s, e + 1)]
                if span_confs and min(span_confs) >= args.tau_ner:
                    confident_spans.append((s, e, etype))

            if len(confident_spans) < 2:
                continue

            # Predict relations on confident span pairs
            span_list = [(s, e) for (s, e, _) in confident_spans]
            type_by_span = {(s, e): t for (s, e, t) in confident_spans}
            pairs = [(h, t) for h in span_list for t in span_list if h != t]
            if not pairs:
                continue

            with torch.no_grad():
                re_logits = model.forward_re(hidden[0], word_ids, pairs)
            re_probs = torch.softmax(re_logits, dim=-1)
            pred_rels = re_logits.argmax(dim=-1).tolist()
            pred_confs = re_probs.max(dim=-1).values.tolist()

            for (h_span, t_span), rel_id, conf in zip(pairs, pred_rels, pred_confs):
                if rel_id == ds_mod.NO_REL_ID:
                    continue
                rel_name = ds_mod.ID2REL[rel_id]
                tau = cast_thresholds.get(rel_name, default_tau)
                if conf < tau:
                    continue

                hs, he = h_span
                ts, te = t_span
                head_phrase = " ".join(words[hs:he + 1])
                tail_phrase = " ".join(words[ts:te + 1])
                h_type = type_by_span.get(h_span, "Unknown")
                t_type = type_by_span.get(t_span, "Unknown")

                record = {
                    "synth_sentence": " ".join(words),
                    "source_sentence": " ".join(words),
                    "head": head_phrase,
                    "rel": rel_name,
                    "tail": tail_phrase,
                    "rel_id": int(rel_id),
                    "entity_type": h_type,
                    "tail_entity_type": t_type,
                    "containment": 1.0,
                    "pseudo_label": True,
                    "re_confidence": float(conf),
                }
                fout.write(json.dumps(record) + "\n")
                n_relations += 1
                rel_counts[rel_name] = rel_counts.get(rel_name, 0) + 1

            if n_sentences % 200 == 0:
                print(f"  [{n_sentences}/{len(train_loader.dataset)}] relations={n_relations}")

    eval_mod.ID2BIO = orig_id2bio

    print(f"\n=== DONE ===")
    print(f"  sentences: {n_sentences}")
    print(f"  relations: {n_relations}")
    print(f"  per class: {rel_counts}")
    print(f"  wrote: {out_path}")


if __name__ == "__main__":
    main()
