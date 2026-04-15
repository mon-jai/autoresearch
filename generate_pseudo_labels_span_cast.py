"""
Stage 2-032: CAST pseudo-label generation using a SPAN NER teacher.

Key difference from generate_pseudo_labels_cast.py (stage2-016):
- Uses forward_span_ner() instead of forward_ner() + BIO conversion.
- Entities come directly as (start, end, entity_type) spans with
  per-span confidence, avoiding the BIO→span conversion noise that
  made stage2-031 (BIO teacher + span student) neutral.
- Greedy non-overlapping span filtering (highest confidence first).

CAST adaptive thresholds are applied to RE predictions (unchanged).
"""
import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer

from data.scierc import ENTITY_TYPES, ID2REL, NO_REL_ID, NUM_BIO_TAGS, NUM_RELATIONS
from models.bert_kg_encoder import BertKGExtractor


# Gold RE distribution target (SciERC train)
GOLD_SHARE = {
    "USED-FOR": 0.525,
    "CONJUNCTION": 0.124,
    "EVALUATE-FOR": 0.097,
    "HYPONYM-OF": 0.093,
    "PART-OF": 0.056,
    "FEATURE-OF": 0.054,
    "COMPARE": 0.052,
}

# Per-relation tau_re: strict for overrepresented, relaxed for underrepresented
PER_REL_TAU = {
    "USED-FOR": 0.82,
    "CONJUNCTION": 0.65,
    "EVALUATE-FOR": 0.60,
    "HYPONYM-OF": 0.60,
    "PART-OF": 0.50,
    "FEATURE-OF": 0.50,
    "COMPARE": 0.50,
}

# Entity type mapping (1-indexed, 0 = NONE)
ENTITY_TYPE2ID = {t: i + 1 for i, t in enumerate(ENTITY_TYPES)}
ID2ENTITY_TYPE = {i + 1: t for i, t in enumerate(ENTITY_TYPES)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="allenai/scibert_scivocab_uncased")
    p.add_argument("--teacher-ckpt", required=True,
                   help="Path to span teacher checkpoint (from train_span.py --save-best-to)")
    p.add_argument("--arxiv-jsonl", default=None)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--max-span-width", type=int, default=8)
    p.add_argument("--tau-ner", type=float, default=0.80,
                   help="Min span confidence to keep an entity prediction.")
    p.add_argument("--tau-ner-percentile", type=int, default=0,
                   help="If >0, compute tau_ner as this percentile of all span "
                        "confidences (overrides --tau-ner).")
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--device", default=None)
    return p.parse_args()


def _clean_arxiv_text(text):
    clean = text.replace("\n", " ")
    abs_idx = clean.lower().find("abstract")
    if abs_idx >= 0:
        clean = clean[abs_idx + len("abstract"):].strip()
        clean = clean.lstrip("0123456789 .")
    clean = re.sub(r'arXiv:\S+', '', clean)
    clean = re.sub(r'\S+@\S+\.\S+', '', clean)
    clean = re.sub(r'https?://\S+', '', clean)
    clean = re.sub(r'\b[A-Z][A-Z\s,\.]+\d+\b', '', clean)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean


def _is_clean_sentence(sent):
    words = sent.split()
    if len(words) < 5 or len(words) > 80:
        return False
    if re.search(r'arXiv:', sent, re.IGNORECASE):
        return False
    if re.search(r'\bv\d+\b.*\[cs\.', sent):
        return False
    if re.search(r'\S+@\S+\.\S+', sent):
        return False
    alpha_ratio = sum(c.isalpha() for c in sent) / max(len(sent), 1)
    return alpha_ratio >= 0.5


def _filter_overlapping_spans(scored_spans):
    """Greedy non-overlapping filter: keep highest confidence first."""
    scored_spans.sort(key=lambda x: -x[3])  # sort by confidence desc
    taken = set()
    filtered = []
    for (s, e, etype, conf) in scored_spans:
        overlap = any(not (e < ts or te < s) for (ts, te) in taken)
        if not overlap:
            filtered.append((s, e, etype, conf))
            taken.add((s, e))
    return filtered


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    num_entity_types = len(ENTITY_TYPES)

    print(f"=== CAST span-teacher pseudo-label generation ===")
    print(f"  teacher:         {args.teacher_ckpt}")
    print(f"  tau_ner:         {args.tau_ner}")
    print(f"  max_span_width:  {args.max_span_width}")
    print(f"  per-rel tau_re:")
    for rel, tau in sorted(PER_REL_TAU.items(), key=lambda x: -x[1]):
        print(f"    {rel:20s}: {tau}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load span teacher
    model = BertKGExtractor(
        args.model_name,
        num_bio_tags=NUM_BIO_TAGS,
        num_relations=NUM_RELATIONS,
        num_entity_types=num_entity_types,
        use_span_ner=True,
        max_span_width=args.max_span_width,
    ).to(device)
    ckpt = torch.load(args.teacher_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["encoder"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    teacher_metrics = ckpt.get("metrics", {})
    print(f"  teacher NER F1:    {teacher_metrics.get('ner_f1', 'N/A')}")
    print(f"  teacher Triple F1: {teacher_metrics.get('triple_f1', 'N/A')}")

    # Load arXiv
    arxiv_jsonl = args.arxiv_jsonl or str(
        Path(__file__).parent / "data" / "arxiv_real" / "cs_validation.jsonl"
    )
    raw_texts = []
    with open(arxiv_jsonl) as f:
        for line in f:
            raw_texts.append(json.loads(line).get("text", ""))
    print(f"  arXiv docs: {len(raw_texts)}")

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from collections import Counter
    rel_counter = Counter()
    ent_type_counter = Counter()
    n_sentences = 0
    n_with_entities = 0

    with out_path.open("w") as fout:
        for doc_idx, text in enumerate(raw_texts):
            clean = _clean_arxiv_text(text)
            sentences = [s.strip() + "." for s in clean.split(". ")
                         if _is_clean_sentence(s.strip())]

            for sent in sentences:
                words = sent.split()
                n_words = len(words)
                n_sentences += 1

                enc = tokenizer(
                    words, is_split_into_words=True,
                    return_tensors="pt", padding=True,
                    truncation=True, max_length=args.max_length,
                )
                word_ids = enc.word_ids(batch_index=0)
                enc_device = {k: v.to(device) for k, v in enc.items()}

                with torch.no_grad():
                    hidden = model.encode(
                        modality="text",
                        input_ids=enc_device["input_ids"],
                        attention_mask=enc_device["attention_mask"],
                    )
                    span_logits, candidates = model.forward_span_ner(
                        hidden[0], word_ids, n_words, args.max_span_width,
                    )

                if not candidates:
                    continue

                # Extract confident entity spans
                span_probs = torch.softmax(span_logits, dim=-1)
                pred_types = span_logits.argmax(dim=-1).tolist()
                pred_confs = span_probs.max(dim=-1).values.tolist()

                scored_spans = []
                for (s, e), etype_id, conf in zip(candidates, pred_types, pred_confs):
                    if etype_id > 0 and conf >= args.tau_ner:
                        etype = ID2ENTITY_TYPE.get(etype_id, "Method")
                        scored_spans.append((s, e, etype, conf))

                # Greedy non-overlapping filter
                entities = _filter_overlapping_spans(scored_spans)

                if len(entities) < 2:
                    continue
                n_with_entities += 1
                for (_, _, et, _) in entities:
                    ent_type_counter[et] += 1

                # RE prediction on all entity pairs
                span_pairs = [((h[0], h[1]), (t[0], t[1]))
                              for h in entities for t in entities if h != t]
                if not span_pairs:
                    continue

                with torch.no_grad():
                    re_logits = model.forward_re(hidden[0], word_ids, span_pairs)
                re_probs = torch.softmax(re_logits, dim=-1)
                pred_rels = re_logits.argmax(dim=-1).tolist()
                pred_rel_confs = re_probs.max(dim=-1).values.tolist()

                type_by_span = {(s, e): t for (s, e, t, _) in entities}
                conf_by_span = {(s, e): c for (s, e, _, c) in entities}

                for (h_span, t_span), rel_id, conf in zip(span_pairs, pred_rels, pred_rel_confs):
                    if rel_id == NO_REL_ID:
                        continue
                    rel_name = ID2REL[rel_id]
                    tau_re = PER_REL_TAU.get(rel_name, 0.70)
                    if conf < tau_re:
                        continue

                    hs, he = h_span
                    ts, te = t_span
                    h_conf = conf_by_span.get(h_span, 0.0)
                    t_conf = conf_by_span.get(t_span, 0.0)

                    record = {
                        "synth_sentence": sent,
                        "source_sentence": sent,
                        "head": " ".join(words[hs:he + 1]),
                        "rel": rel_name,
                        "tail": " ".join(words[ts:te + 1]),
                        "rel_id": int(rel_id),
                        "entity_type": type_by_span.get(h_span, "Method"),
                        "tail_entity_type": type_by_span.get(t_span, "Method"),
                        "containment": 1.0,
                        "pseudo_label": True,
                        "ner_confidence": float(min(h_conf, t_conf)),
                        "re_confidence": float(conf),
                    }
                    fout.write(json.dumps(record) + "\n")
                    rel_counter[rel_name] += 1

            if (doc_idx + 1) % 500 == 0:
                print(f"  [{doc_idx+1}/{len(raw_texts)}] "
                      f"sentences={n_sentences} with_entities={n_with_entities}")

    total = sum(rel_counter.values())
    print(f"\n=== CAST span-teacher pseudo-labels ===")
    print(f"  Total sentences processed: {n_sentences}")
    print(f"  Sentences with 2+ entities: {n_with_entities}")
    print(f"  Total relation pseudo-labels: {total}")
    if total > 0:
        for rel, cnt in rel_counter.most_common():
            pct = cnt / total * 100
            gold_pct = GOLD_SHARE.get(rel, 0) * 100
            print(f"  {rel:20s}: {cnt:4d} ({pct:5.1f}%) [gold: {gold_pct:.1f}%]")
    print(f"\n  Entity type distribution:")
    for et, cnt in ent_type_counter.most_common():
        print(f"    {et:25s}: {cnt:4d}")
    print(f"  Output: {out_path}")


if __name__ == "__main__":
    main()
