"""
Stage 2-016: CAST-inspired class-adaptive pseudo-label generation.

CAST (ACL Findings 2023) uses per-class thresholds based on class precision.
We adapt this: classes overrepresented in pseudo-labels get stricter thresholds,
underrepresented classes get relaxed thresholds.

Gold distribution (SciERC train):
  USED-FOR: 52.5% | CONJUNCTION: 12.4% | EVALUATE-FOR: 9.7%
  HYPONYM-OF: 9.3% | PART-OF: 5.6% | FEATURE-OF: 5.4% | COMPARE: 5.2%

Flat τ_re=0.70 pseudo-label distribution:
  USED-FOR: 72.7% (overrepresented +20pp)
  All others: underrepresented

Fix: per-relation τ_re that pushes pseudo-labels toward gold distribution.
"""
import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer

from data.scierc import ID2REL, NO_REL_ID
from models.bert_kg_encoder import BertKGExtractor
from eval.triple_f1 import _bio_to_spans, _word_level_bio_from_token_logits


# Gold distribution target (from SciERC train)
GOLD_SHARE = {
    "USED-FOR": 0.525,
    "CONJUNCTION": 0.124,
    "EVALUATE-FOR": 0.097,
    "HYPONYM-OF": 0.093,
    "PART-OF": 0.056,
    "FEATURE-OF": 0.054,
    "COMPARE": 0.052,
}

# Per-relation τ_re: strict for overrepresented, relaxed for underrepresented
# Computed to push pseudo-label distribution closer to gold
PER_REL_TAU = {
    "USED-FOR": 0.82,       # strict — overrepresented
    "CONJUNCTION": 0.65,    # moderate
    "EVALUATE-FOR": 0.60,   # slightly relaxed
    "HYPONYM-OF": 0.60,     # slightly relaxed
    "PART-OF": 0.50,        # relaxed — underrepresented
    "FEATURE-OF": 0.50,     # relaxed — underrepresented
    "COMPARE": 0.50,        # relaxed — underrepresented
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="allenai/scibert_scivocab_uncased")
    p.add_argument("--teacher-ckpt", default="checkpoints/stage2_007_best.pt")
    p.add_argument("--arxiv-jsonl", default=None)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--tau-ner", type=float, default=0.90)
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


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"=== CAST-adaptive pseudo-label generation ===")
    print(f"  teacher:   {args.teacher_ckpt}")
    print(f"  τ_ner:     {args.tau_ner}")
    print(f"  per-rel τ_re:")
    for rel, tau in sorted(PER_REL_TAU.items(), key=lambda x: -x[1]):
        print(f"    {rel:20s}: {tau}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = BertKGExtractor(args.model_name).to(device)
    ckpt = torch.load(args.teacher_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["encoder"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"  teacher F1: {ckpt.get('metrics', {}).get('triple_f1', 'N/A')}")

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
    n_sentences = 0

    with out_path.open("w") as fout:
        for doc_idx, text in enumerate(raw_texts):
            clean = _clean_arxiv_text(text)
            sentences = [s.strip() + "." for s in clean.split(". ")
                         if _is_clean_sentence(s.strip())]

            for sent in sentences:
                words = sent.split()
                n_sentences += 1

                enc = tokenizer(
                    words, is_split_into_words=True,
                    return_tensors="pt", padding=True,
                    truncation=True, max_length=args.max_length,
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

                logits = ner_logits[0]
                probs = torch.softmax(logits, dim=-1)
                word_bio = _word_level_bio_from_token_logits(logits, word_ids)
                spans = _bio_to_spans(word_bio)

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
                    if span_confs and min(span_confs) >= args.tau_ner:
                        entities.append((s, e, etype, min(span_confs)))

                if len(entities) < 2:
                    continue

                span_pairs = [((h[0], h[1]), (t[0], t[1]))
                              for h in entities for t in entities if h != t]
                if not span_pairs:
                    continue

                with torch.no_grad():
                    re_logits = model.forward_re(hidden[0], word_ids, span_pairs)
                re_probs = torch.softmax(re_logits, dim=-1)
                pred_rels = re_logits.argmax(dim=-1).tolist()
                pred_confs = re_probs.max(dim=-1).values.tolist()

                type_by_span = {(s, e): t for (s, e, t, _) in entities}
                for (h_span, t_span), rel_id, conf in zip(span_pairs, pred_rels, pred_confs):
                    if rel_id == NO_REL_ID:
                        continue
                    rel_name = ID2REL[rel_id]
                    # CAST: per-relation threshold
                    tau_re = PER_REL_TAU.get(rel_name, 0.70)
                    if conf < tau_re:
                        continue

                    hs, he = h_span
                    ts, te = t_span
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
                        "ner_confidence": float(min(
                            c for (s, e, t, c) in entities
                            if (s, e) in (h_span, t_span)
                        )) if any((s, e) in (h_span, t_span) for (s, e, t, c) in entities) else 0.0,
                        "re_confidence": float(conf),
                    }
                    fout.write(json.dumps(record) + "\n")
                    rel_counter[rel_name] += 1

            if (doc_idx + 1) % 500 == 0:
                print(f"  [{doc_idx+1}/{len(raw_texts)}] sentences={n_sentences}")

    total = sum(rel_counter.values())
    print(f"\n=== CAST-adaptive pseudo-labels ===")
    print(f"  Total relations: {total}")
    for rel, cnt in rel_counter.most_common():
        pct = cnt / total * 100 if total > 0 else 0
        gold_pct = GOLD_SHARE.get(rel, 0) * 100
        print(f"  {rel:20s}: {cnt:4d} ({pct:5.1f}%) [gold: {gold_pct:.1f}%]")
    print(f"  Output: {out_path}")


if __name__ == "__main__":
    main()
