"""
Stage 2-011: Generate pseudo-labels on arXiv abstracts using the
frozen stage2b encoder as teacher.

For each sentence in the arXiv corpus:
  1. Run encoder → NER logits → BIO tags with confidence
  2. Extract entity spans where min-token confidence > τ_ner
  3. For each pair of confident entities → RE logits → relation + confidence
  4. Keep relations where rel ≠ NO_REL and confidence > τ_re
  5. Write accepted sentences to jsonl (synth_loader compatible format)

Usage:
    uv run python generate_pseudo_labels.py \
        --teacher-ckpt checkpoints/stage2_007_best.pt \
        --out-jsonl data/stage2e_pseudo_arXiv.jsonl
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from data.scierc import BIO_TAG2ID, ID2BIO, ID2REL, NO_REL_ID, NUM_BIO_TAGS
from data.arxiv_real import build_arxiv_loader
from models.bert_kg_encoder import BertKGExtractor
from models.critic import RealismCritic
from eval.triple_f1 import _bio_to_spans, _word_level_bio_from_token_logits


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="allenai/scibert_scivocab_uncased")
    p.add_argument("--teacher-ckpt", default="checkpoints/stage2_007_best.pt")
    p.add_argument("--arxiv-jsonl", default=None)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--tau-ner", type=float, default=0.7,
                   help="Min per-token softmax confidence for NER spans.")
    p.add_argument("--tau-re", type=float, default=0.5,
                   help="Min softmax confidence for relation predictions.")
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--device", default=None)
    return p.parse_args()


def _load_teacher(model_name, ckpt_path, device):
    model = BertKGExtractor(model_name).to(device)
    hidden = model.backbone.hidden_size
    critic = RealismCritic(hidden).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["encoder"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"[pseudo] loaded teacher step={ckpt['step']} "
          f"triple_f1={ckpt['metrics']['triple_f1']:.4f}")
    return model


def _tokenize_sentence_words(words, tokenizer, max_length, device):
    """Tokenize a word list and return encoding + word_ids."""
    enc = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    word_ids = enc.word_ids(batch_index=0)
    enc = {k: v.to(device) for k, v in enc.items()}
    return enc, word_ids


def _extract_confident_entities(ner_logits, word_ids, num_words, tau_ner):
    """
    From NER logits (1, T, C), extract entity spans where every token
    in the span has softmax confidence >= tau_ner.

    Returns list of (start, end_inclusive, entity_type, min_confidence).
    """
    logits = ner_logits[0]  # (T, C)
    probs = torch.softmax(logits, dim=-1)  # (T, C)

    # Get constrained BIO tags
    word_bio = _word_level_bio_from_token_logits(logits, word_ids)
    spans = _bio_to_spans(word_bio)  # [(s, e, type), ...]

    # Compute per-word confidence (first subword's max prob on its predicted tag)
    word_conf = {}
    seen = set()
    for tok_i, wid in enumerate(word_ids):
        if wid is None or wid in seen:
            continue
        if wid < len(word_bio):
            tag_id = word_bio[wid]
            word_conf[wid] = probs[tok_i, tag_id].item()
        seen.add(wid)

    confident_spans = []
    for (s, e, etype) in spans:
        # Min confidence across all words in the span
        span_confs = [word_conf.get(w, 0.0) for w in range(s, e + 1)]
        if not span_confs:
            continue
        min_conf = min(span_confs)
        if min_conf >= tau_ner:
            confident_spans.append((s, e, etype, min_conf))

    return confident_spans


def _extract_confident_relations(model, hidden, word_ids, entities, tau_re):
    """
    For each ordered pair of confident entity spans, predict relation.
    Keep if rel ≠ NO_REL and confidence > tau_re.

    Returns list of ((hs, he), (ts, te), rel_id, h_type, t_type, confidence).
    """
    if len(entities) < 2:
        return []

    spans = [(s, e) for (s, e, _, _) in entities]
    type_by_span = {(s, e): t for (s, e, t, _) in entities}
    pairs = [(h, t) for h in spans for t in spans if h != t]
    if not pairs:
        return []

    with torch.no_grad():
        re_logits = model.forward_re(hidden[0], word_ids, pairs)
    re_probs = torch.softmax(re_logits, dim=-1)
    pred_rels = re_logits.argmax(dim=-1).tolist()
    pred_confs = re_probs.max(dim=-1).values.tolist()

    results = []
    for (h_span, t_span), rel_id, conf in zip(pairs, pred_rels, pred_confs):
        if rel_id == NO_REL_ID:
            continue
        if conf < tau_re:
            continue
        h_type = type_by_span.get(h_span, "Method")
        t_type = type_by_span.get(t_span, "Method")
        results.append((h_span, t_span, rel_id, h_type, t_type, conf))

    return results


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"=== Stage 2-011: pseudo-label generation ===")
    print(f"  teacher:   {args.teacher_ckpt}")
    print(f"  τ_ner:     {args.tau_ner}")
    print(f"  τ_re:      {args.tau_re}")
    print(f"  output:    {args.out_jsonl}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = _load_teacher(args.model_name, args.teacher_ckpt, device)

    # Load arXiv raw text — we need the actual text, not just tokenized ids.
    # The arxiv_real module stores text in a jsonl; we read it directly.
    from data.arxiv_real import ArxivRealDataset
    arxiv_ds = ArxivRealDataset(
        tokenizer=tokenizer,
        max_length=args.max_length,
        jsonl_path=args.arxiv_jsonl,
    )
    print(f"  arXiv docs: {len(arxiv_ds)}")

    # Read raw text directly from the same jsonl ArxivRealDataset uses.
    arxiv_jsonl_path = args.arxiv_jsonl
    if arxiv_jsonl_path is None:
        arxiv_jsonl_path = str(Path(__file__).parent / "data" / "arxiv_real" / "cs_validation.jsonl")
    raw_texts = []
    with open(arxiv_jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            raw_texts.append(rec.get("text", ""))
    print(f"  raw texts loaded: {len(raw_texts)}")

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_sentences = 0
    n_with_entities = 0
    n_relations_written = 0

    with out_path.open("w") as fout:
        for doc_idx, text in enumerate(raw_texts):
            # Clean arXiv header noise: strip everything before "Abstract"
            # and remove metadata lines (arXiv IDs, emails, affiliations).
            clean = text.replace("\n", " ")
            # Try to find "Abstract" marker and skip header
            abs_idx = clean.lower().find("abstract")
            if abs_idx >= 0:
                clean = clean[abs_idx:]
                # Skip the "Abstract" word itself
                clean = clean[len("abstract"):].strip()
                # Remove leading numbers/whitespace left over
                clean = clean.lstrip("0123456789 .")
            # Remove arXiv IDs, emails, URLs
            import re
            clean = re.sub(r'arXiv:\S+', '', clean)
            clean = re.sub(r'\S+@\S+\.\S+', '', clean)
            clean = re.sub(r'https?://\S+', '', clean)
            # Remove lines that look like author/affiliation (mostly caps + numbers)
            clean = re.sub(r'\b[A-Z][A-Z\s,\.]+\d+\b', '', clean)
            clean = re.sub(r'\s+', ' ', clean).strip()

            # Split into sentences (simple: by period + space)
            sentences = []
            for s in clean.split(". "):
                s = s.strip()
                if len(s.split()) >= 5:  # skip very short fragments
                    if not s.endswith("."):
                        s += "."
                    sentences.append(s)

            for sent in sentences:
                words = sent.split()
                if len(words) < 5 or len(words) > 80:
                    continue
                n_sentences += 1

                enc, word_ids = _tokenize_sentence_words(
                    words, tokenizer, args.max_length, device,
                )

                with torch.no_grad():
                    hidden = model.encode(
                        modality="text",
                        input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"],
                    )
                    ner_logits = model.forward_ner(hidden)

                entities = _extract_confident_entities(
                    ner_logits, word_ids, len(words), args.tau_ner,
                )
                if not entities:
                    continue
                n_with_entities += 1

                relations = _extract_confident_relations(
                    model, hidden, word_ids, entities, args.tau_re,
                )
                if not relations:
                    continue

                for (h_span, t_span, rel_id, h_type, t_type, conf) in relations:
                    hs, he = h_span
                    ts, te = t_span
                    head_phrase = " ".join(words[hs:he + 1])
                    tail_phrase = " ".join(words[ts:te + 1])
                    record = {
                        "synth_sentence": sent,
                        "source_sentence": sent,  # self-reference (pseudo-label)
                        "head": head_phrase,
                        "rel": ID2REL[rel_id],
                        "tail": tail_phrase,
                        "rel_id": int(rel_id),
                        "entity_type": h_type,
                        "tail_entity_type": t_type,
                        "containment": 1.0,  # by construction (spans from the sentence)
                        "pseudo_label": True,
                        "ner_confidence": float(min(
                            c for (s, e, t, c) in entities
                            if (s, e) in (h_span, t_span)
                        )) if any((s, e) in (h_span, t_span) for (s, e, t, c) in entities) else 0.0,
                        "re_confidence": float(conf),
                    }
                    fout.write(json.dumps(record) + "\n")
                    n_relations_written += 1

            if (doc_idx + 1) % 200 == 0:
                print(f"  [{doc_idx+1}/{len(raw_texts)}] sentences={n_sentences} "
                      f"with_entities={n_with_entities} relations={n_relations_written}")

    print(f"\n=== DONE ===")
    print(f"  total sentences processed: {n_sentences}")
    print(f"  sentences with confident entities: {n_with_entities}")
    print(f"  relations written: {n_relations_written}")
    print(f"  wrote: {out_path}")


if __name__ == "__main__":
    main()
