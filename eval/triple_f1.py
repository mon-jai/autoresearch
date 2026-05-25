"""
Triple F1 evaluation for SciERC.

We report THREE metrics, in increasing order of strictness:

1. **NER F1** (BIO tag agreement, micro): pure entity span recovery.
2. **RE F1 (gold spans)**: with GOLD entity spans, classify all pairs and
   measure micro-F1 over the relation labels. Decouples relation
   classification from span errors.
3. **Triple F1 (full pipeline)**: extract spans from NER predictions, form
   all pairs of predicted spans, classify each, and compare to gold
   `(head_span, tail_span, rel)` triples. This is the headline number.

A predicted triple counts as correct iff:
    - head span matches a gold span exactly (start, end) — type doesn't have to match
    - tail span matches a gold span exactly
    - relation type matches
"""
from typing import Optional

import torch
import torch.nn.functional as F

from data.scierc import BIO_TAG2ID, ID2BIO, NUM_RELATIONS, NUM_BIO_TAGS, NO_REL_ID, ENTITY_TYPES


def _bio_to_spans(bio_ids: list) -> list:
    """Convert a list of BIO tag ids to list of (start, end_inclusive, type) spans."""
    spans = []
    cur_start = None
    cur_type = None
    for i, tid in enumerate(bio_ids):
        tag = ID2BIO[tid]
        if tag == "O":
            if cur_start is not None:
                spans.append((cur_start, i - 1, cur_type))
                cur_start = None
                cur_type = None
        elif tag.startswith("B-"):
            if cur_start is not None:
                spans.append((cur_start, i - 1, cur_type))
            cur_start = i
            cur_type = tag[2:]
        elif tag.startswith("I-"):
            if cur_start is None:
                # I- without B-: treat as B-
                cur_start = i
                cur_type = tag[2:]
            elif cur_type != tag[2:]:
                # type changes mid-span: close old, start new
                spans.append((cur_start, i - 1, cur_type))
                cur_start = i
                cur_type = tag[2:]
    if cur_start is not None:
        spans.append((cur_start, len(bio_ids) - 1, cur_type))
    return spans


def _is_valid_bio_transition(prev_tag_id: int, cand_tag_id: int) -> bool:
    """
    Stage 2-006: BIO transition constraint.
    Valid:   anything → O
             anything → B-X
             B-X      → I-X
             I-X      → I-X
    Invalid: O   → I-X
             B-X → I-Y  (Y != X)
             I-X → I-Y  (Y != X)
    """
    cand = ID2BIO[cand_tag_id]
    if cand == "O" or cand.startswith("B-"):
        return True
    # cand starts with "I-"
    cand_type = cand[2:]
    prev = ID2BIO[prev_tag_id]
    return prev == f"B-{cand_type}" or prev == f"I-{cand_type}"


def _word_level_emissions(token_logits, word_ids):
    """
    Aggregate per-token logits into per-word emissions by taking the
    first subword's logits for each word. Returns a list of length
    (num_unique_words), each element shape (NUM_BIO_TAGS,).
    """
    seen = set()
    word_emissions: dict = {}
    for i, wid in enumerate(word_ids):
        if wid is None or wid in seen:
            continue
        word_emissions[wid] = token_logits[i]
        seen.add(wid)
    if not word_emissions:
        return []
    max_wid = max(word_emissions.keys())
    out = []
    for w in range(max_wid + 1):
        if w in word_emissions:
            out.append(word_emissions[w])
        else:
            # Word missing (shouldn't happen with HF tokenizer + is_split_into_words=True)
            # Default to O — make a fake one-hot zero vector with O dominant
            zero = token_logits[0] * 0
            zero[BIO_TAG2ID["O"]] = 1.0
            out.append(zero)
    return out


def _word_level_bio_from_token_logits(token_logits, word_ids):
    """
    Stage 2-006: BIO-constrained greedy decoding (Viterbi-equivalent without
    transition scores). For each word, pick the highest-scoring tag whose
    transition from the previous tag is valid.

    Equivalent to plain argmax when there are no constraint violations,
    so this can ONLY improve metrics, never hurt them.

    token_logits: (T, NUM_BIO_TAGS) raw logits for one example
    word_ids:     list[int|None] length T
    Returns: list of word-level BIO tag ids
    """
    word_emissions = _word_level_emissions(token_logits, word_ids)
    if not word_emissions:
        return []

    O_id = BIO_TAG2ID["O"]
    out = []
    prev_tag = O_id  # virtual start-of-sequence = O
    for emission in word_emissions:
        # Sort tag indices by score descending
        scores = emission.tolist()
        ranked = sorted(range(len(scores)), key=lambda t: scores[t], reverse=True)
        chosen = O_id
        for cand in ranked:
            if _is_valid_bio_transition(prev_tag, cand):
                chosen = cand
                break
        out.append(chosen)
        prev_tag = chosen
    return out


def _prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


@torch.no_grad()
def evaluate(model, dataloader, device) -> dict:
    """
    Run the model on dataloader and return:
        {
            "ner_p", "ner_r", "ner_f1",          # span-level (start, end, type)
            "re_p", "re_r", "re_f1",             # relation classification on gold spans
            "triple_p", "triple_r", "triple_f1", # full pipeline
            "n_examples", "n_gold_entities", "n_gold_triples"
        }
    """
    model.eval()

    ner_tp = ner_fp = ner_fn = 0
    re_tp = re_fp = re_fn = 0
    triple_tp = triple_fp = triple_fn = 0
    n_examples = 0
    n_gold_entities = 0
    n_gold_triples = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        word_ids_list = batch["word_ids"]
        gold_entities_list = batch["gold_entities"]
        gold_relations_list = batch["gold_relations"]
        num_words_list = batch["num_words"]

        hidden = model.encode(modality="text", input_ids=input_ids, attention_mask=attention_mask)
        ner_logits = model.forward_ner(hidden)  # (B, T, C)

        # CRF Viterbi decode (batch-level) if available; otherwise per-example greedy.
        crf_decoded = None
        if getattr(model, "use_crf", False) and model.crf is not None:
            # CRF.decode needs a mask; use attention_mask (True = valid token).
            crf_mask = attention_mask.bool()
            crf_decoded = model.crf.decode(ner_logits, mask=crf_mask)  # list of list[int]

        for b_idx in range(input_ids.size(0)):
            n_examples += 1
            num_words = num_words_list[b_idx]
            gold_ents = gold_entities_list[b_idx]   # list[(s,e,type)]
            gold_rels = gold_relations_list[b_idx]  # list[((hs,he),(ts,te),rel_id)]

            n_gold_entities += len(gold_ents)
            n_gold_triples += len(gold_rels)

            # ── 1. NER F1 ────────────────────────────────────────────
            if crf_decoded is not None:
                # CRF gives token-level tag ids; convert to word-level.
                token_tags = crf_decoded[b_idx]
                wids = word_ids_list[b_idx]
                word_bio = []
                seen = set()
                for tok_i, wid in enumerate(wids):
                    if wid is None or wid in seen:
                        continue
                    if tok_i < len(token_tags):
                        word_bio.append(token_tags[tok_i])
                    seen.add(wid)
            else:
                word_bio = _word_level_bio_from_token_logits(ner_logits[b_idx], word_ids_list[b_idx])
            pred_spans = _bio_to_spans(word_bio)
            pred_ent_set = {(s, e, t) for (s, e, t) in pred_spans}
            gold_ent_set = {(s, e, t) for (s, e, t) in gold_ents}

            ner_tp += len(pred_ent_set & gold_ent_set)
            ner_fp += len(pred_ent_set - gold_ent_set)
            ner_fn += len(gold_ent_set - pred_ent_set)

            # ── 2. RE F1 on GOLD spans ───────────────────────────────
            # Form all ordered pairs of distinct gold spans, classify each,
            # and DROP predictions of NO_REL (the model's "no relation" class).
            gold_span_list = [(s, e) for (s, e, _) in gold_ents]
            gold_pairs = [(a, b) for a in gold_span_list for b in gold_span_list if a != b]
            if gold_pairs:
                re_logits = model.forward_re(hidden[b_idx], word_ids_list[b_idx], gold_pairs)
                re_pred = re_logits.argmax(dim=-1).tolist()
                pred_rel_triples = {
                    (h, t, pid) for (h, t), pid in zip(gold_pairs, re_pred) if pid != NO_REL_ID
                }
                gold_rel_triples = {(h, t, rid) for (h, t, rid) in gold_rels}
                re_tp += len(pred_rel_triples & gold_rel_triples)
                re_fp += len(pred_rel_triples - gold_rel_triples)
                re_fn += len(gold_rel_triples - pred_rel_triples)

            # ── 3. Triple F1 (full pipeline) ─────────────────────────
            pred_span_list = [(s, e) for (s, e, _) in pred_spans]
            pred_pairs = [(a, b) for a in pred_span_list for b in pred_span_list if a != b]
            if pred_pairs:
                pred_re_logits = model.forward_re(hidden[b_idx], word_ids_list[b_idx], pred_pairs)
                pred_re_ids = pred_re_logits.argmax(dim=-1).tolist()
                pred_full_triples = {
                    (h, t, pid) for (h, t), pid in zip(pred_pairs, pred_re_ids) if pid != NO_REL_ID
                }
            else:
                pred_full_triples = set()

            gold_full_triples = {(h, t, rid) for (h, t, rid) in gold_rels}
            triple_tp += len(pred_full_triples & gold_full_triples)
            triple_fp += len(pred_full_triples - gold_full_triples)
            triple_fn += len(gold_full_triples - pred_full_triples)

    np, nr, nf = _prf(ner_tp, ner_fp, ner_fn)
    rp, rr, rf = _prf(re_tp, re_fp, re_fn)
    tp_, tr_, tf_ = _prf(triple_tp, triple_fp, triple_fn)

    return {
        "ner_p": np, "ner_r": nr, "ner_f1": nf,
        "re_p": rp, "re_r": rr, "re_f1": rf,
        "triple_p": tp_, "triple_r": tr_, "triple_f1": tf_,
        "n_examples": n_examples,
        "n_gold_entities": n_gold_entities,
        "n_gold_triples": n_gold_triples,
    }


def main():
    import argparse
    import importlib
    import inspect
    import sys

    p = argparse.ArgumentParser(
        description="Triple F1 evaluation (BIO or span, auto-detected from checkpoint).")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--dataset", default="scierc",
                   choices=["scierc", "scier", "conll04", "ade", "accord", "cuad"])
    p.add_argument("--model-name", default=None,
                   help="HuggingFace model ID (must match training backbone)")
    p.add_argument("--split", default="test", choices=["train", "dev", "test"])
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for datasets with runtime train/dev splits")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--device", default=None)
    p.add_argument("--span-threshold", type=float, default=0.5,
                   help="NER confidence threshold (span models only)")
    p.add_argument("--out-json", default=None, help="Write metrics dict as JSON to this path.")
    args = p.parse_args()

    _DATASET_REGISTRY = {
        "scierc": "data.scierc", "scier": "data.scier",
        "conll04": "data.conll04", "ade": "data.ade",
        "accord": "data.code_accord", "cuad": "data.cuad",
    }
    _DEFAULT_MODEL = {
        "scierc": "allenai/scibert_scivocab_uncased",
        "scier": "allenai/scibert_scivocab_uncased",
        "conll04": "bert-base-uncased",
        "ade": "allenai/scibert_scivocab_uncased",
        "accord": "bert-base-uncased",
        "cuad": "microsoft/deberta-large",
    }
    if args.model_name is None:
        args.model_name = _DEFAULT_MODEL.get(args.dataset, "bert-base-uncased")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ds_mod = importlib.import_module(_DATASET_REGISTRY[args.dataset])

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dl_kwargs = dict(batch_size=args.batch_size, max_length=args.max_length)
    if "seed" in inspect.signature(ds_mod.build_dataloaders).parameters:
        dl_kwargs["seed"] = args.seed
    train_loader, dev_loader, test_loader = ds_mod.build_dataloaders(tokenizer, **dl_kwargs)
    loader = {"train": train_loader, "dev": dev_loader, "test": test_loader}[args.split]

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["encoder"] if "encoder" in ckpt else ckpt
    is_span = any(k.startswith("span_ner_head.") for k in state)

    if is_span:
        from eval.span_f1 import evaluate_span, load_span_model
        print(f"Detected span model. Loading: {args.checkpoint}")
        model, entity_type2id, id2entity_type = load_span_model(
            args.checkpoint, args.model_name, ds_mod, device)
        print(f"Evaluating {args.split} split ({len(loader.dataset)} examples)...")
        metrics = evaluate_span(
            model, loader, device, ds_mod,
            entity_type2id, id2entity_type,
            span_threshold=args.span_threshold,
            verbose=True,
        )
        print(f"\n{args.split.upper()} NER F1={metrics['ner_f1']:.4f}  "
              f"Triple F1={metrics['triple_f1']:.4f}")
    else:
        from models.bert_kg_encoder import BertKGExtractor
        if args.dataset != "scierc":
            print(f"  [warn] BIO eval uses SciERC tag scheme; NER F1 unreliable for "
                  f"dataset={args.dataset!r}. Use a span checkpoint for other datasets.")
        print(f"Detected BIO model. Loading: {args.checkpoint}")
        model = BertKGExtractor(
            args.model_name,
            num_bio_tags=ds_mod.NUM_BIO_TAGS,
            num_relations=ds_mod.NUM_RELATIONS,
        ).to(device)
        model_sd = model.state_dict()
        filtered_state = {k: v for k, v in state.items()
                          if k in model_sd and v.shape == model_sd[k].shape}
        missing, _ = model.load_state_dict(filtered_state, strict=False)
        if missing:
            print(f"  [warn] missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        model.eval()
        print(f"Evaluating {args.split} split ({len(loader.dataset)} examples)...")
        metrics = evaluate(model, loader, device)
        print(f"\n{args.split.upper()} NER F1={metrics['ner_f1']:.4f}  "
              f"RE F1={metrics['re_f1']:.4f}  "
              f"Triple F1={metrics['triple_f1']:.4f}")

    if args.out_json:
        import json as _json
        from pathlib import Path as _Path
        _out = _Path(args.out_json)
        _out.parent.mkdir(parents=True, exist_ok=True)
        with open(_out, "w") as _f:
            _json.dump(metrics, _f, indent=2)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
