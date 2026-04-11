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


def _word_level_bio_from_token_logits(token_logits, word_ids):
    """
    token_logits: (T, NUM_BIO_TAGS) raw logits for one example
    word_ids:     list[int|None] length T
    Returns: list of word-level BIO tag ids (length = num unique non-None word ids)
    """
    pred_token = token_logits.argmax(dim=-1).tolist()  # length T

    # Take the first subword's prediction for each word
    word_bio = {}
    for i, wid in enumerate(word_ids):
        if wid is None:
            continue
        if wid not in word_bio:
            word_bio[wid] = pred_token[i]
    if not word_bio:
        return []
    max_wid = max(word_bio.keys())
    return [word_bio.get(w, BIO_TAG2ID["O"]) for w in range(max_wid + 1)]


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

        hidden = model.encode(input_ids, attention_mask)
        ner_logits = model.forward_ner(hidden)  # (B, T, C)

        for b_idx in range(input_ids.size(0)):
            n_examples += 1
            num_words = num_words_list[b_idx]
            gold_ents = gold_entities_list[b_idx]   # list[(s,e,type)]
            gold_rels = gold_relations_list[b_idx]  # list[((hs,he),(ts,te),rel_id)]

            n_gold_entities += len(gold_ents)
            n_gold_triples += len(gold_rels)

            # ── 1. NER F1 ────────────────────────────────────────────
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
