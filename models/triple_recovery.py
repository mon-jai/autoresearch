"""
Stage 2c triple-recovery scorer.

Given a sampled sentence from the LoRA decoder AND the source (h, r, t)
triple the sentence was supposed to express, run the frozen recovery
encoder over the sentence and check whether the source triple is
recoverable.

Score: 1.0 if the source triple is exactly recovered,
       0.5 if a triple with matching (head_string, tail_string) substrings
           exists but the relation is wrong,
       0.3 if only the relation is recovered among *some* pair,
       0.0 otherwise.

This reward is the "β" term in Stage 2c's composite reward, and directly
addresses the Stage 2b Finding B domain-hijack failure mode
(see reports/stage2/stage2_008_DESIGN.md).
"""
from typing import List, Tuple

import torch

from data.scierc import NO_REL_ID, REL2ID
from eval.triple_f1 import _bio_to_spans, _word_level_bio_from_token_logits


def _tokenize_sentence(sentence: str, tokenizer, max_length: int = 128, device="cuda"):
    """
    Word-level tokenization with is_split_into_words=True so we can use the
    encoder's existing span-to-word machinery.
    """
    words = sentence.strip().split()
    if not words:
        words = ["."]
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
    return enc, word_ids, words


def _extract_triples_from_sentence(
    sentence: str,
    frozen_encoder,
    tokenizer,
    device: str,
    max_length: int = 128,
):
    """
    Run frozen encoder on `sentence`, return a list of
    (head_word_string, tail_word_string, rel_id) triples that the encoder
    predicted.
    """
    frozen_encoder.eval()
    enc, word_ids, words = _tokenize_sentence(sentence, tokenizer, max_length, device)

    with torch.no_grad():
        hidden = frozen_encoder.encode(
            modality="text",
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        )
        ner_logits = frozen_encoder.forward_ner(hidden)  # (1, T, C)

    word_bio = _word_level_bio_from_token_logits(ner_logits[0], word_ids)
    pred_spans = _bio_to_spans(word_bio)   # list[(s, e, type)]
    span_list = [(s, e) for (s, e, _) in pred_spans]
    if len(span_list) < 2:
        return []

    pairs = [(a, b) for a in span_list for b in span_list if a != b]
    with torch.no_grad():
        re_logits = frozen_encoder.forward_re(hidden[0], word_ids, pairs)
        pred_re = re_logits.argmax(dim=-1).tolist()

    triples = []
    for (h_span, t_span), rel_id in zip(pairs, pred_re):
        if rel_id == NO_REL_ID:
            continue
        hs, he = h_span
        ts, te = t_span
        if he >= len(words) or te >= len(words):
            continue
        head_str = " ".join(words[hs:he + 1]).lower()
        tail_str = " ".join(words[ts:te + 1]).lower()
        triples.append((head_str, tail_str, rel_id))
    return triples


def _string_overlap(a: str, b: str) -> float:
    """Symmetric word-set overlap in [0, 1]."""
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(len(sa), len(sb))


def score_one(
    sentence: str,
    source_triple: Tuple[str, str, str],   # (head_str, rel_str, tail_str)
    frozen_encoder,
    tokenizer,
    device: str = "cuda",
) -> float:
    """
    Return a scalar in [0, 1] measuring how well the source triple is
    recovered from `sentence`.
    """
    src_h, src_rel_str, src_t = source_triple
    src_rel_id = REL2ID.get(src_rel_str, NO_REL_ID)
    src_h_l = src_h.lower()
    src_t_l = src_t.lower()

    pred_triples = _extract_triples_from_sentence(
        sentence, frozen_encoder, tokenizer, device,
    )
    if not pred_triples:
        return 0.0

    best = 0.0
    for (p_h, p_t, p_rel) in pred_triples:
        h_sim = _string_overlap(p_h, src_h_l)
        t_sim = _string_overlap(p_t, src_t_l)
        span_sim = (h_sim + t_sim) / 2
        rel_match = (p_rel == src_rel_id)

        if span_sim >= 0.8 and rel_match:
            score = 1.0
        elif span_sim >= 0.8:
            score = 0.5
        elif rel_match and span_sim >= 0.3:
            score = 0.3
        else:
            score = 0.0
        if score > best:
            best = score
    return best


def score_batch(
    sentences: List[str],
    source_triples: List[Tuple[str, str, str]],
    frozen_encoder,
    tokenizer,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Vectorized wrapper. Returns a (B,) float tensor on `device`.
    No gradient (frozen encoder + discrete metric).
    """
    assert len(sentences) == len(source_triples)
    scores = [
        score_one(s, t, frozen_encoder, tokenizer, device)
        for s, t in zip(sentences, source_triples)
    ]
    return torch.tensor(scores, device=device, dtype=torch.float32)
