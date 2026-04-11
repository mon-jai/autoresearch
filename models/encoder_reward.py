"""
Stage 2d β-term: encoder-based L_rec reward on a synthetic sentence.

Migration plan §4.3 defines the Stage 2c/d reward as

    reward = α · L_real(synth) + β · (1 − L_rec(E(synth)))

This module computes the `L_rec` side: given a sentence sampled from the
LoRA decoder and the (head, rel, tail) triple that sentence was supposed
to express, run the frozen stage2b encoder over the sentence with the
source triple as the only gold label and return the NER+RE
cross-entropy loss.

The loss is continuous (unlike the discrete triple-recovery score we
tried in Stage 2c / stage2_008), so β always has non-zero gradient with
respect to small decoder moves — that is the structural fix stage2_009
depends on.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from data.scierc import BIO_TAG2ID, NO_REL_ID, REL2ID
from models.bert_kg_encoder import compute_loss


@dataclass
class _SynthGold:
    """Fake sci-batch row built from one sampled sentence + source triple."""

    words: List[str]
    gold_entities: List[Tuple[int, int, str]]        # (s, e, type)
    gold_relations: List[Tuple[Tuple[int, int], Tuple[int, int], int]]


def _find_span(words: List[str], phrase: str) -> Optional[Tuple[int, int]]:
    """
    Longest-common-prefix-ish match: find the first span of words whose
    concatenation (lowercased, whitespace-joined) is either equal to the
    phrase or one is a prefix of the other.

    Returns (start, end_inclusive) or None.
    """
    target = phrase.lower().strip()
    if not target:
        return None
    lwords = [w.lower() for w in words]

    # 1. exact contiguous match
    target_tokens = target.split()
    tl = len(target_tokens)
    for i in range(len(lwords) - tl + 1):
        if lwords[i : i + tl] == target_tokens:
            return (i, i + tl - 1)

    # 2. fuzzy: any span of words whose joined lowercased text contains
    #    >= 50% of the target's tokens (set overlap).
    target_set = set(target_tokens)
    if not target_set:
        return None
    best = None
    best_score = 0.0
    max_span = min(tl + 2, len(lwords))
    for span_len in range(1, max_span + 1):
        for i in range(len(lwords) - span_len + 1):
            span_tokens = lwords[i : i + span_len]
            overlap = len(set(span_tokens) & target_set)
            score = overlap / max(len(target_set), 1)
            if score > best_score and score >= 0.5:
                best_score = score
                best = (i, i + span_len - 1)
    return best


def build_synth_batch(
    sentence: str,
    source_triple: Tuple[str, str, str],   # (head, rel_str, tail)
    tokenizer,
    max_length: int,
    device: str,
) -> Optional[dict]:
    """
    Build a one-example sci batch from a sampled sentence + source triple.
    Returns None if we cannot locate both head and tail in the sentence —
    the caller should treat that as maximum L_rec (β term = 0).
    """
    head_str, rel_str, tail_str = source_triple
    words = sentence.strip().split()
    if len(words) < 2:
        return None

    hs = _find_span(words, head_str)
    ts = _find_span(words, tail_str)
    if hs is None or ts is None:
        return None

    rel_id = REL2ID.get(rel_str, NO_REL_ID)
    if rel_id == NO_REL_ID:
        return None

    # Tokenize word-by-word so the encoder's existing word_ids path works.
    enc = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    word_ids = enc.word_ids(batch_index=0)

    # Build per-token BIO labels from the head/tail spans.
    # Entity type is not available for a generated sentence; we pick
    # 'Method' as a neutral default (it exists in SciERC's type set).
    ner_label_name_head = "B-Method"
    ner_label_name_head_i = "I-Method"
    labels_token_level = [BIO_TAG2ID["O"]] * len(word_ids)
    for (start, end) in (hs, ts):
        first = True
        for tok_i, wid in enumerate(word_ids):
            if wid is None:
                continue
            if start <= wid <= end:
                labels_token_level[tok_i] = (
                    BIO_TAG2ID[ner_label_name_head if first else ner_label_name_head_i]
                )
                first = False

    # Mask special tokens + padding with -100 so CE ignores them.
    for tok_i, wid in enumerate(word_ids):
        if wid is None:
            labels_token_level[tok_i] = -100

    ner_labels = torch.tensor(labels_token_level, dtype=torch.long).unsqueeze(0)

    batch = {
        "input_ids": enc["input_ids"].to(device),
        "attention_mask": enc["attention_mask"].to(device),
        "ner_labels": ner_labels.to(device),
        "word_ids": [word_ids],
        "gold_entities": [[(hs[0], hs[1], "Method"), (ts[0], ts[1], "Method")]],
        "gold_relations": [[(hs, ts, rel_id)]],
        "num_words": [len(words)],
        "words": [words],
    }
    return batch


@torch.no_grad()
def l_rec_on_synth(
    encoder,
    sentence: str,
    source_triple: Tuple[str, str, str],
    tokenizer,
    device: str,
    max_length: int = 128,
    re_weight: float = 1.0,
    max_loss: float = 4.0,
) -> float:
    """
    Return the scalar encoder NER+RE loss of a sentence against the source
    triple. No grad — this is a reward signal, not a trainable path.

    If span location fails, returns `max_loss` (meaning β term = 0 for
    this sample).
    """
    batch = build_synth_batch(sentence, source_triple, tokenizer, max_length, device)
    if batch is None:
        return max_loss
    encoder.eval()
    try:
        total_loss, _, _, _ = compute_loss(encoder, batch, device, re_weight=re_weight)
    except Exception:
        return max_loss
    return float(total_loss.item())


def l_rec_batch(
    encoder,
    sentences: List[str],
    source_triples: List[Tuple[str, str, str]],
    tokenizer,
    device: str,
    max_length: int = 128,
    max_loss: float = 4.0,
) -> torch.Tensor:
    """
    Vectorized over a batch. Returns (B,) float tensor of L_rec values.
    No gradient. Caller turns this into the β reward:

        rec_reward = clamp(1 − l_rec / L_REC_SCALE, 0, 1)
    """
    assert len(sentences) == len(source_triples)
    out = [
        l_rec_on_synth(encoder, s, t, tokenizer, device, max_length, max_loss=max_loss)
        for s, t in zip(sentences, source_triples)
    ]
    return torch.tensor(out, device=device, dtype=torch.float32)


# ── String containment reward (Stage 2d v3) ─────────────────────────
# Replaces L_rec-based β term. Directly checks whether the decoder's
# output mentions the source entities. More robust than encoder-based
# L_rec, which returns ~4.0 even on successful span matches because the
# frozen encoder can't extract triples from Qwen paraphrases.


def _substr_in(phrase: str, text: str) -> bool:
    """Case-insensitive substring check."""
    return phrase.lower().strip() in text.lower()


def string_containment_reward_single(
    sentence: str,
    source_triple: Tuple[str, str, str],
) -> float:
    """
    Returns 1.0 if both head and tail appear in sentence,
    0.5 if exactly one appears, 0.0 if neither.
    """
    head, _rel, tail = source_triple
    h = _substr_in(head, sentence)
    t = _substr_in(tail, sentence)
    return 0.5 * float(h) + 0.5 * float(t)


def string_containment_batch(
    sentences: List[str],
    source_triples: List[Tuple[str, str, str]],
    device: str,
) -> torch.Tensor:
    """Returns (B,) float tensor of containment rewards in [0, 0.5, 1.0]."""
    assert len(sentences) == len(source_triples)
    out = [
        string_containment_reward_single(s, t)
        for s, t in zip(sentences, source_triples)
    ]
    return torch.tensor(out, device=device, dtype=torch.float32)
