"""
Stage 2e: load the LoRA-generated synth dataset (jsonl) and produce
SciERC-compatible batches for encoder training.

Each synth example has:
    - synth_sentence (string)
    - head, rel, tail (source triple)
    - containment (0.0 / 0.5 / 1.0)

We tokenize the sentence word-by-word, locate head/tail spans via
substring matching, and produce NER labels (BIO) + a single gold
relation — the same batch format that `compute_loss()` expects.
"""
import json
from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset, DataLoader


def _find_span_in_words(words, phrase):
    """Find contiguous span matching phrase (case-insensitive). Returns (start, end_inclusive) or None."""
    target_tokens = phrase.lower().strip().split()
    if not target_tokens:
        return None
    tl = len(target_tokens)
    lwords = [w.lower() for w in words]
    for i in range(len(lwords) - tl + 1):
        if lwords[i : i + tl] == target_tokens:
            return (i, i + tl - 1)
    return None


class SynthDataset(Dataset):
    """
    Reads the jsonl from generate_synth_dataset.py and presents it in the
    same format as SciERCSentenceDataset.__getitem__().
    """
    def __init__(self, jsonl_path, tokenizer, max_length: int = 128,
                 ds_mod=None, entity_type: str = None,
                 min_containment: float = 0.0):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ds_mod = ds_mod
        self.entity_type = entity_type or ds_mod.ENTITY_TYPES[0]
        self.examples = []

        with open(jsonl_path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("containment", 1.0) < min_containment:
                    continue
                words = rec["synth_sentence"].strip().split()
                if len(words) < 3:
                    continue
                head_span = _find_span_in_words(words, rec["head"])
                tail_span = _find_span_in_words(words, rec["tail"])
                if head_span is None or tail_span is None:
                    continue
                rel_id = rec.get("rel_id")
                if rel_id is None:
                    rel_id = ds_mod.REL2ID.get(rec["rel"], ds_mod.NO_REL_ID)
                if rel_id == ds_mod.NO_REL_ID:
                    continue
                # Use the source entity type from the jsonl if available;
                # fall back to the constructor default (Fix 1 for stage2-010).
                h_type = rec.get("entity_type", self.entity_type)
                # Tail type isn't stored separately; use head type as proxy
                # (SciERC triples often share the same type for both spans).
                t_type = rec.get("tail_entity_type", h_type)
                self.examples.append({
                    "words": words,
                    "ner": [(head_span[0], head_span[1], h_type),
                            (tail_span[0], tail_span[1], t_type)],
                    "relations": [(head_span, tail_span, rel_id)],
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        words = ex["words"]

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        word_ids = encoding.word_ids()

        # BIO labels from ner spans
        word_bio = self.ds_mod._bio_tags_for_sentence(len(words), ex["ner"])
        token_labels = []
        prev_word_id = None
        for wid in word_ids:
            if wid is None:
                token_labels.append(-100)
            elif wid != prev_word_id:
                token_labels.append(word_bio[wid])
            else:
                token_labels.append(-100)
            prev_word_id = wid

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "word_ids": word_ids,
            "ner_labels": token_labels,
            "gold_entities": ex["ner"],
            "gold_relations": ex["relations"],
            "num_words": len(words),
            "words": words,
        }


def build_synth_loader(tokenizer, jsonl_path, batch_size: int = 16,
                       max_length: int = 128, num_workers: int = 0,
                       min_containment: float = 0.0, ds_mod=None):
    if ds_mod is None:
        from data import scierc as ds_mod
    ds = SynthDataset(jsonl_path, tokenizer, max_length, ds_mod=ds_mod,
                      min_containment=min_containment)
    pad_id = tokenizer.pad_token_id or 0
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=lambda b: ds_mod.collate_fn(b, pad_token_id=pad_id),
    )
    return loader
