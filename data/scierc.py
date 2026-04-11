"""
SciERC dataset for Stage 2 KG extraction training.

Returns batches with:
    input_ids:       (B, T)         BERT token ids
    attention_mask:  (B, T)         BERT attention mask
    word_ids:        list[list[int|None]]  word index for each subword token
    ner_tags:        (B, T)         BIO tag id per token (per-token NER target)
    triples:         list[list[(h_span, t_span, rel_type_id)]]  per example

Span format: (word_start, word_end_inclusive)
We work at the document level (concatenate all sentences in a doc, since
SciERC's relations are intra-sentence but we collapse for simpler batching).
For Stage 2a we use sentence-level granularity instead — simpler and faster.
"""
import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader

# ── Label vocabulary ─────────────────────────────────────────────────
ENTITY_TYPES = ["Task", "Method", "Metric", "Material", "OtherScientificTerm", "Generic"]
RELATION_TYPES = [
    "USED-FOR", "FEATURE-OF", "HYPONYM-OF", "EVALUATE-FOR",
    "PART-OF", "COMPARE", "CONJUNCTION",
]

# BIO tag set: O + (B-, I-) per entity type → 1 + 2*6 = 13
BIO_TAGS = ["O"] + [f"{p}-{t}" for t in ENTITY_TYPES for p in ("B", "I")]
BIO_TAG2ID = {t: i for i, t in enumerate(BIO_TAGS)}
ID2BIO = {i: t for t, i in BIO_TAG2ID.items()}
NUM_BIO_TAGS = len(BIO_TAGS)

# NO_REL is the 0th class (negative pairs); the 7 named relations follow.
# Stage 2-004 fix: previously NUM_RELATIONS=7 and the loss only saw positive
# pairs, so the head never learned to suppress negatives. Now NUM_RELATIONS=8
# and every ordered pair of gold spans is a training example.
NO_REL_ID = 0
REL2ID = {"NO_REL": NO_REL_ID, **{r: i + 1 for i, r in enumerate(RELATION_TYPES)}}
ID2REL = {i: r for r, i in REL2ID.items()}
NUM_RELATIONS = len(REL2ID)  # 8


def _bio_tags_for_sentence(num_words: int, ner_spans: list) -> list[int]:
    """Convert SciERC ner spans (start, end_inclusive, type) to per-word BIO ids."""
    tags = [BIO_TAG2ID["O"]] * num_words
    for s, e, t in ner_spans:
        if s >= num_words or e >= num_words:
            continue  # malformed
        tags[s] = BIO_TAG2ID[f"B-{t}"]
        for k in range(s + 1, e + 1):
            tags[k] = BIO_TAG2ID[f"I-{t}"]
    return tags


class SciERCSentenceDataset(Dataset):
    """
    Sentence-level SciERC for Stage 2a baseline.

    SciERC's `sentences[i]` is sentence i (list of word strings).
    `ner[i]` and `relations[i]` are aligned per-sentence — but the span
    indices are *document-global*, so we need to subtract the sentence
    offset to get sentence-local indices.
    """
    def __init__(self, json_path, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []  # one entry per sentence

        with open(json_path) as f:
            for line in f:
                doc = json.loads(line)
                sentences = doc["sentences"]
                ner = doc["ner"]
                relations = doc["relations"]

                # Compute per-sentence word offsets
                offsets = [0]
                for s in sentences:
                    offsets.append(offsets[-1] + len(s))

                for sent_idx, words in enumerate(sentences):
                    if not words:
                        continue
                    sent_offset = offsets[sent_idx]
                    sent_len = len(words)

                    # Local NER spans
                    local_ner = []
                    for s, e, t in ner[sent_idx]:
                        ls = s - sent_offset
                        le = e - sent_offset
                        if 0 <= ls < sent_len and 0 <= le < sent_len:
                            local_ner.append((ls, le, t))

                    # Local relations
                    local_rels = []
                    for hs, he, ts, te, rel in relations[sent_idx]:
                        lhs, lhe = hs - sent_offset, he - sent_offset
                        lts, lte = ts - sent_offset, te - sent_offset
                        if all(0 <= x < sent_len for x in [lhs, lhe, lts, lte]):
                            local_rels.append(((lhs, lhe), (lts, lte), REL2ID[rel]))

                    self.examples.append({
                        "words": words,
                        "ner": local_ner,
                        "relations": local_rels,
                    })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        words = ex["words"]

        # Tokenize at word level so we can map back
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
        word_ids = encoding.word_ids()  # list[int|None], length = len(input_ids)

        # Per-token BIO labels (label first subword of each word; -100 for padding/CLS/SEP/continuation)
        word_bio = _bio_tags_for_sentence(len(words), ex["ner"])
        token_labels = []
        prev_word_id = None
        for wid in word_ids:
            if wid is None:
                token_labels.append(-100)  # special token
            elif wid != prev_word_id:
                token_labels.append(word_bio[wid])
            else:
                token_labels.append(-100)  # continuation subword
            prev_word_id = wid

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "word_ids": word_ids,
            "ner_labels": token_labels,
            "gold_entities": ex["ner"],          # list of (s, e, type_str)
            "gold_relations": ex["relations"],   # list of ((hs,he),(ts,te), rel_id)
            "num_words": len(words),
        }


def collate_fn(batch, pad_token_id: int = 0):
    """Pad input_ids/attention_mask/ner_labels to max length in batch."""
    max_len = max(len(b["input_ids"]) for b in batch)
    B = len(batch)

    input_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long)
    ner_labels = torch.full((B, max_len), -100, dtype=torch.long)

    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        input_ids[i, :L] = torch.tensor(b["input_ids"], dtype=torch.long)
        attention_mask[i, :L] = torch.tensor(b["attention_mask"], dtype=torch.long)
        ner_labels[i, :L] = torch.tensor(b["ner_labels"], dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "ner_labels": ner_labels,
        "word_ids": [b["word_ids"] for b in batch],
        "gold_entities": [b["gold_entities"] for b in batch],
        "gold_relations": [b["gold_relations"] for b in batch],
        "num_words": [b["num_words"] for b in batch],
    }


def build_dataloaders(tokenizer, data_dir: Optional[Path] = None, batch_size: int = 16,
                     max_length: int = 128, num_workers: int = 0):
    if data_dir is None:
        data_dir = Path(__file__).parent / "scierc"

    train = SciERCSentenceDataset(data_dir / "train.json", tokenizer, max_length)
    dev = SciERCSentenceDataset(data_dir / "dev.json", tokenizer, max_length)
    test = SciERCSentenceDataset(data_dir / "test.json", tokenizer, max_length)

    pad_id = tokenizer.pad_token_id
    coll = lambda b: collate_fn(b, pad_token_id=pad_id)

    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=coll, num_workers=num_workers),
        DataLoader(dev, batch_size=batch_size, shuffle=False, collate_fn=coll, num_workers=num_workers),
        DataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=coll, num_workers=num_workers),
    )
