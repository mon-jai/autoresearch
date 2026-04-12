"""
CoNLL04 dataset for multi-dataset validation of Stage 2 pipeline.

Reads SpERT-format JSON (the standard distribution format):
    [{"tokens": [...], "entities": [{type, start, end}, ...],
      "relations": [{type, head, tail}, ...]}, ...]

Returns batches with the same interface as scierc.py:
    input_ids, attention_mask, word_ids, ner_labels,
    gold_entities, gold_relations, num_words, words
"""
import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader

# ── Label vocabulary ─────────────────────────────────────────────────
ENTITY_TYPES = ["Loc", "Org", "Peop", "Other"]
RELATION_TYPES = ["Located_In", "Work_For", "OrgBased_In", "Live_In", "Kill"]

# BIO tag set: O + (B-, I-) per entity type → 1 + 2*4 = 9
BIO_TAGS = ["O"] + [f"{p}-{t}" for t in ENTITY_TYPES for p in ("B", "I")]
BIO_TAG2ID = {t: i for i, t in enumerate(BIO_TAGS)}
ID2BIO = {i: t for t, i in BIO_TAG2ID.items()}
NUM_BIO_TAGS = len(BIO_TAGS)

NO_REL_ID = 0
REL2ID = {"NO_REL": NO_REL_ID, **{r: i + 1 for i, r in enumerate(RELATION_TYPES)}}
ID2REL = {i: r for r, i in REL2ID.items()}
NUM_RELATIONS = len(REL2ID)  # 6


def _bio_tags_for_sentence(num_words: int, entities: list) -> list[int]:
    """Convert entity spans (start, end_exclusive, type_str) to per-word BIO ids."""
    tags = [BIO_TAG2ID["O"]] * num_words
    for s, e, t in entities:
        if s >= num_words or e > num_words:
            continue
        tag_key = f"B-{t}"
        if tag_key not in BIO_TAG2ID:
            continue
        tags[s] = BIO_TAG2ID[tag_key]
        for k in range(s + 1, e):
            tags[k] = BIO_TAG2ID[f"I-{t}"]
    return tags


class CoNLL04SentenceDataset(Dataset):
    """
    Sentence-level CoNLL04 in SpERT JSON format.

    SpERT format per example:
        tokens:    ["John", "works", "at", "Google"]
        entities:  [{"type": "Peop", "start": 0, "end": 1},
                    {"type": "Org",  "start": 3, "end": 4}]
        relations: [{"type": "Work_For", "head": 0, "tail": 1}]
                   (head/tail are indices into the entities list)
    """
    def __init__(self, json_path, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        with open(json_path) as f:
            data = json.load(f)

        for doc in data:
            words = doc["tokens"]
            if not words:
                continue

            # Parse entities: SpERT uses exclusive end
            ner_spans = []
            entity_spans = []  # keep (start, end_inclusive) for relation mapping
            for ent in doc.get("entities", []):
                s, e, t = ent["start"], ent["end"], ent["type"]
                ner_spans.append((s, e, t))  # end is exclusive for BIO conversion
                entity_spans.append((s, e - 1))  # end_inclusive for relation spans

            # Parse relations: head/tail index into entities list
            relations = []
            for rel in doc.get("relations", []):
                h_idx, t_idx = rel["head"], rel["tail"]
                if h_idx >= len(entity_spans) or t_idx >= len(entity_spans):
                    continue
                h_span = entity_spans[h_idx]
                t_span = entity_spans[t_idx]
                rel_id = REL2ID.get(rel["type"])
                if rel_id is None:
                    continue
                relations.append((h_span, t_span, rel_id))

            # Convert NER to (start, end_inclusive, type) for gold_entities output
            gold_ner = [(s, e - 1, t) for s, e, t in ner_spans]

            self.examples.append({
                "words": words,
                "ner": gold_ner,            # (start, end_inclusive, type_str)
                "ner_spans_excl": ner_spans, # (start, end_exclusive, type_str) for BIO
                "relations": relations,      # ((hs,he_incl),(ts,te_incl), rel_id)
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

        # Per-token BIO labels (first subword of each word; -100 for special/continuation)
        word_bio = _bio_tags_for_sentence(len(words), ex["ner_spans_excl"])
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
        "words": [b["words"] for b in batch],
    }


def build_dataloaders(tokenizer, data_dir: Optional[Path] = None, batch_size: int = 16,
                     max_length: int = 128, num_workers: int = 0):
    if data_dir is None:
        data_dir = Path(__file__).parent / "conll04"

    train = CoNLL04SentenceDataset(data_dir / "conll04_train.json", tokenizer, max_length)
    dev = CoNLL04SentenceDataset(data_dir / "conll04_dev.json", tokenizer, max_length)
    test = CoNLL04SentenceDataset(data_dir / "conll04_test.json", tokenizer, max_length)

    pad_id = tokenizer.pad_token_id
    coll = lambda b: collate_fn(b, pad_token_id=pad_id)

    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=coll, num_workers=num_workers),
        DataLoader(dev, batch_size=batch_size, shuffle=False, collate_fn=coll, num_workers=num_workers),
        DataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=coll, num_workers=num_workers),
    )
