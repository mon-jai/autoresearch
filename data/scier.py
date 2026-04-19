"""
SciER dataset (EMNLP 2024) — 106 full-text scientific publications.
24K entities, 12K relations. 3 entity types, 9 relation types.

Format compatible with SciERC pipeline (same collate_fn, same batch structure).

Reference: Zhang et al., "SciER: An Entity and Relation Extraction Dataset
for Datasets, Methods, and Tasks in Scientific Documents", EMNLP 2024.
"""
import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader

# ── Label vocabulary ─────────────────────────────────────────────────
ENTITY_TYPES = ["Dataset", "Method", "Task"]
RELATION_TYPES = [
    "Used-For", "Part-Of", "Compare-With", "Evaluated-With",
    "Trained-With", "SubClass-Of", "SubTask-Of", "Synonym-Of", "Benchmark-For",
]

# BIO tag set: O + (B-, I-) per entity type → 1 + 2*3 = 7
BIO_TAGS = ["O"] + [f"{p}-{t}" for t in ENTITY_TYPES for p in ("B", "I")]
BIO_TAG2ID = {t: i for i, t in enumerate(BIO_TAGS)}
ID2BIO = {i: t for t, i in BIO_TAG2ID.items()}
NUM_BIO_TAGS = len(BIO_TAGS)

NO_REL_ID = 0
REL2ID = {"NO_REL": NO_REL_ID, **{r: i + 1 for i, r in enumerate(RELATION_TYPES)}}
ID2REL = {i: r for r, i in REL2ID.items()}
NUM_RELATIONS = len(REL2ID)  # 10


def _bio_tags_for_sentence(num_words: int, ner_spans: list) -> list:
    """Convert ner spans (start, end_inclusive, type) to per-word BIO ids."""
    tags = [BIO_TAG2ID["O"]] * num_words
    for s, e, t in ner_spans:
        if s >= num_words or e >= num_words:
            continue
        tag_b = f"B-{t}"
        tag_i = f"I-{t}"
        if tag_b not in BIO_TAG2ID:
            continue
        tags[s] = BIO_TAG2ID[tag_b]
        for k in range(s + 1, e + 1):
            tags[k] = BIO_TAG2ID[tag_i]
    return tags


class SciERSentenceDataset(Dataset):
    """
    Sentence-level SciER for span NER+RE training.

    SciER PLM format uses document-level indices in ner/relations,
    aligned per-sentence (same as SciERC).
    """
    def __init__(self, jsonl_path, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        with open(jsonl_path) as f:
            for line in f:
                doc = json.loads(line)
                sentences = doc["sentences"]
                ner = doc["ner"]
                relations = doc["relations"]

                offsets = [0]
                for s in sentences:
                    offsets.append(offsets[-1] + len(s))

                for sent_idx, words in enumerate(sentences):
                    if not words:
                        continue
                    sent_offset = offsets[sent_idx]
                    sent_len = len(words)

                    local_ner = []
                    for s, e, t in ner[sent_idx]:
                        ls = s - sent_offset
                        le = e - sent_offset
                        if 0 <= ls < sent_len and 0 <= le < sent_len:
                            local_ner.append((ls, le, t))

                    local_rels = []
                    for hs, he, ts, te, rel in relations[sent_idx]:
                        lhs, lhe = hs - sent_offset, he - sent_offset
                        lts, lte = ts - sent_offset, te - sent_offset
                        if all(0 <= x < sent_len for x in [lhs, lhe, lts, lte]):
                            if rel in REL2ID:
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

        word_bio = _bio_tags_for_sentence(len(words), ex["ner"])
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
    """Pad batch to max length. Same as SciERC collate_fn."""
    max_len = max(len(b["input_ids"]) for b in batch)
    B = len(batch)
    input_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long)
    ner_labels = torch.full((B, max_len), -100, dtype=torch.long)

    word_ids_list = []
    gold_entities_list = []
    gold_relations_list = []
    num_words_list = []
    words_list = []

    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        input_ids[i, :L] = torch.tensor(b["input_ids"], dtype=torch.long)
        attention_mask[i, :L] = torch.tensor(b["attention_mask"], dtype=torch.long)
        ner_labels[i, :L] = torch.tensor(b["ner_labels"], dtype=torch.long)
        word_ids_list.append(b["word_ids"])
        gold_entities_list.append(b["gold_entities"])
        gold_relations_list.append(b["gold_relations"])
        num_words_list.append(b["num_words"])
        words_list.append(b["words"])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "ner_labels": ner_labels,
        "word_ids": word_ids_list,
        "gold_entities": gold_entities_list,
        "gold_relations": gold_relations_list,
        "num_words": num_words_list,
        "words": words_list,
    }


DEFAULT_DATA_DIR = Path(__file__).parent / "scier_dataset" / "SciER" / "PLM"


def build_dataloaders(tokenizer, data_dir=None, batch_size: int = 16,
                      max_length: int = 128, num_workers: int = 0):
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    pad_id = tokenizer.pad_token_id

    def make_loader(split, shuffle):
        ds = SciERSentenceDataset(data_dir / f"{split}.jsonl", tokenizer, max_length)
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
            collate_fn=lambda b: collate_fn(b, pad_token_id=pad_id),
        )

    return make_loader("train", True), make_loader("dev", False), make_loader("test", False)
