"""
CUAD (Contract Understanding Atticus Dataset) — contract clause NER + RE.
Converts SQuAD-style QA annotations to span NER + co-occurrence relations.

41 clause types are grouped into 8 entity categories.
Relations: clause co-occurrence within the same sentence/paragraph.

Download: git clone https://github.com/TheAtticusProject/cuad.git data/cuad_repo
Then unzip data/cuad_repo/data.zip.
"""
import json
import re
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader


# ── Entity categories (41 clause types → 8 groups) ──────────────────
CLAUSE_TO_CATEGORY = {
    # Termination
    "Termination For Convenience": "Termination",
    "Post-Termination Services": "Termination",
    "Notice Period To Terminate Renewal": "Termination",
    "Renewal Term": "Termination",
    "Expiration Date": "Termination",
    # IP
    "Ip Ownership Assignment": "IP",
    "Joint Ip Ownership": "IP",
    "Source Code Escrow": "IP",
    "License Grant": "IP",
    "Irrevocable Or Perpetual License": "IP",
    "Non-Transferable License": "IP",
    "Unlimited/All-You-Can-Eat-License": "IP",
    "Affiliate License-Licensee": "IP",
    "Affiliate License-Licensor": "IP",
    # Payment
    "Revenue/Profit Sharing": "Payment",
    "Price Restrictions": "Payment",
    "Liquidated Damages": "Payment",
    "Minimum Commitment": "Payment",
    # Liability
    "Cap On Liability": "Liability",
    "Uncapped Liability": "Liability",
    "Warranty Duration": "Liability",
    # Insurance
    "Insurance": "Insurance",
    # Compliance
    "Governing Law": "Compliance",
    "Anti-Assignment": "Compliance",
    "Audit Rights": "Compliance",
    "Most Favored Nation": "Compliance",
    "Covenant Not To Sue": "Compliance",
    "Third Party Beneficiary": "Compliance",
    "Rofr/Rofo/Rofn": "Compliance",
    # Change
    "Change Of Control": "Change",
    "Non-Compete": "Change",
    "Non-Disparagement": "Change",
    "No-Solicit Of Customers": "Change",
    "No-Solicit Of Employees": "Change",
    "Competitive Restriction Exception": "Change",
    "Exclusivity": "Change",
    "Volume Restriction": "Change",
    # Other (metadata/dates/names)
    "Document Name": "Other",
    "Parties": "Other",
    "Agreement Date": "Other",
    "Effective Date": "Other",
}

ENTITY_TYPES = ["Termination", "IP", "Payment", "Liability",
                "Insurance", "Compliance", "Change", "Other"]

# BIO tag set: O + (B-, I-) per entity type → 1 + 2*8 = 17
BIO_TAGS = ["O"] + [f"{p}-{t}" for t in ENTITY_TYPES for p in ("B", "I")]
BIO_TAG2ID = {t: i for i, t in enumerate(BIO_TAGS)}
ID2BIO = {i: t for t, i in BIO_TAG2ID.items()}
NUM_BIO_TAGS = len(BIO_TAGS)

# Relations: co-occurrence within same sentence
RELATION_TYPES = ["co-occurs"]
NO_REL_ID = 0
REL2ID = {"NO_REL": NO_REL_ID, "co-occurs": 1}
ID2REL = {i: r for r, i in REL2ID.items()}
NUM_RELATIONS = len(REL2ID)  # 2


def _extract_clause_type(question: str) -> Optional[str]:
    """Extract clause type from CUAD question string."""
    m = re.search(r'"(.+?)"', question)
    return m.group(1) if m else None


def _char_to_word_offset(text: str, char_offset: int) -> int:
    """Convert character offset to word offset in whitespace-tokenized text."""
    words_before = text[:char_offset].split()
    return len(words_before)


def _segment_contract(context: str, min_words: int = 10, max_words: int = 100):
    """Split contract into sentence-level segments.

    Returns list of (segment_text, segment_words, char_start, char_end).
    """
    # Split by double newlines first (paragraph breaks)
    paragraphs = re.split(r'\n\s*\n', context)

    segments = []
    running_char = 0

    for para in paragraphs:
        # Track paragraph position in original context
        para_start = context.find(para, running_char)
        if para_start < 0:
            para_start = running_char
        running_char = para_start + len(para)

        para = para.strip()
        if not para:
            continue

        # Split paragraph into sentences
        sentences = re.split(r'(?<=[.;])\s+', para)
        sent_char = para_start

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            sent_start = context.find(sent, sent_char)
            if sent_start < 0:
                sent_start = sent_char
            sent_char = sent_start + len(sent)

            words = sent.split()
            if len(words) < min_words:
                continue

            # Truncate very long sentences
            if len(words) > max_words:
                # Take first max_words
                truncated = " ".join(words[:max_words])
                segments.append((
                    truncated, words[:max_words],
                    sent_start, sent_start + len(truncated),
                ))
            else:
                segments.append((sent, words, sent_start, sent_start + len(sent)))

    return segments


def _build_examples_from_contract(contract_data: dict, max_span_width: int = 8):
    """Convert one CUAD contract into sentence-level NER+RE examples."""
    examples = []

    for para in contract_data["paragraphs"]:
        context = para["context"]
        segments = _segment_contract(context)
        if not segments:
            continue

        # Collect all QA answers with their clause categories
        clause_spans = []  # (char_start, char_end, category, answer_text)
        for qa in para["qas"]:
            if qa.get("is_impossible"):
                continue
            clause_type = _extract_clause_type(qa["question"])
            if not clause_type:
                continue
            category = CLAUSE_TO_CATEGORY.get(clause_type, "Other")

            for ans in qa["answers"]:
                char_start = ans["answer_start"]
                char_end = char_start + len(ans["text"])
                clause_spans.append((char_start, char_end, category, ans["text"]))

        if not clause_spans:
            continue

        # For each segment, find overlapping clause spans
        for seg_text, seg_words, seg_char_start, seg_char_end in segments:
            seg_entities = []  # (word_start, word_end_inclusive, category)

            for cs, ce, cat, ans_text in clause_spans:
                # Check overlap with segment
                overlap_start = max(cs, seg_char_start)
                overlap_end = min(ce, seg_char_end)
                if overlap_start >= overlap_end:
                    continue

                # Convert to word offsets within segment
                # Find word index of overlap_start relative to segment
                prefix = context[seg_char_start:overlap_start]
                ws = len(prefix.split()) if prefix.strip() else 0

                overlap_text = context[overlap_start:overlap_end]
                overlap_words = overlap_text.split()
                n_overlap_words = len(overlap_words)

                if n_overlap_words == 0:
                    continue

                # Clamp span to max_span_width
                we = min(ws + min(n_overlap_words, max_span_width) - 1,
                         len(seg_words) - 1)
                if ws > we or ws >= len(seg_words):
                    continue

                seg_entities.append((ws, we, cat))

            if not seg_entities:
                continue

            # Deduplicate entities (same span, same type)
            seg_entities = list(set(seg_entities))

            # Build co-occurrence relations between all entity pairs
            relations = []
            for i in range(len(seg_entities)):
                for j in range(len(seg_entities)):
                    if i != j:
                        h = (seg_entities[i][0], seg_entities[i][1])
                        t = (seg_entities[j][0], seg_entities[j][1])
                        if h != t:
                            relations.append((h, t, REL2ID["co-occurs"]))

            # Deduplicate relations
            relations = list(set(relations))

            examples.append({
                "words": seg_words,
                "ner": seg_entities,
                "relations": relations,
            })

    return examples


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
            if k < num_words:
                tags[k] = BIO_TAG2ID[tag_i]
    return tags


class CUADDataset(Dataset):
    """Sentence-level CUAD for span NER+RE training."""

    def __init__(self, examples: list, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = examples

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
    """Pad batch to max length. Same as CODE-ACCORD collate_fn."""
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


# Data directory: expects cuad_repo/ with CUADv1.json, train_separate_questions.json, test.json
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cuad_repo"


def build_dataloaders(tokenizer, data_dir=None, batch_size: int = 16,
                      max_length: int = 128, num_workers: int = 0,
                      dev_ratio: float = 0.15, seed: int = 42):
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR

    # Load train and test splits
    train_json = data_dir / "train_separate_questions.json"
    test_json = data_dir / "test.json"

    if not train_json.exists():
        raise FileNotFoundError(
            f"CUAD data not found at {data_dir}. Download with:\n"
            "  git clone https://github.com/TheAtticusProject/cuad.git data/cuad_repo\n"
            "  cd data/cuad_repo && unzip data.zip"
        )

    with open(train_json) as f:
        train_data = json.load(f)
    with open(test_json) as f:
        test_data = json.load(f)

    # Convert to NER+RE examples
    train_examples = []
    for contract in train_data["data"]:
        train_examples.extend(_build_examples_from_contract(contract))

    test_examples = []
    for contract in test_data["data"]:
        test_examples.extend(_build_examples_from_contract(contract))

    # Create dev split from training data
    rng = random.Random(seed)
    rng.shuffle(train_examples)
    n_dev = int(len(train_examples) * dev_ratio)
    dev_examples = train_examples[:n_dev]
    train_examples = train_examples[n_dev:]

    pad_id = tokenizer.pad_token_id

    # Stats
    train_ents = sum(len(e["ner"]) for e in train_examples)
    train_rels = sum(len(e["relations"]) for e in train_examples)
    print(f"  CUAD: train={len(train_examples)} dev={len(dev_examples)} test={len(test_examples)}")
    print(f"  Train entities: {train_ents}, relations: {train_rels}")
    print(f"  Dev entities: {sum(len(e['ner']) for e in dev_examples)}")
    print(f"  Entity categories: {ENTITY_TYPES}")

    def make_loader(examples, shuffle):
        ds = CUADDataset(examples, tokenizer, max_length)
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
            collate_fn=lambda b: collate_fn(b, pad_token_id=pad_id),
        )

    return make_loader(train_examples, True), make_loader(dev_examples, False), make_loader(test_examples, False)
