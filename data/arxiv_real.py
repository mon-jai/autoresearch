"""
Held-out arXiv real-text corpus for the Stage 2b realism critic.

Provides positive samples ("real scientific text") for the binary
real/fake critic. Use cycles through the corpus indefinitely so a long
training run never starves.

The text comes from `data/download_arxiv_real.py` which truncates each
arXiv paper to its first 1024 characters and saves cs.* documents to
`data/arxiv_real/cs_validation.jsonl`.
"""
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

DEFAULT_PATH = Path(__file__).parent / "arxiv_real" / "cs_validation.jsonl"


class ArxivRealDataset(Dataset):
    def __init__(self, jsonl_path=None, tokenizer=None, max_length=128):
        if jsonl_path is None:
            jsonl_path = DEFAULT_PATH
        if tokenizer is None:
            raise ValueError("tokenizer is required")
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(jsonl_path) as f:
            self.examples = [json.loads(line)["text"] for line in f]
        if not self.examples:
            raise RuntimeError(f"empty arXiv corpus at {jsonl_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }


def collate_real(batch, pad_token_id: int = 0):
    """Pad input_ids/attention_mask to max length in batch."""
    max_len = max(len(b["input_ids"]) for b in batch)
    B = len(batch)
    input_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        input_ids[i, :L] = torch.tensor(b["input_ids"], dtype=torch.long)
        attention_mask[i, :L] = torch.tensor(b["attention_mask"], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def build_arxiv_loader(tokenizer, batch_size: int = 16, max_length: int = 128,
                      jsonl_path=None, num_workers: int = 0):
    ds = ArxivRealDataset(jsonl_path=jsonl_path, tokenizer=tokenizer, max_length=max_length)
    pad_id = tokenizer.pad_token_id
    coll = lambda b: collate_real(b, pad_token_id=pad_id)
    # shuffle=True so the critic sees varied real samples each epoch
    return DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=coll,
                      num_workers=num_workers, drop_last=True)


def cycle(loader):
    """Yield batches forever — used so the train loop never runs out."""
    while True:
        for batch in loader:
            yield batch
