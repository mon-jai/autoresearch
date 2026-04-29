"""
Generate dataset-aware relation replay JSONL from gold train triples.

The output is compatible with data/synth_loader.py. Unlike free-form LLM
augmentation, relation replay keeps the original sentence text and gold
head/tail spans, but presents each positive relation as a focused training
example. This tests whether ACCORD needs more relation exposure rather than
new loss/head tricks.
"""
import argparse
import importlib
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

from train_span import DATASET_REGISTRY


DEFAULT_MODEL = {
    "accord": "microsoft/deberta-base",
    "scierc": "allenai/scibert_scivocab_uncased",
    "scier": "allenai/scibert_scivocab_uncased",
    "conll04": "bert-base-uncased",
    "ade": "bert-base-uncased",
    "cuad": "bert-base-uncased",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="accord", choices=list(DATASET_REGISTRY))
    p.add_argument("--model-name", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-examples", type=int, default=0,
                   help="0 = all positive train relations after dev split.")
    p.add_argument("--copies", type=int, default=1,
                   help="Repeat each relation this many times.")
    return p.parse_args()


def iter_train_relations(train_loader, ds_mod):
    for batch in train_loader:
        for words, ents, rels in zip(
            batch["words"], batch["gold_entities"], batch["gold_relations"]
        ):
            type_by_span = {(s, e): t for (s, e, t) in ents}
            for h_span, t_span, rel_id in rels:
                if rel_id == ds_mod.NO_REL_ID:
                    continue
                hs, he = h_span
                ts, te = t_span
                yield {
                    "synth_sentence": " ".join(words),
                    "head": " ".join(words[hs:he + 1]),
                    "tail": " ".join(words[ts:te + 1]),
                    "rel": ds_mod.ID2REL[int(rel_id)],
                    "rel_id": int(rel_id),
                    "entity_type": type_by_span.get(h_span, ds_mod.ENTITY_TYPES[0]),
                    "tail_entity_type": type_by_span.get(t_span, ds_mod.ENTITY_TYPES[0]),
                    "containment": 1.0,
                    "source_sentence": " ".join(words),
                }


def main():
    args = parse_args()
    random.seed(args.seed)
    ds_mod = importlib.import_module(DATASET_REGISTRY[args.dataset])
    model_name = args.model_name or DEFAULT_MODEL.get(args.dataset, "bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_dir = Path(args.data_dir) if args.data_dir else None
    train_loader, _, _ = ds_mod.build_dataloaders(
        tokenizer, data_dir=data_dir, batch_size=args.batch_size,
        max_length=args.max_length, seed=args.seed,
    )

    records = list(iter_train_relations(train_loader, ds_mod))
    if args.max_examples > 0:
        records = records[:args.max_examples]
    records = records * max(args.copies, 1)
    random.shuffle(records)

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fout:
        for rec in records:
            fout.write(json.dumps(rec) + "\n")

    print(f"dataset={args.dataset} train_relation_records={len(records)}")
    print(f"wrote={out_path}")


if __name__ == "__main__":
    main()
