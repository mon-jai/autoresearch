"""
Span-based Triple F1 evaluation for BertKGExtractor checkpoints.

Uses the span NER head (forward_span_ner) rather than the BIO head, matching
the evaluation logic used during train_span.py training (evaluate_span).

Standalone CLI:
    uv run python eval/span_f1.py \\
        --checkpoint checkpoints/train_span_scierc_scibert_bio_mt_s42_best.pt \\
        --dataset scierc --split test --seed 42

Module usage:
    from eval.span_f1 import evaluate_span, load_span_model
"""
import argparse
import importlib
import inspect
import sys
from pathlib import Path

import torch

from eval.triple_f1 import _prf
from models.bert_kg_encoder import BertKGExtractor


DATASET_REGISTRY = {
    "scierc": "data.scierc",
    "scier": "data.scier",
    "conll04": "data.conll04",
    "ade": "data.ade",
    "accord": "data.code_accord",
    "cuad": "data.cuad",
}

DEFAULT_MODEL = {
    "scierc": "allenai/scibert_scivocab_uncased",
    "scier": "allenai/scibert_scivocab_uncased",
    "conll04": "bert-base-uncased",
    "ade": "allenai/scibert_scivocab_uncased",
    "accord": "bert-base-uncased",
    "cuad": "microsoft/deberta-large",
}


def evaluate_span(model, dataloader, device, ds_mod, entity_type2id, id2entity_type,
                  max_span_width=8, span_threshold=0.5, verbose=False,
                  span_proposal=False, span_proposal_expand=1):
    """
    Evaluate with span-based NER predictions feeding into RE.

    Mirrors train_span.py::evaluate_span; extracted here for standalone use.
    Returns {"ner_f1", "triple_f1", "n_examples"}.
    """
    model.eval()
    NO_REL = ds_mod.NO_REL_ID

    ner_tp = ner_fp = ner_fn = 0
    triple_tp = triple_fp = triple_fn = 0
    n_examples = 0
    total_pred_ents = total_gold_ents = 0
    total_pred_rels = total_gold_rels = total_re_pairs = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            word_ids_list = batch["word_ids"]
            gold_entities_list = batch["gold_entities"]
            gold_relations_list = batch["gold_relations"]
            num_words_list = batch["num_words"]

            hidden = model.encode(modality="text", input_ids=input_ids,
                                  attention_mask=attention_mask)

            bio_logits = None
            if model.bio_enrich != "none" or span_proposal:
                bio_logits = model.forward_ner(hidden)

            for b_idx in range(input_ids.size(0)):
                n_examples += 1
                n_words = num_words_list[b_idx]
                gold_ents = gold_entities_list[b_idx]
                gold_rels = gold_relations_list[b_idx]
                bio_logits_b = bio_logits[b_idx] if bio_logits is not None else None

                bio_props = None
                if span_proposal and bio_logits_b is not None:
                    bio_props = BertKGExtractor.bio_guided_proposals(
                        bio_logits_b, word_ids_list[b_idx], n_words,
                        expand=span_proposal_expand,
                    )

                use_breg = hasattr(model, "boundary_reg") and model.boundary_reg
                result = model.forward_span_ner(
                    hidden[b_idx], word_ids_list[b_idx], n_words, max_span_width,
                    bio_logits_b=bio_logits_b, bio_proposals=bio_props,
                )
                if use_breg:
                    span_logits, candidates, boundary_offsets = result
                else:
                    span_logits, candidates = result
                    boundary_offsets = None

                total_gold_ents += len(gold_ents)
                gold_full = {(h, t, r) for (h, t, r) in gold_rels}
                total_gold_rels += len(gold_full)

                if not candidates:
                    ner_fn += len(gold_ents)
                    triple_fn += len(gold_full)
                    continue

                span_probs = torch.softmax(span_logits, dim=-1)
                pred_types = span_logits.argmax(dim=-1).tolist()
                pred_confs = span_probs.max(dim=-1).values.tolist()

                pred_spans = []
                for idx_c, ((s, e), etype_id, conf) in enumerate(
                        zip(candidates, pred_types, pred_confs)):
                    if etype_id > 0 and conf >= span_threshold:
                        if boundary_offsets is not None:
                            ds_ = round(boundary_offsets[idx_c, 0].item())
                            de_ = round(boundary_offsets[idx_c, 1].item())
                            s = max(0, min(s + ds_, n_words - 1))
                            e = max(s, min(e + de_, n_words - 1))
                        etype = id2entity_type.get(etype_id, "Unknown")
                        pred_spans.append((s, e, etype, conf))

                scored = sorted(pred_spans, key=lambda x: -x[3])
                taken = set()
                filtered = []
                for (s, e, t, c) in scored:
                    overlap = any(not (e < ts or te < s) for (ts, te) in taken)
                    if not overlap:
                        filtered.append((s, e, t))
                        taken.add((s, e))
                pred_spans = filtered
                total_pred_ents += len(pred_spans)

                pred_ent_set = {(s, e, t) for (s, e, t) in pred_spans}
                gold_ent_set = {(s, e, t) for (s, e, t) in gold_ents}
                ner_tp += len(pred_ent_set & gold_ent_set)
                ner_fp += len(pred_ent_set - gold_ent_set)
                ner_fn += len(gold_ent_set - pred_ent_set)

                pred_span_list = [(s, e) for (s, e, _) in pred_spans]
                pred_pairs = [(a, b) for a in pred_span_list for b in pred_span_list if a != b]
                total_re_pairs += len(pred_pairs)
                if pred_pairs:
                    re_logits = model.forward_re(hidden[b_idx], word_ids_list[b_idx], pred_pairs)
                    pred_re_ids = re_logits.argmax(dim=-1).tolist()
                    pred_full = {(h, t, p) for (h, t), p in zip(pred_pairs, pred_re_ids) if p != NO_REL}
                else:
                    pred_full = set()
                total_pred_rels += len(pred_full)
                triple_tp += len(pred_full & gold_full)
                triple_fp += len(pred_full - gold_full)
                triple_fn += len(gold_full - pred_full)

    _, _, nf = _prf(ner_tp, ner_fp, ner_fn)
    tp_, tr_, tf_ = _prf(triple_tp, triple_fp, triple_fn)

    if verbose:
        ner_p = ner_tp / max(ner_tp + ner_fp, 1)
        ner_r = ner_tp / max(ner_tp + ner_fn, 1)
        tri_p = triple_tp / max(triple_tp + triple_fp, 1)
        tri_r = triple_tp / max(triple_tp + triple_fn, 1)
        no_rel_frac = 1.0 - total_pred_rels / max(total_re_pairs, 1)
        print(f"  NER  P={ner_p:.4f} R={ner_r:.4f} F1={nf:.4f}")
        print(f"  Triple P={tri_p:.4f} R={tri_r:.4f} F1={tf_:.4f} "
              f"(pred_ents={total_pred_ents} gold_ents={total_gold_ents} "
              f"NO_REL%={no_rel_frac:.2f})")
    return {
        "ner_p": ner_tp / max(ner_tp + ner_fp, 1),
        "ner_r": ner_tp / max(ner_tp + ner_fn, 1),
        "ner_f1": nf,
        "triple_p": tp_,
        "triple_r": tr_,
        "triple_f1": tf_,
        "n_examples": n_examples,
    }


def load_span_model(checkpoint_path: str, model_name: str, ds_mod, device):
    """
    Load a BertKGExtractor checkpoint and return (model, entity_type2id, id2entity_type).
    Raises ValueError if the checkpoint does not contain a span NER head.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["encoder"] if "encoder" in ckpt else ckpt

    has_span_head = any(k.startswith("span_ner_head.") for k in state)
    if not has_span_head:
        raise ValueError(
            f"Checkpoint {checkpoint_path!r} has no span_ner_head — "
            "use eval/triple_f1.py main() for BIO-based evaluation."
        )

    n_bio = ds_mod.NUM_BIO_TAGS
    n_rel = ds_mod.NUM_RELATIONS
    n_ent = len(ds_mod.ENTITY_TYPES)
    max_span_width = 8

    # Infer max_span_width from span_width_emb if present
    if "span_width_emb.weight" in state:
        max_span_width = state["span_width_emb.weight"].shape[0]

    # Detect re_context_span from re_head input dimension
    re_head_w = state.get("re_head.0.weight")
    re_context_span = False
    if re_head_w is not None:
        from models.bert_kg_encoder import BertBackbone
        tmp = BertBackbone(model_name)
        h = tmp.hidden_size
        del tmp
        re_context_span = (re_head_w.shape[1] == h * 3)

    model = BertKGExtractor(
        model_name,
        num_bio_tags=n_bio,
        num_relations=n_rel,
        num_entity_types=n_ent,
        use_span_ner=True,
        max_span_width=max_span_width,
    ).to(device)
    model.re_context_span = re_context_span
    if re_context_span:
        from models.bert_kg_encoder import BertBackbone
        h = BertKGExtractor(model_name).backbone.hidden_size
        import torch.nn as nn
        model.re_head = nn.Sequential(
            nn.Linear(h * 3, h), nn.GELU(), nn.Dropout(0.1), nn.Linear(h, n_rel)
        ).to(device)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  [warn] unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    model.eval()

    entity_type2id = {t: i + 1 for i, t in enumerate(ds_mod.ENTITY_TYPES)}
    id2entity_type = {v: k for k, v in entity_type2id.items()}
    return model, entity_type2id, id2entity_type


def main():
    p = argparse.ArgumentParser(
        description="Span-based Triple F1 evaluation for BertKGExtractor checkpoints.")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--dataset", default="scierc", choices=list(DATASET_REGISTRY.keys()))
    p.add_argument("--model-name", default=None,
                   help="HuggingFace model ID (must match training backbone)")
    p.add_argument("--split", default="test", choices=["train", "dev", "test"])
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for datasets with runtime train/dev splits (accord, cuad)")
    p.add_argument("--max-span-width", type=int, default=8)
    p.add_argument("--span-threshold", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.model_name is None:
        args.model_name = DEFAULT_MODEL.get(args.dataset, "bert-base-uncased")

    ds_mod = importlib.import_module(DATASET_REGISTRY[args.dataset])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    dl_kwargs = dict(batch_size=args.batch_size, max_length=args.max_length)
    dl_params = inspect.signature(ds_mod.build_dataloaders).parameters
    if "seed" in dl_params:
        dl_kwargs["seed"] = args.seed
    train_loader, dev_loader, test_loader = ds_mod.build_dataloaders(tokenizer, **dl_kwargs)
    loader = {"train": train_loader, "dev": dev_loader, "test": test_loader}[args.split]

    print(f"Loading checkpoint: {args.checkpoint}")
    model, entity_type2id, id2entity_type = load_span_model(
        args.checkpoint, args.model_name, ds_mod, device)

    print(f"Evaluating {args.split} split ({len(loader.dataset)} examples)...")
    metrics = evaluate_span(
        model, loader, device, ds_mod,
        entity_type2id, id2entity_type,
        max_span_width=args.max_span_width,
        span_threshold=args.span_threshold,
        verbose=True,
    )
    print(f"\n{args.split.upper()} NER F1={metrics['ner_f1']:.4f}  "
          f"Triple F1={metrics['triple_f1']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
