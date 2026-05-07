"""
Multi-dataset training with span-based NER head.

The span NER head classifies all candidate (start, end) word spans as
entity_type or NONE. This replaces the BIO token-classification approach
(stage2-001 through stage2-023) with a span enumeration approach similar
to SpERT (Eberts & Ulges, 2020).

The RE head is unchanged — it still takes pairs of predicted entity
spans and classifies relations.

Usage:
    uv run python train_span.py --dataset scierc --max-steps 1500
    uv run python train_span.py --dataset conll04 --max-steps 1500
"""
import argparse
import importlib
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from models.bert_kg_encoder import BertKGExtractor


DATASET_REGISTRY = {
    "scierc": "data.scierc",
    "scier": "data.scier",
    "conll04": "data.conll04",
    "ade": "data.ade",
    "accord": "data.code_accord",
    "cuad": "data.cuad",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="scierc",
                   choices=list(DATASET_REGISTRY.keys()))
    p.add_argument("--model-name", default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--warmup-steps", type=int, default=250)
    p.add_argument("--max-span-width", type=int, default=8)
    p.add_argument("--re-weight", type=float, default=1.0)
    p.add_argument("--neg-sample-ratio", type=float, default=0.5,
                   help="Ratio of negative spans to positive spans for NER training. "
                        "0.5 = half as many negatives as positives.")
    p.add_argument("--focal-gamma", type=float, default=2.0,
                   help="Focal loss gamma. 0 = standard CE. 2.0 = recommended for imbalance.")
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-best-to", default=None)
    p.add_argument("--synth-jsonl", default="",
                   help="Path to synth/cycle JSONL. Empty = gold only. "
                        "Used for both CAST pseudo-labels and CycleGT round-trip data.")
    p.add_argument("--relation-replay", action="store_true",
                   help="Build a synth loader from positive train relations in the "
                        "current process's train split. This avoids dev leakage from "
                        "separately generated replay JSONL files.")
    p.add_argument("--relation-replay-copies", type=int, default=1)
    p.add_argument("--cycle-jsonl", default="", dest="cycle_jsonl_alias",
                   help="Alias for --synth-jsonl (CycleGT round-trip data).")
    p.add_argument("--synth-weight", type=float, default=0.3)
    p.add_argument("--cycle-weight", type=float, default=None,
                   help="Alias for --synth-weight (cycle consistency weight).")
    p.add_argument("--gold-only-steps", type=int, default=500,
                   help="Train on gold-only for this many steps before mixing synth.")
    p.add_argument("--pretrain-ckpt", default="",
                   help="Path to ELECTRA cooperative pre-training checkpoint. "
                        "Loads discriminator backbone weights, reinitializes task heads.")
    p.add_argument("--cl-weight", type=float, default=0.0,
                   help="Weight for supervised contrastive (InfoNCE) loss on span embeddings. "
                        "0 = disabled (default). Recommended: 0.05-0.2.")
    p.add_argument("--cl-tau", type=float, default=0.1,
                   help="Temperature for contrastive loss. Lower = sharper.")
    p.add_argument("--cl-entity-only", action="store_true",
                   help="Only use entity spans (not NONE) for contrastive loss.")
    p.add_argument("--bio-weight", type=float, default=0.0,
                   help="Weight for auxiliary BIO NER loss (multi-task). "
                        "0 = disabled. STSN (2024) shows BIO labels improve span reps.")
    p.add_argument("--bio-start", type=float, default=0.0,
                   help="If >0, use curriculum: start bio_weight at this value and "
                        "linearly decay to --bio-end. Overrides --bio-weight.")
    p.add_argument("--bio-end", type=float, default=0.0,
                   help="End bio_weight for curriculum decay (used with --bio-start).")
    p.add_argument("--bio-enrich", default="none", choices=["none", "logits", "probs"],
                   help="STSN-style: concat averaged BIO features per span into span repr. "
                        "'logits' = raw BIO logits, 'probs' = softmax probabilities.")
    p.add_argument("--rdrop-weight", type=float, default=0.0,
                   help="Weight for R-Drop KL-divergence consistency loss. "
                        "0 = disabled. Passes batch twice with different dropout, "
                        "adds KL(p1||p2) + KL(p2||p1) on span NER logits.")
    p.add_argument("--iou-neg-weight", type=float, default=0.0,
                   help="IoU-based weight for hard negative spans. "
                        "0 = disabled. When >0, negative spans overlapping gold "
                        "entities get loss weighted by (1 + iou_neg_weight * IoU). "
                        "SpERT.MT (2023) shows +2.88% RE on SciERC with IoU scaling.")
    p.add_argument("--label-smoothing", type=float, default=0.0,
                   help="Label smoothing for NER focal loss. 0 = disabled. "
                        "0.05-0.1 recommended. Prevents overconfident predictions "
                        "on hard boundary negatives.")
    p.add_argument("--re-focal-gamma", type=float, default=0.0,
                   help="Focal loss gamma for RE head. 0 = standard CE (default). "
                        "RE pairs are ~93%% NO_REL — focal loss downweights easy "
                        "negatives so the model focuses on hard relation cases.")
    p.add_argument("--re-train-conf", type=float, default=0.5,
                   help="Confidence threshold for predicted spans used in RE training. "
                        "Lower = more candidate pairs (noisier but more diverse). "
                        "Higher = cleaner pairs but fewer training signals.")
    p.add_argument("--span-proposal", action="store_true",
                   help="Use BIO-guided span proposals: merge BIO-decoded spans "
                        "(with boundary expansion) into exhaustive candidates. "
                        "Allows spans wider than max-span-width via BIO guidance.")
    p.add_argument("--span-proposal-expand", type=int, default=1,
                   help="Boundary expansion for BIO proposals (±N words). Default 1.")
    p.add_argument("--boundary-reg", action="store_true",
                   help="Add boundary regression head: predict (Δ_start, Δ_end) offsets "
                        "to refine span boundaries. Trained with smooth L1 loss.")
    p.add_argument("--boundary-reg-weight", type=float, default=0.1,
                   help="Weight for boundary regression loss.")
    p.add_argument("--boundary-refine", action="store_true",
                   help="Add 1D conv boundary refinement module (SRT-style). "
                        "Applies learnable 1D conv over span vectors to capture "
                        "local boundary patterns before NER classification.")
    p.add_argument("--eer-alpha", type=float, default=0.0,
                   help="Expected Entity Ratio loss alpha. 0 = disabled. "
                        "When >0, negative span losses are downweighted by "
                        "(1 - alpha * entity_density) where entity_density = "
                        "n_gold_entities / n_candidates. Treats unannotated "
                        "tokens as latent variables (Effland & Collins 2021).")
    p.add_argument("--re-neg-subsample", type=float, default=0.0,
                   help="RE negative subsampling ratio. 0 = disabled (use all pairs). "
                        "When >0, keep at most N × (number of positive RE pairs) "
                        "NO_REL pairs during training. Addresses 92%% NO_REL class "
                        "imbalance. Recommended: 3.0-5.0.")
    p.add_argument("--re-adv-neg", action="store_true",
                   help="Self-adversarial negative sampling for RE (RotatE-style). "
                        "Instead of random NO_REL subsampling, uses the model's current "
                        "predictions to preferentially sample hard negatives (pairs the "
                        "model incorrectly scores as real relations). Requires "
                        "--re-neg-subsample >0 to set the sampling budget.")
    p.add_argument("--re-adv-temp", type=float, default=0.5,
                   help="Temperature for self-adversarial RE sampling. Lower = sharper "
                        "distribution (more focus on hardest negatives). Default 0.5.")
    p.add_argument("--doc-window-size", type=int, default=1,
                   help="For document-aware datasets, join N consecutive sentences "
                        "from the same source document into one context window. "
                        "1 keeps sentence-level training.")
    p.add_argument("--doc-window-stride", type=int, default=1,
                   help="Stride for --doc-window-size sliding windows.")
    args = p.parse_args()
    # Resolve cycle aliases
    if args.cycle_jsonl_alias and not args.synth_jsonl:
        args.synth_jsonl = args.cycle_jsonl_alias
    if args.cycle_weight is not None:
        args.synth_weight = args.cycle_weight
    return args


def _write_relation_replay_jsonl(train_dataset, ds_mod, out_path, copies=1):
    """Write focused relation examples from the already-created train split."""
    records = []
    for ex in train_dataset.examples:
        words = ex["words"]
        type_by_span = {(s, e): t for (s, e, t) in ex["ner"]}
        for h_span, t_span, rel_id in ex["relations"]:
            if rel_id == ds_mod.NO_REL_ID:
                continue
            hs, he = h_span
            ts, te = t_span
            records.append({
                "synth_sentence": " ".join(words),
                "head": " ".join(words[hs:he + 1]),
                "tail": " ".join(words[ts:te + 1]),
                "rel": ds_mod.ID2REL[int(rel_id)],
                "rel_id": int(rel_id),
                "entity_type": type_by_span.get(h_span, ds_mod.ENTITY_TYPES[0]),
                "tail_entity_type": type_by_span.get(t_span, ds_mod.ENTITY_TYPES[0]),
                "containment": 1.0,
                "source_sentence": " ".join(words),
            })
    records = records * max(copies, 1)
    random.shuffle(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fout:
        for rec in records:
            fout.write(json.dumps(rec) + "\n")
    return len(records)


def focal_loss(logits, targets, gamma=2.0, weights=None, label_smoothing=0.0):
    """Focal loss for class-imbalanced classification.

    Args:
        weights: optional per-sample weights (same length as targets).
        label_smoothing: label smoothing factor (0.0 = no smoothing).
    """
    ce = F.cross_entropy(logits, targets, reduction="none",
                         label_smoothing=label_smoothing)
    pt = torch.exp(-ce)
    fl = (1 - pt) ** gamma * ce
    if weights is not None:
        fl = fl * weights
    return fl.mean()


def supervised_contrastive_loss(span_vecs, labels, tau=0.1, entity_only=False):
    """
    Supervised InfoNCE contrastive loss on span representations.

    Args:
        span_vecs: (N, D) L2-normalized span vectors
        labels: (N,) entity type ids (0=NONE, 1+=entity types)
        tau: temperature
        entity_only: if True, only use entity spans (label > 0)

    Returns:
        scalar loss
    """
    if entity_only:
        mask = labels > 0
        if mask.sum() < 2:
            return span_vecs.new_tensor(0.0)
        span_vecs = span_vecs[mask]
        labels = labels[mask]

    N = span_vecs.size(0)
    if N < 2:
        return span_vecs.new_tensor(0.0)

    # Similarity matrix
    sim = torch.mm(span_vecs, span_vecs.t()) / tau  # (N, N)

    # Positive mask: same label, exclude self
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    pos_mask.fill_diagonal_(0)

    # Need at least one positive pair
    if pos_mask.sum() == 0:
        return span_vecs.new_tensor(0.0)

    # Log-sum-exp over all non-self entries (denominator)
    self_mask = torch.eye(N, device=span_vecs.device).bool()
    sim_masked = sim.masked_fill(self_mask, float('-inf'))
    log_denom = torch.logsumexp(sim_masked, dim=1)  # (N,)

    # For each anchor, average log-prob over its positive pairs
    # loss_i = -1/|P(i)| * sum_{p in P(i)} (sim(i,p)/tau - log_denom(i))
    n_pos_per_anchor = pos_mask.sum(dim=1)  # (N,)
    has_pos = n_pos_per_anchor > 0

    if not has_pos.any():
        return span_vecs.new_tensor(0.0)

    log_prob = sim - log_denom.unsqueeze(1)  # (N, N)
    # Mask to only positive pairs, sum, average per anchor
    pos_log_prob = (log_prob * pos_mask).sum(dim=1)  # (N,)
    loss_per_anchor = -pos_log_prob[has_pos] / n_pos_per_anchor[has_pos]

    return loss_per_anchor.mean()


def _build_span_labels(gold_entities, num_words, max_span_width, entity_type2id):
    """
    Build a dict: (start, end_inclusive) -> entity_type_id (1-indexed).
    Spans not in gold get label 0 (NONE).
    """
    gold_span_labels = {}
    for (s, e, etype) in gold_entities:
        if e - s + 1 <= max_span_width:
            etype_id = entity_type2id.get(etype, 0)
            if etype_id > 0:
                gold_span_labels[(s, e)] = etype_id
    return gold_span_labels


def compute_span_loss(model, batch, device, ds_mod, entity_type2id,
                      re_weight=1.0, neg_sample_ratio=0.5, max_span_width=8,
                      focal_gamma=2.0, cl_weight=0.0, cl_tau=0.1,
                      cl_entity_only=False, bio_weight=0.0,
                      return_bio_logits=False, iou_neg_weight=0.0,
                      label_smoothing=0.0, re_focal_gamma=0.0,
                      re_train_conf=0.5,
                      span_proposal=False, span_proposal_expand=1,
                      boundary_reg_weight=0.0, eer_alpha=0.0,
                      re_neg_subsample=0.0,
                      re_adv_neg=False, re_adv_temp=0.5):
    """Compute span NER loss + RE loss + optional BIO auxiliary loss."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    word_ids_list = batch["word_ids"]
    gold_entities_list = batch["gold_entities"]
    gold_relations_list = batch["gold_relations"]
    num_words_list = batch["num_words"]

    hidden = model.encode(modality="text", input_ids=input_ids, attention_mask=attention_mask)

    # Auxiliary BIO NER loss (multi-task, per STSN 2024)
    # Also compute BIO logits when bio_enrich needs them for span repr
    bio_loss = hidden.new_tensor(0.0)
    bio_logits = None
    need_bio = bio_weight > 0 or return_bio_logits or model.bio_enrich != "none" or span_proposal
    if need_bio:
        ner_labels = batch["ner_labels"].to(device)
        bio_logits = model.forward_ner(hidden)  # (B, T, NUM_BIO_TAGS)
        if bio_weight > 0:
            bio_loss = F.cross_entropy(
                bio_logits.view(-1, bio_logits.size(-1)),
                ner_labels.view(-1),
                ignore_index=-100,
            )

    span_losses = []
    re_losses = []
    NO_REL = ds_mod.NO_REL_ID
    use_cl = cl_weight > 0
    # Accumulate span vecs/labels across batch for batch-level contrastive
    batch_cl_vecs = []
    batch_cl_labels = []

    for b_idx in range(input_ids.size(0)):
        n_words = num_words_list[b_idx]
        gold_ents = gold_entities_list[b_idx]
        gold_rels = gold_relations_list[b_idx]

        # BIO logits for this example (for bio_enrich)
        bio_logits_b = bio_logits[b_idx].detach() if bio_logits is not None else None

        # BIO-guided span proposals (merge with exhaustive candidates)
        bio_props = None
        if span_proposal and bio_logits_b is not None:
            bio_props = BertKGExtractor.bio_guided_proposals(
                bio_logits_b, word_ids_list[b_idx], n_words,
                expand=span_proposal_expand,
            )

        # Span NER (optionally return span vectors for contrastive loss)
        # forward_span_ner returns extra boundary_offsets when boundary_reg=True
        use_breg = boundary_reg_weight > 0 and hasattr(model, 'boundary_reg') and model.boundary_reg
        boundary_offsets = None
        if use_cl:
            result = model.forward_span_ner(
                hidden[b_idx], word_ids_list[b_idx], n_words, max_span_width,
                return_span_vecs=True, bio_logits_b=bio_logits_b,
                bio_proposals=bio_props,
            )
            if use_breg:
                span_logits, candidates, span_vecs, boundary_offsets = result
            else:
                span_logits, candidates, span_vecs = result
        else:
            result = model.forward_span_ner(
                hidden[b_idx], word_ids_list[b_idx], n_words, max_span_width,
                bio_logits_b=bio_logits_b, bio_proposals=bio_props,
            )
            if use_breg:
                span_logits, candidates, boundary_offsets = result
            else:
                span_logits, candidates = result
        if not candidates:
            continue

        # _build_span_labels needs to cover BIO-proposed spans too (may exceed max_span_width)
        gold_labels = _build_span_labels(gold_ents, n_words,
                                         max(max_span_width, 64) if bio_props else max_span_width,
                                         entity_type2id)

        # Build target tensor
        targets = []
        for (s, e) in candidates:
            targets.append(gold_labels.get((s, e), 0))
        targets = torch.tensor(targets, device=device, dtype=torch.long)

        # Collect span vectors for batch-level contrastive loss
        if use_cl and span_vecs is not None:
            cl_vecs = F.normalize(span_vecs, p=2, dim=-1)
            batch_cl_vecs.append(cl_vecs)
            batch_cl_labels.append(targets)

        # Negative sampling: keep all positives + sample negatives
        pos_mask = targets > 0
        neg_mask = targets == 0
        n_pos = pos_mask.sum().item()
        n_neg_keep = max(int(n_pos * neg_sample_ratio), 1)
        neg_indices = neg_mask.nonzero(as_tuple=True)[0]
        if len(neg_indices) > n_neg_keep:
            perm = torch.randperm(len(neg_indices), device=device)[:n_neg_keep]
            neg_indices = neg_indices[perm]
        keep_indices = torch.cat([pos_mask.nonzero(as_tuple=True)[0], neg_indices])

        if len(keep_indices) > 0:
            # Per-sample weights for negative spans
            sample_weights = None

            # IoU-weighted hard negative loss (SpERT.MT 2023)
            if iou_neg_weight > 0:
                gold_spans_se = [(s, e) for (s, e, _) in gold_ents]
                if gold_spans_se:
                    sample_weights = torch.ones(len(keep_indices), device=device)
                    kept_targets = targets[keep_indices]
                    for idx_k, ki in enumerate(keep_indices.tolist()):
                        if kept_targets[idx_k] == 0:  # negative span
                            cs, ce_ = candidates[ki]
                            max_iou = 0.0
                            for gs, ge in gold_spans_se:
                                inter = max(0, min(ce_, ge) - max(cs, gs) + 1)
                                union = (ce_ - cs + 1) + (ge - gs + 1) - inter
                                if union > 0:
                                    max_iou = max(max_iou, inter / union)
                            sample_weights[idx_k] = 1.0 + iou_neg_weight * max_iou

            # EER loss: downweight negatives by expected entity density
            # (Effland & Collins, TACL 2021 — treat unannotated as latent)
            if eer_alpha > 0 and len(candidates) > 0:
                entity_density = n_pos / len(candidates)
                if sample_weights is None:
                    sample_weights = torch.ones(len(keep_indices), device=device)
                kept_targets = targets[keep_indices]
                neg_weight = max(1.0 - eer_alpha * entity_density, 0.1)
                sample_weights[kept_targets == 0] *= neg_weight

            span_loss = focal_loss(span_logits[keep_indices], targets[keep_indices],
                                   gamma=focal_gamma, weights=sample_weights,
                                   label_smoothing=label_smoothing)
            span_losses.append(span_loss)

        # Boundary regression loss: smooth L1 on (Δ_start, Δ_end) for positive spans
        if use_breg and boundary_offsets is not None:
            if gold_ents:
                breg_preds = []
                breg_targets = []
                for idx_c, (cs, ce) in enumerate(candidates):
                    if targets[idx_c] > 0:  # positive span — has a matching gold
                        # Find the gold span with same type assignment
                        gs, ge = cs, ce  # default: no offset
                        for (g_s, g_e, g_t) in gold_ents:
                            if entity_type2id.get(g_t, 0) == targets[idx_c].item():
                                # Check overlap
                                if not (ce < g_s or g_e < cs):
                                    gs, ge = g_s, g_e
                                    break
                        breg_preds.append(boundary_offsets[idx_c])
                        breg_targets.append(torch.tensor(
                            [gs - cs, ge - ce], device=device, dtype=torch.float))
                if breg_preds:
                    breg_preds = torch.stack(breg_preds)
                    breg_targets = torch.stack(breg_targets)
                    breg_loss = F.smooth_l1_loss(breg_preds, breg_targets)
                    span_losses.append(boundary_reg_weight * breg_loss)

        # RE loss — use union of gold + predicted entity spans so the RE head
        # is trained on the same noisy-entity distribution it sees at eval time.
        # Gold spans ensure positive pairs are always available; predicted FP
        # spans add NO_REL pairs that calibrate the head for eval.
        gold_span_set = {(s, e) for (s, e, _) in gold_ents}
        with torch.no_grad():
            pred_types_re = span_logits.argmax(dim=-1).tolist()
            pred_confs_re = torch.softmax(span_logits, dim=-1).max(dim=-1).values.tolist()
        pred_span_set = set()
        for (s, e), etype_id, conf in zip(candidates, pred_types_re, pred_confs_re):
            if etype_id > 0 and conf >= re_train_conf:
                pred_span_set.add((s, e))
        re_spans = list(pred_span_set | gold_span_set)
        if len(re_spans) >= 2:
            rel_lookup = {(h, t): rid for (h, t, rid) in gold_rels}
            pairs = [(h, t) for h in re_spans for t in re_spans if h != t]
            if pairs:
                pair_targets = [rel_lookup.get((h, t), NO_REL) for (h, t) in pairs]

                # RE negative subsampling: keep all positive pairs + subsample NO_REL
                if re_neg_subsample > 0:
                    pos_idx = [i for i, t in enumerate(pair_targets) if t != NO_REL]
                    neg_idx = [i for i, t in enumerate(pair_targets) if t == NO_REL]
                    n_pos_re = len(pos_idx)
                    if n_pos_re > 0 and len(neg_idx) > int(n_pos_re * re_neg_subsample):
                        n_neg_keep_re = int(n_pos_re * re_neg_subsample)
                        if re_adv_neg and n_neg_keep_re < len(neg_idx):
                            # Self-adversarial sampling (RotatE-style): prefer hard negatives
                            # that the model currently scores as real relations.
                            with torch.no_grad():
                                all_re_logits = model.forward_re(
                                    hidden[b_idx], word_ids_list[b_idx], pairs)
                                neg_logits = all_re_logits[neg_idx]  # (n_neg, n_rels)
                                # Score = sum of non-NO_REL probabilities
                                adv_scores = torch.softmax(neg_logits, dim=-1)[:, 1:].sum(dim=-1)
                                adv_weights = torch.softmax(adv_scores / re_adv_temp, dim=0)
                                sampled = torch.multinomial(
                                    adv_weights, n_neg_keep_re, replacement=False).tolist()
                            neg_sample = [neg_idx[i] for i in sampled]
                        else:
                            neg_sample = random.sample(neg_idx, n_neg_keep_re)
                        keep_idx = sorted(pos_idx + neg_sample)
                        pairs = [pairs[i] for i in keep_idx]
                        pair_targets = [pair_targets[i] for i in keep_idx]

                pair_targets_t = torch.tensor(pair_targets, device=device, dtype=torch.long)
                re_logits = model.forward_re(hidden[b_idx], word_ids_list[b_idx], pairs)
                if re_focal_gamma > 0:
                    re_losses.append(focal_loss(re_logits, pair_targets_t,
                                                gamma=re_focal_gamma))
                else:
                    re_losses.append(F.cross_entropy(re_logits, pair_targets_t))

    ner_loss = torch.stack(span_losses).mean() if span_losses else hidden.new_tensor(0.0)
    re_loss = torch.stack(re_losses).mean() if re_losses else hidden.new_tensor(0.0)
    # Batch-level contrastive: concatenate all span vecs across batch, then one CL call
    if use_cl and batch_cl_vecs:
        all_vecs = torch.cat(batch_cl_vecs, dim=0)
        all_labels = torch.cat(batch_cl_labels, dim=0)
        contrastive_loss = supervised_contrastive_loss(
            all_vecs, all_labels, tau=cl_tau, entity_only=cl_entity_only,
        )
    else:
        contrastive_loss = hidden.new_tensor(0.0)
    total = ner_loss + re_weight * re_loss + cl_weight * contrastive_loss + bio_weight * bio_loss
    if return_bio_logits:
        return total, ner_loss.detach(), re_loss.detach(), contrastive_loss.detach(), bio_loss.detach(), bio_logits
    return total, ner_loss.detach(), re_loss.detach(), contrastive_loss.detach(), bio_loss.detach()


def evaluate_span(model, dataloader, device, ds_mod, entity_type2id, id2entity_type,
                  max_span_width=8, span_threshold=0.5, verbose=False,
                  span_proposal=False, span_proposal_expand=1):
    """Evaluate with span-based NER predictions feeding into RE."""
    from eval.triple_f1 import _prf
    model.eval()
    NO_REL = ds_mod.NO_REL_ID

    ner_tp = ner_fp = ner_fn = 0
    triple_tp = triple_fp = triple_fn = 0
    n_examples = 0
    # Diagnostic counters
    total_pred_ents = 0
    total_gold_ents = 0
    total_pred_rels = 0
    total_gold_rels = 0
    total_re_pairs = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            word_ids_list = batch["word_ids"]
            gold_entities_list = batch["gold_entities"]
            gold_relations_list = batch["gold_relations"]
            num_words_list = batch["num_words"]

            hidden = model.encode(modality="text", input_ids=input_ids, attention_mask=attention_mask)

            # Compute BIO logits for bio_enrich and/or span proposals
            bio_logits = None
            if model.bio_enrich != "none" or span_proposal:
                bio_logits = model.forward_ner(hidden)  # (B, T, NUM_BIO_TAGS)

            for b_idx in range(input_ids.size(0)):
                n_examples += 1
                n_words = num_words_list[b_idx]
                gold_ents = gold_entities_list[b_idx]
                gold_rels = gold_relations_list[b_idx]
                bio_logits_b = bio_logits[b_idx] if bio_logits is not None else None

                # BIO-guided span proposals for eval
                bio_props = None
                if span_proposal and bio_logits_b is not None:
                    bio_props = BertKGExtractor.bio_guided_proposals(
                        bio_logits_b, word_ids_list[b_idx], n_words,
                        expand=span_proposal_expand,
                    )

                use_breg = hasattr(model, 'boundary_reg') and model.boundary_reg
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

                # Predict spans: take argmax, keep those != NONE (0)
                span_probs = torch.softmax(span_logits, dim=-1)
                pred_types = span_logits.argmax(dim=-1).tolist()
                pred_confs = span_probs.max(dim=-1).values.tolist()

                pred_spans = []  # (s, e, etype, conf)
                for idx_c, ((s, e), etype_id, conf) in enumerate(
                        zip(candidates, pred_types, pred_confs)):
                    if etype_id > 0 and conf >= span_threshold:
                        # Apply boundary regression offsets
                        if boundary_offsets is not None:
                            ds = round(boundary_offsets[idx_c, 0].item())
                            de = round(boundary_offsets[idx_c, 1].item())
                            s_new = max(0, min(s + ds, n_words - 1))
                            e_new = max(s_new, min(e + de, n_words - 1))
                            s, e = s_new, e_new
                        etype = id2entity_type.get(etype_id, "Unknown")
                        pred_spans.append((s, e, etype, conf))

                # Remove overlapping spans: keep highest confidence
                # (greedy non-overlapping: sort by confidence, skip overlaps)
                scored = sorted(
                    pred_spans,
                    key=lambda x: -x[3],
                )
                taken = set()
                filtered = []
                for (s, e, t, c) in scored:
                    overlap = any(
                        not (e < ts or te < s)
                        for (ts, te) in taken
                    )
                    if not overlap:
                        filtered.append((s, e, t))
                        taken.add((s, e))
                pred_spans = filtered
                total_pred_ents += len(pred_spans)

                # NER F1
                pred_ent_set = {(s, e, t) for (s, e, t) in pred_spans}
                gold_ent_set = {(s, e, t) for (s, e, t) in gold_ents}
                ner_tp += len(pred_ent_set & gold_ent_set)
                ner_fp += len(pred_ent_set - gold_ent_set)
                ner_fn += len(gold_ent_set - pred_ent_set)

                # Triple F1 (full pipeline)
                pred_span_list = [(s, e) for (s, e, _) in pred_spans]
                pred_pairs = [(a, b) for a in pred_span_list for b in pred_span_list if a != b]
                total_re_pairs += len(pred_pairs)
                if pred_pairs:
                    pred_re_logits = model.forward_re(hidden[b_idx], word_ids_list[b_idx], pred_pairs)
                    pred_re_ids = pred_re_logits.argmax(dim=-1).tolist()
                    pred_full = {(h, t, p) for (h, t), p in zip(pred_pairs, pred_re_ids) if p != NO_REL}
                else:
                    pred_full = set()
                total_pred_rels += len(pred_full)
                triple_tp += len(pred_full & gold_full)
                triple_fp += len(pred_full - gold_full)
                triple_fn += len(gold_full - pred_full)

    _, _, nf = _prf(ner_tp, ner_fp, ner_fn)
    _, _, tf = _prf(triple_tp, triple_fp, triple_fn)

    if verbose:
        ner_p = ner_tp / max(ner_tp + ner_fp, 1)
        ner_r = ner_tp / max(ner_tp + ner_fn, 1)
        tri_p = triple_tp / max(triple_tp + triple_fp, 1)
        tri_r = triple_tp / max(triple_tp + triple_fn, 1)
        no_rel_frac = 1.0 - total_pred_rels / max(total_re_pairs, 1)
        print(f"    [diag] pred_ents={total_pred_ents} gold_ents={total_gold_ents} | "
              f"pred_rels={total_pred_rels} gold_rels={total_gold_rels} | "
              f"re_pairs={total_re_pairs} NO_REL%={no_rel_frac:.2f}")
        print(f"    [diag] NER  P={ner_p:.3f} R={ner_r:.3f} F1={nf:.3f} | "
              f"Triple P={tri_p:.3f} R={tri_r:.3f} F1={tf:.3f}")
        print(f"    [diag] triple_tp={triple_tp} triple_fp={triple_fp} triple_fn={triple_fn}")

    return {"ner_f1": nf, "triple_f1": tf, "n_examples": n_examples}


def cycle(loader):
    while True:
        yield from loader


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    ds_mod = importlib.import_module(DATASET_REGISTRY[args.dataset])
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_name is None:
        args.model_name = {
            "scierc": "allenai/scibert_scivocab_uncased",
            "scier": "allenai/scibert_scivocab_uncased",
            "conll04": "bert-base-uncased",
            "ade": "allenai/scibert_scivocab_uncased",
        }.get(args.dataset, "bert-base-uncased")

    print(f"=== Span-based NER training ({args.dataset}) ===")
    print(f"  encoder:        {args.model_name}")
    print(f"  max_span_width: {args.max_span_width}")
    print(f"  neg_sample:     {args.neg_sample_ratio}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # Pass seed to build_dataloaders for datasets that create dev split at runtime
    # (CODE-ACCORD, CUAD). Datasets with fixed splits (SciERC, SciER, CoNLL04, ADE)
    # ignore the seed kwarg. This ensures each seed gets its own train/dev split,
    # making multi-seed evaluation measure true generalization, not init luck on a
    # single fixed dev set.
    import inspect
    dl_kwargs = dict(batch_size=args.batch_size, max_length=args.max_length)
    dl_params = inspect.signature(ds_mod.build_dataloaders).parameters
    if "seed" in dl_params:
        dl_kwargs["seed"] = args.seed
    if "doc_window_size" in dl_params:
        dl_kwargs["doc_window_size"] = args.doc_window_size
    if "doc_window_stride" in dl_params:
        dl_kwargs["doc_window_stride"] = args.doc_window_stride
    train_loader, dev_loader, test_loader = ds_mod.build_dataloaders(
        tokenizer, **dl_kwargs,
    )
    print(f"  train: {len(train_loader.dataset)} | dev: {len(dev_loader.dataset)}")

    # Entity type mapping: type_name -> id (1-indexed, 0 = NONE)
    entity_types = ds_mod.ENTITY_TYPES
    entity_type2id = {t: i + 1 for i, t in enumerate(entity_types)}
    id2entity_type = {i + 1: t for i, t in enumerate(entity_types)}
    num_entity_types = len(entity_types)
    print(f"  entity types:   {entity_types} ({num_entity_types})")

    # Patch scierc dicts only if running a different dataset
    import data.scierc as scierc_mod
    if args.dataset != "scierc":
        scierc_mod.ID2BIO.clear()
        scierc_mod.ID2BIO.update(ds_mod.ID2BIO)
        scierc_mod.BIO_TAG2ID.clear()
        scierc_mod.BIO_TAG2ID.update(ds_mod.BIO_TAG2ID)
        scierc_mod.NO_REL_ID = ds_mod.NO_REL_ID

    model = BertKGExtractor(
        args.model_name,
        num_bio_tags=ds_mod.NUM_BIO_TAGS,
        num_relations=ds_mod.NUM_RELATIONS,
        num_entity_types=num_entity_types,
        use_span_ner=True,
        max_span_width=args.max_span_width,
        bio_enrich=args.bio_enrich,
        boundary_reg=args.boundary_reg,
        boundary_refine=args.boundary_refine,
    ).to(device)

    # Load ELECTRA cooperative pre-training checkpoint if provided
    if args.pretrain_ckpt:
        ckpt = torch.load(args.pretrain_ckpt, map_location=device)
        if "discriminator" in ckpt:
            pretrained_sd = ckpt["discriminator"]
        elif "encoder" in ckpt:
            pretrained_sd = ckpt["encoder"]
        else:
            pretrained_sd = ckpt
        # Filter out keys with shape mismatches (task heads differ across datasets)
        model_sd = model.state_dict()
        filtered_sd = {
            k: v for k, v in pretrained_sd.items()
            if k in model_sd and model_sd[k].shape == v.shape
        }
        skipped = [k for k in pretrained_sd if k not in filtered_sd]
        if skipped:
            print(f"  Skipping {len(skipped)} shape-mismatched keys: {skipped[:5]}...")
        missing, unexpected = model.load_state_dict(filtered_sd, strict=False)
        # Reinitialize task-specific heads for fresh fine-tuning
        model.span_ner_head.reset_parameters()
        model.span_width_emb.reset_parameters()
        model.span_width_proj.reset_parameters()
        for m in model.re_head:
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        print(f"  Loaded pre-trained weights from {args.pretrain_ckpt}")
        print(f"    loaded keys: {len(filtered_sd)}, skipped: {len(skipped)}")
        print(f"    missing: {len(missing)}, unexpected: {len(unexpected)}")

    print(f"  span_ner_head:  {model.span_ner_head}")
    print(f"  re_head out:    {model.re_head[-1].out_features}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps,
    )

    if args.relation_replay:
        replay_path = Path(f"/tmp/{args.dataset}_relation_replay_s{args.seed}.jsonl")
        n_replay = _write_relation_replay_jsonl(
            train_loader.dataset, ds_mod, replay_path,
            copies=args.relation_replay_copies,
        )
        args.synth_jsonl = str(replay_path)
        print(f"  relation_replay: {n_replay} examples -> {args.synth_jsonl}")

    use_synth = bool(args.synth_jsonl)
    synth_loader = None
    if use_synth:
        from data.synth_loader import build_synth_loader
        synth_loader = build_synth_loader(
            tokenizer, args.synth_jsonl,
            batch_size=args.batch_size, max_length=args.max_length,
            ds_mod=ds_mod,
        )
        print(f"  synth: {len(synth_loader.dataset)}")

    gold_iter = cycle(train_loader)
    synth_iter = cycle(synth_loader) if synth_loader else None
    best_metrics = {"triple_f1": -1.0}
    best_step = -1

    use_cl = args.cl_weight > 0
    if use_cl:
        print(f"  contrastive:    weight={args.cl_weight} tau={args.cl_tau} entity_only={args.cl_entity_only}")

    # Bio-weight curriculum: if --bio-start > 0, linearly decay from bio_start to bio_end
    use_bio_curriculum = args.bio_start > 0
    if use_bio_curriculum:
        print(f"  bio_curriculum: {args.bio_start} → {args.bio_end} over {args.max_steps} steps")
    elif args.bio_weight > 0:
        print(f"  bio_multitask:  weight={args.bio_weight}")
    if args.bio_enrich != "none":
        print(f"  bio_enrich:     {args.bio_enrich}")
    if args.rdrop_weight > 0:
        print(f"  rdrop:          weight={args.rdrop_weight}")

    model.train()
    t0 = time.time()
    step = 0
    while step < args.max_steps:
        optimizer.zero_grad()
        batch = next(gold_iter)

        # Compute effective bio_weight (curriculum or constant)
        if use_bio_curriculum:
            bio_w_eff = args.bio_start - (args.bio_start - args.bio_end) * (step / max(args.max_steps - 1, 1))
        else:
            bio_w_eff = args.bio_weight

        use_rdrop = args.rdrop_weight > 0
        if use_rdrop:
            # R-Drop: two forward passes with different dropout, KL on BIO logits
            gold_loss, ner_loss, re_loss, cl_loss, bio_l, bio_logits1 = compute_span_loss(
                model, batch, device, ds_mod, entity_type2id,
                re_weight=args.re_weight, neg_sample_ratio=args.neg_sample_ratio,
                max_span_width=args.max_span_width, focal_gamma=args.focal_gamma,
                cl_weight=args.cl_weight, cl_tau=args.cl_tau,
                cl_entity_only=args.cl_entity_only,
                bio_weight=bio_w_eff, return_bio_logits=True,
                iou_neg_weight=args.iou_neg_weight,
                label_smoothing=args.label_smoothing,
                re_focal_gamma=args.re_focal_gamma,
                re_train_conf=args.re_train_conf,
                span_proposal=args.span_proposal,
                span_proposal_expand=args.span_proposal_expand,
                boundary_reg_weight=args.boundary_reg_weight,
                eer_alpha=args.eer_alpha,
                re_neg_subsample=args.re_neg_subsample,
                re_adv_neg=args.re_adv_neg,
                re_adv_temp=args.re_adv_temp,
            )
            gold_loss2, _, _, _, _, bio_logits2 = compute_span_loss(
                model, batch, device, ds_mod, entity_type2id,
                re_weight=args.re_weight, neg_sample_ratio=args.neg_sample_ratio,
                max_span_width=args.max_span_width, focal_gamma=args.focal_gamma,
                cl_weight=args.cl_weight, cl_tau=args.cl_tau,
                cl_entity_only=args.cl_entity_only,
                bio_weight=bio_w_eff, return_bio_logits=True,
                iou_neg_weight=args.iou_neg_weight,
                label_smoothing=args.label_smoothing,
                re_focal_gamma=args.re_focal_gamma,
                re_train_conf=args.re_train_conf,
                span_proposal=args.span_proposal,
                span_proposal_expand=args.span_proposal_expand,
                boundary_reg_weight=args.boundary_reg_weight,
                eer_alpha=args.eer_alpha,
                re_neg_subsample=args.re_neg_subsample,
                re_adv_neg=args.re_adv_neg,
                re_adv_temp=args.re_adv_temp,
            )
            # Average the two losses + symmetric KL on BIO logits
            gold_loss = (gold_loss + gold_loss2) / 2
            if bio_logits1 is not None and bio_logits2 is not None:
                p = F.log_softmax(bio_logits1.view(-1, bio_logits1.size(-1)), dim=-1)
                q = F.log_softmax(bio_logits2.view(-1, bio_logits2.size(-1)), dim=-1)
                rdrop_loss = (F.kl_div(p, q.exp(), reduction='batchmean') +
                              F.kl_div(q, p.exp(), reduction='batchmean')) / 2
            else:
                rdrop_loss = gold_loss.new_tensor(0.0)
            gold_loss = gold_loss + args.rdrop_weight * rdrop_loss
        else:
            gold_loss, ner_loss, re_loss, cl_loss, bio_l = compute_span_loss(
                model, batch, device, ds_mod, entity_type2id,
                re_weight=args.re_weight, neg_sample_ratio=args.neg_sample_ratio,
                max_span_width=args.max_span_width, focal_gamma=args.focal_gamma,
                cl_weight=args.cl_weight, cl_tau=args.cl_tau,
                cl_entity_only=args.cl_entity_only,
                bio_weight=bio_w_eff,
                iou_neg_weight=args.iou_neg_weight,
                label_smoothing=args.label_smoothing,
                re_focal_gamma=args.re_focal_gamma,
                re_train_conf=args.re_train_conf,
                span_proposal=args.span_proposal,
                span_proposal_expand=args.span_proposal_expand,
                boundary_reg_weight=args.boundary_reg_weight,
                eer_alpha=args.eer_alpha,
                re_neg_subsample=args.re_neg_subsample,
                re_adv_neg=args.re_adv_neg,
                re_adv_temp=args.re_adv_temp,
            )

        synth_loss_val = 0.0
        if use_synth and step >= args.gold_only_steps and synth_iter:
            synth_batch = next(synth_iter)
            try:
                s_loss, _, _, _, _ = compute_span_loss(
                    model, synth_batch, device, ds_mod, entity_type2id,
                    re_weight=args.re_weight, neg_sample_ratio=args.neg_sample_ratio,
                    max_span_width=args.max_span_width, focal_gamma=args.focal_gamma,
                    cl_weight=args.cl_weight, cl_tau=args.cl_tau,
                    cl_entity_only=args.cl_entity_only,
                    bio_weight=bio_w_eff,
                    iou_neg_weight=args.iou_neg_weight,
                    label_smoothing=args.label_smoothing,
                    re_focal_gamma=args.re_focal_gamma,
                    re_train_conf=args.re_train_conf,
                    span_proposal=args.span_proposal,
                    span_proposal_expand=args.span_proposal_expand,
                    boundary_reg_weight=args.boundary_reg_weight,
                )
                synth_loss_val = s_loss.item()
                total = gold_loss + args.synth_weight * s_loss
            except Exception:
                total = gold_loss
        else:
            total = gold_loss

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 10 == 0:
            dt = (time.time() - t0) * 1000 / max(step, 1)
            cur_lr = scheduler.get_last_lr()[0]
            cl_str = f" CL={cl_loss.item():.4f}" if use_cl else ""
            bio_str = f" BIO={bio_l.item():.4f}(w={bio_w_eff:.3f})" if bio_w_eff > 0 else ""
            rdrop_str = f" RD={rdrop_loss.item():.4f}" if use_rdrop else ""
            msg = (f"[Step {step:04d}] L={gold_loss.item():.4f} NER={ner_loss.item():.4f} "
                   f"RE={re_loss.item():.4f}{cl_str}{bio_str}{rdrop_str} synth={synth_loss_val:.4f} lr={cur_lr:.2e} | {dt:.0f}ms/step")
            print(msg)
            sys.stdout.flush()
            # Write progress to file for monitoring
            with open("/tmp/train_progress.txt", "a") as _pf:
                _pf.write(msg + "\n")

        if step > 0 and step % args.eval_every == 0:
            metrics = evaluate_span(
                model, dev_loader, device, ds_mod,
                entity_type2id, id2entity_type,
                max_span_width=args.max_span_width,
                verbose=True,
                span_proposal=args.span_proposal,
                span_proposal_expand=args.span_proposal_expand,
            )
            star = ""
            if metrics["triple_f1"] > best_metrics["triple_f1"]:
                best_metrics = dict(metrics)
                best_step = step
                star = " *"
                if args.save_best_to:
                    save_path = Path(args.save_best_to)
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"encoder": model.state_dict(), "step": step,
                                "metrics": metrics}, save_path)
            eval_msg = (f"[Eval @ {step}] NER={metrics['ner_f1']:.4f} "
                        f"Triple={metrics['triple_f1']:.4f}{star}")
            print(eval_msg)
            sys.stdout.flush()
            with open("/tmp/train_progress.txt", "a") as _pf:
                _pf.write(eval_msg + "\n")
            model.train()

        step += 1

    for split_name, loader in [("dev", dev_loader), ("test", test_loader)]:
        metrics = evaluate_span(
            model, loader, device, ds_mod,
            entity_type2id, id2entity_type,
            max_span_width=args.max_span_width,
            span_proposal=args.span_proposal,
            span_proposal_expand=args.span_proposal_expand,
        )
        if split_name == "dev" and metrics["triple_f1"] > best_metrics["triple_f1"]:
            best_metrics = dict(metrics)
            best_step = step
        msg = (f"\n=== {split_name.upper()} (step {step}) ===\n"
               f"  NER={metrics['ner_f1']:.4f} Triple={metrics['triple_f1']:.4f}")
        print(msg)
        sys.stdout.flush()
        with open("/tmp/train_progress.txt", "a") as _pf:
            _pf.write(msg + "\n")
    final_msg = (f"=== BEST DEV (step {best_step}) ===\n"
                 f"  NER={best_metrics['ner_f1']:.4f} Triple={best_metrics['triple_f1']:.4f}\n"
                 f"  time={time.time()-t0:.1f}s")
    print(final_msg)
    sys.stdout.flush()
    with open("/tmp/train_progress.txt", "a") as _pf:
        _pf.write(final_msg + "\n")


if __name__ == "__main__":
    main()
