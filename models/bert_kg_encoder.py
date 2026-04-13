"""
KG extractor with pluggable modality adapters (Stage 2 / Stage 3 forward-compat).

Architecture:

    raw_input (input_ids | pixel_values | rdb_schema | ...)
        ↓
    Adapter[modality]                  ← swappable; lives in models/adapters/
        ↓ (inputs_embeds, attention_mask)   unified hidden dim, modality-agnostic
        ↓
    BERT body (or any transformer that supports inputs_embeds)
        ↓ (B, T, H)
        ├── NER head: token classification (BIO scheme)
        └── RE head:  span-pair classification (NUM_RELATIONS incl. NO_REL)

Adapter contract:
    forward(**raw_inputs) -> dict with at least:
        "inputs_embeds":   (B, T, hidden)   ← matches backbone hidden_size
        "attention_mask":  (B, T)           ← 1=keep, 0=pad

Stage 2:
    TextAdapter wraps BERT's own word_embeddings layer; the math is identical
    to calling bert(input_ids=...) — verified by reproducing stage2-004.

Stage 3 (future, no code changes to backbone or heads needed):
    ImageAdapter:    ViT patch embeddings + Linear projection to hidden_size
    TableAdapter:    schema cell embeddings + Linear projection
    ...

Why this matters now: when we move to engineering documents (Stage 3 in
future_implementation_plan.md), we'll need to ingest flowcharts, RDB schemas,
CAD drawings — each with its own input format. Building the adapter pattern
now means adding a modality is "implement one Adapter class + register it";
no surgery on the model body or heads.

Stage 2 training loss (unchanged from prior version):
    L = NER cross-entropy + λ · RE cross-entropy(NUM_RELATIONS=8 incl. NO_REL)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from data.scierc import NUM_BIO_TAGS, NUM_RELATIONS, NO_REL_ID, BIO_TAG2ID, ID2BIO


# ─── Modality adapters ──────────────────────────────────────────────────


class TextAdapter(nn.Module):
    """
    Wraps BERT's own word_embeddings table so the rest of the model can see
    a unified `inputs_embeds` tensor. Uses the EXACT same embedding parameters
    as the underlying BERT body, so numeric output is identical to calling
    `bert(input_ids=...)`.

    Why not just use bert.embeddings (full module = word + position + token
    type + LayerNorm + dropout)? Because BERT applies all of those internally
    when given inputs_embeds; if we pre-applied them in the adapter we'd
    double-apply LayerNorm. The adapter only does word lookup; everything
    else stays inside the backbone.
    """

    def __init__(self, bert_word_embeddings: nn.Embedding):
        super().__init__()
        # Reference (not copy) — adapter shares parameters with the backbone's
        # word embeddings. Both see the same gradients during training.
        self.word_embeddings = bert_word_embeddings

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return {
            "inputs_embeds": self.word_embeddings(input_ids),
            "attention_mask": attention_mask,
        }


# ─── Backbone wrapper ────────────────────────────────────────────────────


class BertBackbone(nn.Module):
    """
    Thin wrapper around HF BertModel that always consumes inputs_embeds.
    Adapters are responsible for producing `(inputs_embeds, attention_mask)`;
    the backbone is modality-agnostic.
    """

    def __init__(self, model_name: str):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)

    @property
    def hidden_size(self) -> int:
        return self.bert.config.hidden_size

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor):
        out = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return out.last_hidden_state  # (B, T, H)


# ─── Top-level extractor ─────────────────────────────────────────────────


class BertKGExtractor(nn.Module):
    """
    Pluggable KG extractor:
        - register one or more Adapters (one per modality)
        - call .encode(modality, **raw_inputs) to get contextualized hidden states
        - call .forward_ner(hidden) and .forward_re(hidden_b, word_ids_b, pairs)
          to get task-specific logits

    Stage 2 only registers a TextAdapter ("text"). Stage 3 will additionally
    register ImageAdapter, TableAdapter, etc. without touching this class.
    """

    def __init__(self, model_name: str = "bert-base-uncased", dropout: float = 0.1,
                 use_crf: bool = False,
                 num_bio_tags: int = None, num_relations: int = None,
                 num_entity_types: int = None, use_span_ner: bool = False,
                 max_span_width: int = 8):
        super().__init__()
        self.backbone = BertBackbone(model_name)
        hidden = self.backbone.hidden_size

        # Allow explicit override for multi-dataset support (train_multi.py).
        n_bio = num_bio_tags if num_bio_tags is not None else NUM_BIO_TAGS
        n_rel = num_relations if num_relations is not None else NUM_RELATIONS

        self.dropout = nn.Dropout(dropout)
        self.ner_head = nn.Linear(hidden, n_bio)

        # Optional CRF layer for NER (stage2-012, negative result).
        self.use_crf = use_crf
        self.crf = None
        if use_crf:
            from torchcrf import CRF
            self.crf = CRF(n_bio, batch_first=True)

        # ── Span-based NER head (stage2-024) ──────────────────────────
        # Classifies candidate (start, end) spans as entity_type or NONE.
        # NONE = class 0; entity types = 1..num_entity_types.
        # The BIO head is still used for backward compat; span head is
        # an additional head that provides span-level predictions.
        self.use_span_ner = use_span_ner
        self.max_span_width = max_span_width
        if use_span_ner:
            n_ent = num_entity_types if num_entity_types is not None else 6  # SciERC default
            # Span repr = [start; end; max_pool] → 3H input
            self.span_ner_head = nn.Linear(hidden * 3, n_ent + 1)
            self.span_width_emb = nn.Embedding(max_span_width, hidden)
            self.span_width_proj = nn.Linear(hidden, hidden * 3)  # project width emb to 3H

        # RE head — 2H concat + 2-layer MLP.
        self.re_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_rel),
        )

        # ── Pluggable adapters ─────────────────────────────────────────
        self.adapters = nn.ModuleDict()
        # Register the TextAdapter by default — Stage 2 only needs text.
        # Shares the BERT body's word_embeddings parameters so the forward
        # path is mathematically identical to bert(input_ids=...).
        self.register_adapter("text", TextAdapter(self.backbone.bert.embeddings.word_embeddings))

    # ── Adapter registration API ────────────────────────────────────────

    def register_adapter(self, name: str, adapter: nn.Module) -> None:
        """
        Register a new modality adapter. Stage 3 use:

            extractor.register_adapter("image", ImageAdapter(vit_model, hidden))
            extractor.register_adapter("table", TableAdapter(...))

        The adapter must produce `inputs_embeds` of shape (B, T, hidden_size)
        and `attention_mask` of shape (B, T). No other constraint.
        """
        if name in self.adapters:
            raise ValueError(f"Adapter '{name}' already registered. "
                             "Unregister first or use a different name.")
        self.adapters[name] = adapter

    def unregister_adapter(self, name: str) -> None:
        if name in self.adapters:
            del self.adapters[name]

    # ── Forward passes ──────────────────────────────────────────────────

    def encode(self, modality: str = "text", **raw_inputs):
        """
        Run the adapter for `modality` on `raw_inputs`, then the backbone.
        Returns (B, T, H) contextualized hidden states.
        """
        if modality not in self.adapters:
            raise KeyError(f"No adapter registered for modality '{modality}'. "
                           f"Available: {list(self.adapters.keys())}")
        adapter = self.adapters[modality]
        adapter_out = adapter(**raw_inputs)
        return self.backbone(
            inputs_embeds=adapter_out["inputs_embeds"],
            attention_mask=adapter_out["attention_mask"],
        )

    def forward_ner(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.ner_head(self.dropout(hidden_states))  # (B, T, NUM_BIO_TAGS)

    def _first_token_for_word(self, word_ids_b: list, word_idx: int) -> int:
        """Return the first token index mapped to word_idx."""
        for i, wid in enumerate(word_ids_b):
            if wid == word_idx:
                return i
        return 0

    def _last_token_for_word(self, word_ids_b: list, word_idx: int) -> int:
        """Return the last token index mapped to word_idx."""
        last = 0
        for i, wid in enumerate(word_ids_b):
            if wid == word_idx:
                last = i
        return last

    def span_repr_v2(self, hidden_states_b: torch.Tensor, word_ids_b: list,
                     span: tuple) -> torch.Tensor:
        """
        SpERT-style span representation:
          [start_token; end_token; max_pool(span_tokens)]
        Returns (3H,)
        """
        word_start, word_end = span
        start_tok = self._first_token_for_word(word_ids_b, word_start)
        end_tok = self._last_token_for_word(word_ids_b, word_end)
        start_vec = hidden_states_b[start_tok]
        end_vec = hidden_states_b[end_tok]
        pool_vec = self.span_repr(hidden_states_b, word_ids_b, span)
        return torch.cat([start_vec, end_vec, pool_vec], dim=-1)  # (3H,)

    def forward_span_ner(self, hidden_states_b: torch.Tensor, word_ids_b: list,
                         num_words: int, max_span_width: int = 8):
        """
        Span-based NER v2: enumerate candidate spans, compute SpERT-style
        span representations [start; end; max_pool], classify as entity
        type or NONE.

        Returns:
            span_logits: (num_candidates, num_entity_types + 1)
            candidates:  list of (start, end_inclusive) word-level spans
        """
        if not hasattr(self, "span_ner_head"):
            raise RuntimeError("span_ner_head not initialized. Use use_span_ner=True.")
        candidates = []
        for s in range(num_words):
            for e in range(s, min(s + max_span_width, num_words)):
                candidates.append((s, e))
        if not candidates:
            return hidden_states_b.new_zeros((0, self.span_ner_head.out_features)), []

        span_vecs = []
        for (s, e) in candidates:
            vec = self.span_repr_v2(hidden_states_b, word_ids_b, (s, e))
            span_vecs.append(vec)
        span_vecs = torch.stack(span_vecs, dim=0)  # (num_candidates, 3H)

        # Add span width embedding (broadcast to 3H via linear projection)
        widths = torch.tensor(
            [min(e - s, self.span_width_emb.num_embeddings - 1) for (s, e) in candidates],
            device=hidden_states_b.device, dtype=torch.long,
        )
        width_vec = self.span_width_proj(self.span_width_emb(widths))  # (N, 3H)
        span_vecs = span_vecs + width_vec

        logits = self.span_ner_head(self.dropout(span_vecs))
        return logits, candidates

    def span_repr(self, hidden_states: torch.Tensor, word_ids_list: list, span: tuple) -> torch.Tensor:
        """
        Extract span representation by max-pooling subword embeddings whose
        word_ids fall within [span_start, span_end] (both inclusive).
        Returns (H,)
        """
        word_start, word_end = span
        token_idx = [
            i for i, wid in enumerate(word_ids_list)
            if wid is not None and word_start <= wid <= word_end
        ]
        if not token_idx:
            return hidden_states.new_zeros(hidden_states.size(-1))
        return hidden_states[token_idx].max(dim=0).values

    def forward_re(self, hidden_states_b: torch.Tensor, word_ids_b: list, pairs: list) -> torch.Tensor:
        """
        For one example in the batch:
            hidden_states_b: (T, H)
            word_ids_b:      list[int|None] length T
            pairs:           list of ((hs, he), (ts, te))
        Returns: (num_pairs, NUM_RELATIONS)
        """
        if not pairs:
            return hidden_states_b.new_zeros((0, NUM_RELATIONS))
        feats = []
        for (hs, he), (ts, te) in pairs:
            head_vec = self.span_repr(hidden_states_b, word_ids_b, (hs, he))
            tail_vec = self.span_repr(hidden_states_b, word_ids_b, (ts, te))
            feats.append(torch.cat([head_vec, tail_vec], dim=-1))
        feats = torch.stack(feats, dim=0)  # (num_pairs, 2H)
        return self.re_head(self.dropout(feats))


# ─── Loss ────────────────────────────────────────────────────────────────


def compute_loss(model, batch, device, re_weight: float = 1.0):
    """
    Compute Stage 2 loss = NER CE + re_weight · RE CE.

    NER: per-token cross entropy (ignore -100).
    RE:  Stage 2-004 fix — enumerate ALL ordered pairs of gold entity spans.
         Pairs that have an annotated relation get the rel id (1..7).
         Pairs without an annotated relation get NO_REL_ID = 0.
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    ner_labels = batch["ner_labels"].to(device)
    word_ids_list = batch["word_ids"]
    gold_entities_list = batch["gold_entities"]
    gold_relations_list = batch["gold_relations"]

    # Stage 2: text modality. Stage 3 will pass modality="image" or similar
    # from a different training loop.
    hidden = model.encode(modality="text", input_ids=input_ids, attention_mask=attention_mask)

    # NER loss
    ner_logits = model.forward_ner(hidden)
    if model.use_crf and model.crf is not None:
        # CRF requires: (1) mask first timestep = True, (2) no -100 in tags.
        # Use attention_mask as the CRF mask (first token [CLS] is always 1).
        # Replace -100 labels with O-tag (0) — CRF mask will handle ignoring
        # special tokens; the O-tag assignment is just to keep tags in-range.
        crf_mask = attention_mask.bool()  # (B, T)
        crf_labels = ner_labels.clone()
        crf_labels[ner_labels == -100] = 0  # O tag
        ner_loss = -model.crf(ner_logits, crf_labels, mask=crf_mask, reduction="mean")
    else:
        ner_loss = F.cross_entropy(
            ner_logits.view(-1, ner_logits.size(-1)),
            ner_labels.view(-1),
            ignore_index=-100,
        )

    # RE loss — every ordered pair of gold spans, NO_REL for unannotated pairs
    re_losses = []
    for b_idx in range(len(gold_entities_list)):
        ents = gold_entities_list[b_idx]
        rels = gold_relations_list[b_idx]
        if len(ents) < 2:
            continue

        rel_lookup = {(h, t): rid for (h, t, rid) in rels}
        spans = [(s, e) for (s, e, _) in ents]
        pairs = []
        targets = []
        for h in spans:
            for t in spans:
                if h == t:
                    continue
                pairs.append((h, t))
                targets.append(rel_lookup.get((h, t), NO_REL_ID))

        if not pairs:
            continue

        targets_t = torch.tensor(targets, device=device, dtype=torch.long)
        re_logits = model.forward_re(hidden[b_idx], word_ids_list[b_idx], pairs)
        re_losses.append(F.cross_entropy(re_logits, targets_t))

    if re_losses:
        re_loss = torch.stack(re_losses).mean()
    else:
        re_loss = ner_loss.new_tensor(0.0)

    total = ner_loss + re_weight * re_loss
    return total, ner_loss.detach(), re_loss.detach(), ner_logits
