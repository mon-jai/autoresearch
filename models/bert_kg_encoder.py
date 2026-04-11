"""
BERT-based KG extractor for Stage 2.

Architecture:
    BERT-base
        ↓ (per-token contextualized embeddings)
        ├── NER head: Linear(hidden, NUM_BIO_TAGS) → token classification
        └── RE head:  Linear(2*hidden, NUM_RELATIONS) → span-pair classification

For Stage 2a (this file), training uses:
    L_rec = NER cross-entropy + λ · RE cross-entropy

NER predictions are made per-token (BIO scheme).
RE predictions are made per pair of GOLD entity spans during training,
and per pair of PREDICTED entity spans during evaluation.
This separation lets us decouple "extract entities" from "extract relations"
in the eval metric.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from data.scierc import NUM_BIO_TAGS, NUM_RELATIONS, BIO_TAG2ID, ID2BIO


MAX_SPAN_WIDTH = 30  # word-level cap; SciERC spans are short, sentences max ~80 words
WIDTH_EMB_DIM = 32


class BertKGExtractor(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", dropout: float = 0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.ner_head = nn.Linear(hidden, NUM_BIO_TAGS)

        # Stage 2-003: upgraded RE head with span features.
        # Input features per pair:
        #   head_repr   (H)   max-pooled head span embedding
        #   tail_repr   (H)
        #   head ⊙ tail (H)   element-wise product (interaction)
        #   |head − tail| (H) absolute difference (asymmetry)
        #   width_h     (W)   embedding of head span width (in words)
        #   width_t     (W)   embedding of tail span width (in words)
        # = 4H + 2W input dim
        self.span_width_emb = nn.Embedding(MAX_SPAN_WIDTH + 1, WIDTH_EMB_DIM)
        re_input_dim = hidden * 4 + WIDTH_EMB_DIM * 2
        self.re_head = nn.Sequential(
            nn.Linear(re_input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, NUM_RELATIONS),
        )

    def encode(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state  # (B, T, H)

    def forward_ner(self, hidden_states):
        return self.ner_head(self.dropout(hidden_states))  # (B, T, NUM_BIO_TAGS)

    def span_repr(self, hidden_states, word_ids_list, span):
        """
        Extract span representation by max-pooling subword embeddings whose
        word_ids fall within [span_start, span_end].
        Returns (H,)
        """
        word_start, word_end = span  # inclusive
        # Find subword token indices that map to words in [word_start, word_end]
        token_idx = [i for i, wid in enumerate(word_ids_list) if wid is not None and word_start <= wid <= word_end]
        if not token_idx:
            # Should not happen for gold spans, but be safe
            return hidden_states.new_zeros(hidden_states.size(-1))
        return hidden_states[token_idx].max(dim=0).values

    def _width_id(self, span):
        """Return clamped span-width id for embedding lookup."""
        s, e = span  # inclusive
        w = e - s + 1
        if w < 1:
            w = 1
        if w > MAX_SPAN_WIDTH:
            w = MAX_SPAN_WIDTH
        return w  # 1..MAX_SPAN_WIDTH; index 0 unused

    def forward_re(self, hidden_states_b, word_ids_b, pairs):
        """
        For one example in the batch:
            hidden_states_b: (T, H)
            word_ids_b: list[int|None] length T
            pairs: list of ((hs, he), (ts, te))
        Returns: (num_pairs, NUM_RELATIONS)
        """
        if not pairs:
            return hidden_states_b.new_zeros((0, NUM_RELATIONS))
        feats = []
        device = hidden_states_b.device
        for (hs, he), (ts, te) in pairs:
            head_vec = self.span_repr(hidden_states_b, word_ids_b, (hs, he))
            tail_vec = self.span_repr(hidden_states_b, word_ids_b, (ts, te))
            prod = head_vec * tail_vec
            absdiff = (head_vec - tail_vec).abs()
            wh = self.span_width_emb(torch.tensor(self._width_id((hs, he)), device=device))
            wt = self.span_width_emb(torch.tensor(self._width_id((ts, te)), device=device))
            feats.append(torch.cat([head_vec, tail_vec, prod, absdiff, wh, wt], dim=-1))
        feats = torch.stack(feats, dim=0)  # (num_pairs, 4H + 2W)
        return self.re_head(self.dropout(feats))


def compute_loss(model, batch, device, re_weight: float = 1.0):
    """
    Compute Stage 2a loss = NER CE + re_weight · RE CE.

    NER: per-token cross entropy (ignore -100).
    RE:  for each example, classify all GOLD entity span pairs.
         Pairs without an annotated relation get a "no relation" label,
         which we model as a 0 contribution by NOT including them
         (Stage 2a — keep it simple, no negative sampling).
         Pairs WITH a relation contribute CE loss against the gold rel id.
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    ner_labels = batch["ner_labels"].to(device)
    word_ids_list = batch["word_ids"]
    gold_relations = batch["gold_relations"]

    hidden = model.encode(input_ids, attention_mask)  # (B, T, H)

    # NER loss
    ner_logits = model.forward_ner(hidden)  # (B, T, C)
    ner_loss = F.cross_entropy(
        ner_logits.view(-1, ner_logits.size(-1)),
        ner_labels.view(-1),
        ignore_index=-100,
    )

    # RE loss — only on examples that have at least one gold relation
    re_losses = []
    for b_idx in range(len(gold_relations)):
        rels = gold_relations[b_idx]
        if not rels:
            continue
        pairs = [(h, t) for (h, t, _) in rels]
        targets = torch.tensor([rid for (_, _, rid) in rels], device=device, dtype=torch.long)
        re_logits = model.forward_re(hidden[b_idx], word_ids_list[b_idx], pairs)
        re_losses.append(F.cross_entropy(re_logits, targets))

    if re_losses:
        re_loss = torch.stack(re_losses).mean()
    else:
        re_loss = ner_loss.new_tensor(0.0)

    total = ner_loss + re_weight * re_loss
    return total, ner_loss.detach(), re_loss.detach(), ner_logits
