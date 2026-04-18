"""
ELECTRA-style cooperative pre-training components.

SciBERTGenerator: 4-layer SciBERT that shares word + position embeddings
with the full 12-layer discriminator (BertKGExtractor). Trained with MLM
loss on masked positions.

ReplacedTokenDetector: per-token binary classifier on top of the
discriminator's hidden states. Predicts whether each token is original
or was replaced by the generator.

Reference: Clark et al., "ELECTRA: Pre-training Text Encoders as
Discriminators Rather Than Generators", ICLR 2020.
"""
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel


class SciBERTGenerator(nn.Module):
    """
    Small MLM generator that shares token + position embeddings with
    the discriminator backbone.

    Architecture:
        shared word_embeddings   (from discriminator, by reference)
        shared position_embeddings (from discriminator, by reference)
        4 x BertLayer            (own parameters, randomly initialized)
        MLM head                 (Linear + GELU + LN + Linear, weight-tied)
    """

    def __init__(self, disc_backbone, n_layers=4):
        """
        Args:
            disc_backbone: BertBackbone instance (the discriminator's backbone).
                           We share its word_embeddings and position_embeddings.
            n_layers: number of transformer layers for the generator.
        """
        super().__init__()
        disc_config = disc_backbone.bert.config

        # Build a small BERT with n_layers (randomly initialized body)
        gen_config = BertConfig(
            vocab_size=disc_config.vocab_size,
            hidden_size=disc_config.hidden_size,
            num_hidden_layers=n_layers,
            num_attention_heads=disc_config.num_attention_heads,
            intermediate_size=disc_config.intermediate_size,
            max_position_embeddings=disc_config.max_position_embeddings,
            type_vocab_size=disc_config.type_vocab_size,
            hidden_dropout_prob=disc_config.hidden_dropout_prob,
            attention_probs_dropout_prob=disc_config.attention_probs_dropout_prob,
        )
        self.bert = BertModel(gen_config)

        # Share embeddings by reference — the crucial design choice
        self.bert.embeddings.word_embeddings = disc_backbone.bert.embeddings.word_embeddings
        self.bert.embeddings.position_embeddings = disc_backbone.bert.embeddings.position_embeddings

        H = disc_config.hidden_size
        V = disc_config.vocab_size

        # MLM head: project hidden → vocab logits
        self.mlm_dense = nn.Linear(H, H)
        self.mlm_act = nn.GELU()
        self.mlm_ln = nn.LayerNorm(H)
        self.mlm_decoder = nn.Linear(H, V, bias=False)
        self.mlm_bias = nn.Parameter(torch.zeros(V))

        # Tie output weights to shared word_embeddings
        self.mlm_decoder.weight = self.bert.embeddings.word_embeddings.weight

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: (B, T) with [MASK] at selected positions
            attention_mask: (B, T)
        Returns:
            logits: (B, T, V) MLM logits over full vocabulary
        """
        hidden = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = self.mlm_ln(self.mlm_act(self.mlm_dense(hidden)))
        logits = self.mlm_decoder(x) + self.mlm_bias
        return logits


class ReplacedTokenDetector(nn.Module):
    """
    Per-token binary classifier: original (0) vs replaced (1).
    Applied on ALL non-special tokens — the ELECTRA efficiency advantage.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (B, T, H) from discriminator
        Returns:
            logits: (B, T) — >0 means "predict replaced"
        """
        return self.classifier(self.act(self.dense(hidden_states))).squeeze(-1)
