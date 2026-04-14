"""
Stage 2b realism critic — small MLP head that reads the SciBERT [CLS]
embedding and predicts P(text is real arXiv) vs P(text is Decoder-D synth).

This module deliberately does NOT carry its own BERT body. The Stage 2b
training loop calls `model.encode(...)` from the existing BertKGExtractor,
extracts the `[CLS]` token (= position 0 of last hidden state), and feeds
it here. The critic and the encoder share the same SciBERT params:

  - Memory: only one BERT body on GPU
  - Inductive coupling: the same representation must be useful for
    NER + RE *and* for real/fake discrimination. This is the
    "encoder learns from the adversarial signal" pathway that
    Stage 2b is supposed to validate.

Stage 2c will keep this critic but additionally let Decoder D's gradient
flow back via REINFORCE.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RealismCritic(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, 1)  # logit for P(real)

    def forward(self, cls_hidden: torch.Tensor) -> torch.Tensor:
        """
        cls_hidden: (B, H)  — the [CLS] vector from a BERT pass
        Returns:    (B,)    — raw logits; > 0 means "predict real"
        """
        feat = self.act(self.fc1(cls_hidden))
        feat = self.drop(feat)
        return self.fc2(feat).squeeze(-1)

    def features(self, cls_hidden: torch.Tensor) -> torch.Tensor:
        """Return intermediate features (after GELU, before dropout/fc2).
        Used for feature matching loss in Gumbel-STE training."""
        return self.act(self.fc1(cls_hidden))


def critic_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """
    Standard discriminator loss:
        BCE(real, target=1) + BCE(fake, target=0)  averaged.
    Both real_logits and fake_logits are (B,) raw logits.
    """
    real_targets = torch.ones_like(real_logits)
    fake_targets = torch.zeros_like(fake_logits)
    loss_real = F.binary_cross_entropy_with_logits(real_logits, real_targets)
    loss_fake = F.binary_cross_entropy_with_logits(fake_logits, fake_targets)
    return (loss_real + loss_fake) / 2
