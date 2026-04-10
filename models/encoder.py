import torch
import torch.nn as nn
from .transformer import Block, has_ve

class KGEncoder(nn.Module):
    """
    Phase 2 (Compilation): Reconstructs Graph entities and relations 
    from synthesized / unstructured data distributions.
    Optimizes for minimizing L_rec (Graph Edit Distance surrogate).
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        
        # Reconstructs Entity and Relation probabilities
        self.entity_head = nn.Linear(config.n_embd, config.num_entities, bias=False)
        self.relation_head = nn.Linear(config.n_embd * 2, config.num_relations, bias=False)

        # exp47: Projection heads for contrastive learning (SimCLR-style)
        # Maps logits to a better contrastive space without losing representation info
        self.entity_proj = nn.Sequential(
            nn.Linear(config.num_entities, config.num_entities),
            nn.GELU(),
            nn.Linear(config.num_entities, config.num_entities),
        )
        self.relation_proj = nn.Sequential(
            nn.Linear(config.num_relations, config.num_relations),
            nn.GELU(),
            nn.Linear(config.num_relations, config.num_relations),
        )
        
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000):
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        return cos[None, :, None, :], sin[None, :, None, :]

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def forward(self, input_ids):
        B, T = input_ids.size()
        x = self.transformer.wte(input_ids)
        
        cos_sin = (self.cos[:, :T, ...], self.sin[:, :T, ...])
        for i, block in enumerate(self.transformer.h):
            ve = self.value_embeds[str(i)](input_ids) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
            
        # Global pooling over sequence for graph reconstruction
        graph_repr = x.mean(dim=1)
        
        entity_logits = self.entity_proj(self.entity_head(graph_repr))

        # Simplified relation head modeling pairwise connections
        # (In practice entails complex tensor products, here simulated as concat self)
        graph_repr_paired = torch.cat([graph_repr, graph_repr], dim=-1)
        relation_logits = self.relation_proj(self.relation_head(graph_repr_paired))

        return entity_logits, relation_logits

class KBGANGenerator(nn.Module):
    """
    KBGAN-style probability-based generator (inspired by DistMult/ComplEx).
    Generates 'Hard Negative' entities/relations via REINFORCE policy gradient,
    replacing the naive MSE training that caused gradient starvation.
    """
    def __init__(self, config):
        super().__init__()
        self.num_entities = config.num_entities
        self.num_relations = config.num_relations
        # Policy networks: output logits over candidate replacements
        self.ent_policy = nn.Sequential(
            nn.Linear(config.num_entities, config.num_entities * 2),
            nn.LeakyReLU(0.1),
            nn.Linear(config.num_entities * 2, config.num_entities)
        )
        self.rel_policy = nn.Sequential(
            nn.Linear(config.num_relations, config.num_relations * 2),
            nn.LeakyReLU(0.1),
            nn.Linear(config.num_relations * 2, config.num_relations)
        )

    def forward(self, gt_entities, gt_relations):
        """
        Returns hard negatives and log_probs for REINFORCE training.
        hard_neg_entities/relations: perturbed features (differentiable via Gumbel)
        ent_log_prob/rel_log_prob: log probabilities for policy gradient
        """
        # Entity policy: produce distribution, sample via Gumbel-Softmax
        ent_logits = self.ent_policy(gt_entities)
        ent_probs = torch.softmax(ent_logits, dim=-1)
        hard_neg_entities = torch.nn.functional.gumbel_softmax(ent_logits, tau=0.5, hard=True)
        ent_log_prob = (ent_probs * hard_neg_entities).sum(dim=-1).clamp(min=1e-8).log().mean()

        # Relation policy
        rel_logits = self.rel_policy(gt_relations)
        rel_probs = torch.softmax(rel_logits, dim=-1)
        hard_neg_relations = torch.nn.functional.gumbel_softmax(rel_logits, tau=0.5, hard=True)
        rel_log_prob = (rel_probs * hard_neg_relations).sum(dim=-1).clamp(min=1e-8).log().mean()

        return hard_neg_entities, hard_neg_relations, ent_log_prob, rel_log_prob
