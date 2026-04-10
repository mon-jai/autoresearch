import torch
import torch.nn as nn
from .transformer import Block, has_ve

class KGDecoder(nn.Module):
    """
    Phase 1 (Decompilation): Generates synthesized non-structured data 
    (e.g., text representations) from a Ground-Truth Knowledge Graph.
    This acts as the Generator in the Adversarial loop.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        
        # We embed the graph entities/relations into the sequence
        self.graph_emb = nn.Embedding(config.num_entities + config.num_relations, config.n_embd)
        
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
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
        # Default device to CPU, moves to GPU automatically if module moves
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

    def forward(self, kg_features, input_ids):
        # This is a simplified forward pass for the dual-adversarial loop
        B, T = input_ids.size()
        x = self.transformer.wte(input_ids)
        
        # Inject graph features
        # Assuming kg_features is (B, n_embd) from some GNN or simple pooling
        x = x + kg_features.unsqueeze(1)
        
        cos_sin = (self.cos[:, :T, ...], self.sin[:, :T, ...])
        for i, block in enumerate(self.transformer.h):
            ve = self.value_embeds[str(i)](input_ids) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
            
        logits = self.lm_head(x)
        return logits


class RealismCritic(nn.Module):
    """
    Discriminator that ensures synthesized data matches real-world distributions.
    Minimizes L_real.
    """
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.encoder = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.ReLU(),
            nn.Linear(config.n_embd, 1)
        )
        
    def forward(self, input_ids):
        x = self.wte(input_ids)
        # Pool sequence
        x = x.mean(dim=1) 
        score = self.encoder(x)
        return score
