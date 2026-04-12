"""
Stage 2c Decoder D — LoRA-tunable Qwen.

The base transformer weights stay frozen; only LoRA adapters (on attention
projections) receive gradients. The training signal flows via REINFORCE:
we sample a sentence from the decoder, score it with
  (a) the realism critic, and
  (b) a triple-recovery scorer that re-extracts triples from the sentence,
sum those into a scalar reward, and backprop through the adapter using the
per-token logprobs of the sampled sequence.

Key method: `sample_with_logprob(triples)` — returns sampled text along
with the summed token logprob under the LoRA policy *and* under the frozen
base policy (for a KL penalty). This is the contract the REINFORCE loop
depends on.
"""
from typing import List, Tuple

import torch
from peft import LoraConfig, TaskType, get_peft_model

from models.decoder_d import QwenDecoderBase


class LoRAQwenDecoder(QwenDecoderBase):
    """
    Stage 2c decoder. LoRA adapters on Qwen; base weights frozen.
    The base Qwen is kept in a separate reference (`base_model_for_kl`)
    so we can compute KL(LoRA‖base) on sampled tokens without disabling
    the adapters mid-forward.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cuda",
        dtype: "torch.dtype" = torch.bfloat16,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_target_modules: Tuple[str, ...] = ("q_proj", "v_proj", "o_proj"),
        lora_dropout: float = 0.05,
    ):
        super().__init__(model_name=model_name, device=device, dtype=dtype)

        # Keep a second frozen copy of the base model around for KL estimation.
        # This is the cleanest way to get base logprobs on sampled tokens
        # without toggling adapters. 0.5B × 2 ≈ 2GB bf16, trivial on GB10.
        from transformers import AutoModelForCausalLM
        self.base_model_for_kl = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype,
        ).to(device)
        self.base_model_for_kl.eval()
        for p in self.base_model_for_kl.parameters():
            p.requires_grad = False

        # Wrap the primary model in LoRA. PEFT freezes base params internally.
        cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=list(lora_target_modules),
            lora_dropout=lora_dropout,
        )
        self.model = get_peft_model(self.model, cfg)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"[LoRAQwenDecoder] trainable={trainable:,} / total={total:,} "
              f"({100 * trainable / total:.3f}%)")

    # -------------------------------------------------------------------
    # REINFORCE sampling path
    # -------------------------------------------------------------------

    def sample_with_logprob(
        self,
        triples: List[Tuple[str, str, str]],
        max_new_tokens: int = 40,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ):
        """
        Sample one sentence per triple (K=1 per design Q3 default) and
        return the info REINFORCE needs.

        Returns a dict with:
          sentences:   List[str]               decoded sampled texts
          lora_logprob: (B,) float tensor       sum of LoRA logprob on sampled tokens
                                                (grad flows here → REINFORCE)
          base_logprob: (B,) float tensor       sum of frozen-base logprob on same tokens
                                                (no grad, used for KL)
          kl:          (B,) float tensor       lora_logprob − base_logprob, scalar estimate
                                                of KL per sequence
        """
        if not triples:
            empty = torch.zeros(0, device=self.device)
            return {
                "sentences": [],
                "lora_logprob": empty,
                "base_logprob": empty,
                "kl": empty,
            }

        prompts = self.build_prompts(triples)
        enc = self.encode_prompts(prompts)
        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]
        prompt_len = input_ids.shape[1]

        # 1. Sample sequences WITHOUT gradient (just to pick tokens).
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # out: (B, prompt_len + gen_len)
        gen_tokens = out[:, prompt_len:]                # (B, G)
        gen_len = gen_tokens.shape[1]
        if gen_len == 0:
            # Nothing was generated (edge case). Return zero-grad placeholders.
            z = torch.zeros(len(triples), device=self.device)
            return {
                "sentences": self.decode_new_tokens(out, prompt_len),
                "lora_logprob": z.clone().requires_grad_(True),
                "base_logprob": z,
                "kl": z,
            }

        # Build a mask over generated tokens that ignores padding.
        pad_id = self.tokenizer.pad_token_id
        gen_mask = (gen_tokens != pad_id).float()        # (B, G)

        # 2. Score the full sequence through LoRA model WITH grad to get
        #    per-token logprobs on the sampled tokens.
        full_attn = torch.cat(
            [attn, torch.ones_like(gen_tokens)], dim=1
        )
        lora_logits = self.model(
            input_ids=out, attention_mask=full_attn,
        ).logits                                         # (B, L, V)

        # Shift: logits at position t predict token at t+1.
        # We want logprobs over tokens at positions [prompt_len .. prompt_len+G-1],
        # predicted by logits at positions [prompt_len-1 .. prompt_len+G-2].
        shift_logits = lora_logits[:, prompt_len - 1 : prompt_len - 1 + gen_len, :]
        shift_logprobs = shift_logits.log_softmax(dim=-1)
        lora_token_lp = shift_logprobs.gather(
            dim=-1, index=gen_tokens.unsqueeze(-1),
        ).squeeze(-1)                                    # (B, G)
        lora_token_lp = lora_token_lp * gen_mask
        lora_seq_lp = lora_token_lp.sum(dim=1)           # (B,)

        # 3. Same with the frozen base, no grad.
        with torch.no_grad():
            base_logits = self.base_model_for_kl(
                input_ids=out, attention_mask=full_attn,
            ).logits
            base_shift = base_logits[:, prompt_len - 1 : prompt_len - 1 + gen_len, :]
            base_shift_lp = base_shift.log_softmax(dim=-1)
            base_token_lp = base_shift_lp.gather(
                dim=-1, index=gen_tokens.unsqueeze(-1),
            ).squeeze(-1) * gen_mask
            base_seq_lp = base_token_lp.sum(dim=1)       # (B,)

        # KL(LoRA‖base) ≈ E_lora[logp_lora − logp_base]. We only have one
        # sample per sequence, so the estimator is (lora_lp − base_lp).sum.
        kl = (lora_seq_lp - base_seq_lp.detach())

        sentences = self.decode_new_tokens(out, prompt_len)
        return {
            "sentences": sentences,
            "lora_logprob": lora_seq_lp,    # grad here
            "base_logprob": base_seq_lp,    # no grad
            "kl": kl,                       # grad here (via lora_lp)
        }

    # -------------------------------------------------------------------
    # Checkpoint helpers
    # -------------------------------------------------------------------

    def save_adapters(self, path: str):
        self.model.save_pretrained(path)

    def load_adapters(self, path: str):
        """Load adapter weights from disk into the existing LoRA model.

        The __init__ already wrapped self.model in PeftModel. We just
        need to load the adapter state_dict, NOT wrap a second time
        (which would produce the nested-PeftModel "missing adapter keys"
        warning).
        """
        import os, torch as _torch
        adapter_path = os.path.join(path, "adapter_model.bin")
        if not os.path.exists(adapter_path):
            # Try safetensors format
            adapter_path = os.path.join(path, "adapter_model.safetensors")
        if os.path.exists(adapter_path) and adapter_path.endswith(".bin"):
            state_dict = _torch.load(adapter_path, map_location="cpu")
        else:
            from safetensors.torch import load_file
            state_dict = load_file(adapter_path)
        # PEFT adapter keys in the saved file use the inner model's
        # namespace. set_peft_model_state_dict handles the mapping.
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(self.model, state_dict)
