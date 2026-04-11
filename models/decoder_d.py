"""
Decoder D — the LLM that paraphrases (head, relation, tail) triples into
natural scientific sentences.

This file defines the shared base class (`QwenDecoderBase`) plus the Stage 2b
frozen variant (`FrozenQwenDecoder`). The Stage 2c LoRA-tunable variant lives
in `models/decoder_d_lora.py` and inherits from `QwenDecoderBase`.

Default model: Qwen/Qwen2.5-0.5B-Instruct
- 500M params, fits easily on GB10 alongside SciBERT + critic
- Instruct-tuned → handles the prompt template well
"""
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT_TEMPLATE = (
    "Write one short scientific English sentence that expresses this "
    "knowledge graph triple. Do not add explanations, lists, or bullet "
    "points. Use natural prose.\n\n"
    "Triple: ({head}, {rel}, {tail})\n"
    "Sentence:"
)


# Human-readable mapping of SciERC relation types — Qwen handles names
# better than the all-caps codes.
RELATION_PHRASES = {
    "USED-FOR":     "is used for",
    "FEATURE-OF":   "is a feature of",
    "HYPONYM-OF":   "is a kind of",
    "EVALUATE-FOR": "is evaluated for",
    "PART-OF":      "is part of",
    "COMPARE":      "is compared with",
    "CONJUNCTION":  "and",
}


def humanize_relation(rel: str) -> str:
    return RELATION_PHRASES.get(rel, rel.lower().replace("-", " "))


class QwenDecoderBase:
    """
    Shared base for FrozenQwenDecoder (Stage 2b) and LoRAQwenDecoder (Stage 2c).

    Responsibilities:
      - tokenizer setup (left-padding for decoder-only batched generate)
      - model load with dtype + device
      - prompt building from (h, r, t) triples
      - generate_batch() for inference-only text sampling
      - decode_new_tokens() helper for post-processing

    Subclasses decide whether base params are frozen (Stage 2b) and whether
    to add trainable adapters (Stage 2c LoRA).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cuda",
        dtype: "torch.dtype" = torch.bfloat16,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Decoder-only LLMs require LEFT padding for batched generation;
        # otherwise the new tokens get pasted in the middle of the prompt.
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(device)
        self.device = device

    # ---- prompt / decode helpers --------------------------------------

    def build_prompts(self, triples: List[Tuple[str, str, str]]) -> List[str]:
        return [
            PROMPT_TEMPLATE.format(head=h, rel=humanize_relation(r), tail=t)
            for (h, r, t) in triples
        ]

    def encode_prompts(self, prompts: List[str], max_length: int = 128):
        return self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self.device)

    def decode_new_tokens(self, generated, prompt_len: int) -> List[str]:
        """Strip prompt prefix, keep first line, cap length, avoid empty."""
        sentences = []
        for gen in generated:
            new_tokens = gen[prompt_len:]
            sent = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            sent = sent.strip().split("\n")[0]
            sent = sent[:300]
            if not sent:
                sent = "."
            sentences.append(sent)
        return sentences

    # ---- inference-only path ------------------------------------------

    @torch.no_grad()
    def generate_batch(
        self,
        triples: List[Tuple[str, str, str]],
        max_new_tokens: int = 40,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> List[str]:
        """
        triples: list of (head_str, rel_str, tail_str)
        Returns: list of plain-text sentences (one per triple).
        """
        if not triples:
            return []
        prompts = self.build_prompts(triples)
        enc = self.encode_prompts(prompts)
        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.decode_new_tokens(out, prompt_len=enc["input_ids"].shape[1])


class FrozenQwenDecoder(QwenDecoderBase):
    """
    Stage 2b: all params frozen, no gradients ever flow through.
    Behavior is identical to the pre-refactor class — only the code lives
    in the base class now.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cuda",
        dtype: "torch.dtype" = torch.bfloat16,
    ):
        super().__init__(model_name=model_name, device=device, dtype=dtype)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
