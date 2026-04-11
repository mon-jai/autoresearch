"""
Stage 2b Decoder D — frozen LLM that paraphrases (head, relation, tail) triples
into natural scientific sentences.

NO gradients flow through this. Stage 2c will subclass / replace this with a
LoRA-tunable version. Stage 2b's job is just to produce sentences for the
critic to compare against held-out arXiv text.

Default model: Qwen/Qwen2.5-0.5B-Instruct
- 500M params, fits easily on GB10 alongside SciBERT + critic
- Instruct-tuned → handles the prompt template well
- Validated on the DGX env (qwen family is the workhorse for this project)
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


class FrozenQwenDecoder:
    """
    Wraps a frozen instruction-tuned Qwen model. Call .generate_batch() to
    produce paraphrases of (head, rel, tail) triples.

    Stage 2b uses this purely as a fixed text source. Stage 2c will replace
    or subclass to allow LoRA gradients.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
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
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.device = device

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

        prompts = [
            PROMPT_TEMPLATE.format(head=h, rel=humanize_relation(r), tail=t)
            for (h, r, t) in triples
        ]
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(self.device)

        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Strip the prompt portion of each output
        prompt_len = enc["input_ids"].shape[1]
        sentences = []
        for gen in out:
            new_tokens = gen[prompt_len:]
            sent = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            sent = sent.strip().split("\n")[0]  # first line only
            sent = sent[:300]  # safety cap
            if not sent:
                sent = "."  # never return empty string (would break tokenization)
            sentences.append(sent)

        return sentences
