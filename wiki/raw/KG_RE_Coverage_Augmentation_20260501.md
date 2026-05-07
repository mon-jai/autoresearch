# KG/RE Coverage Augmentation Notes

Date: 2026-05-01

## Why This Was Triggered

After Qwen timeout repair, four KG/RAG variants remained stuck:

- verified KG pruning: hybrid 20%
- confidence KG thresholding: hybrid 30%, KG-only 0%
- ACCORD-template extracted KG: hybrid 0%
- oracle-entity RE graph: hybrid 30%, KG 2-hop 10%

Diagnostics showed:

- entity text recall only 11.6% for normal inference
- oracle entity spans still produce Triple F1 only 0.084
- relation recall remains low even with gold entities

This triggers the Knowledge Pipeline threshold of two stuck experiments.

## Literature Signal

The relevant literature points away from loss/head tweaks and toward schema-aware, entity-anchored data augmentation:

- ODDA (ACL Findings 2025): use model-behavior-driven demonstration selection, schema/attribute constraints, and diversity selection for low-resource RE augmentation.
- GDA (ACL Findings 2023): use entity hints and generation objectives that preserve both semantic consistency and syntax structure.
- Realistic Low-resource RE Benchmark (EMNLP Findings 2022): data augmentation can complement baselines, but self-training is unreliable and multi-triple/cross-context extraction remains hard.
- LAL-JER (2023): label-aware prompts with semantic label descriptions improve joint entity/relation extraction robustness.
- LLM-ACNC (2025): requirement-text KG construction benefits from LLM augmentation plus quality filtering, a close analogue to CODE-ACCORD regulatory text.

## Next Experiment Recommendation

Build a targeted ACCORD augmentation set instead of changing the extractor architecture:

1. Sample under-recalled gold triples, especially `selection`, `part-of`, `necessity`, `equal`, and `greater-equal`.
2. Prompt Qwen with relation definitions and require JSON output containing `sentence`, `head`, `relation`, `tail`.
3. Filter generated examples by exact entity containment and schema validity.
4. Train DeBERTa-base or DeBERTa-large with the augmented data only after the generated set passes structural validation.

Avoid naive relation replay and unfiltered self-training.
