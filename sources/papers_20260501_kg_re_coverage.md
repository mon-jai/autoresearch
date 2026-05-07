# Literature Lookup — KG/RE Coverage Bottleneck (2026-05-01)

## Databases Queried

- OpenAlex `/works` search for `"relation extraction" "data augmentation" "low-resource"` with `from_publication_date:2020-01-01`.
- OpenAlex `/works` search for `"large language model" "relation extraction" "data augmentation"` with `from_publication_date:2023-01-01`.
- Semantic Scholar paper search was attempted but returned HTTP 429 without an API key.
- arXiv API query for low-resource/data-augmentation relation extraction returned no useful results in the sampled response.

## Relevant Results

### ODDA: An OODA-Driven Diverse Data Augmentation Framework for Low-Resource Relation Extraction

- Year: 2025
- DOI: `10.18653/v1/2025.findings-acl.15`
- Open access: https://aclanthology.org/2025.findings-acl.15.pdf
- Key idea: LLM augmentation should explicitly optimize diversity across samples and relations; the framework selects demonstrations from model behavior and applies schema/attribute constraints before selecting a final augmented dataset.
- Relevance: directly matches ACCORD’s failure mode: low relation recall and narrow coverage, especially for `necessity` and `selection`.

### GDA: Generative Data Augmentation Techniques for Relation Extraction Tasks

- Year: 2023
- DOI: `10.18653/v1/2023.findings-acl.649`
- Open access: https://aclanthology.org/2023.findings-acl.649.pdf
- Key idea: relation-text augmentation should preserve semantic consistency and syntax structure; GDA uses complementary modules and entity hints as prior knowledge.
- Relevance: supports entity-pair anchored generation rather than free-form paraphrases.

### Towards Realistic Low-resource Relation Extraction: A Benchmark with Empirical Baseline Study

- Year: 2022
- DOI: `10.18653/v1/2022.findings-emnlp.29`
- Open access: https://aclanthology.org/2022.findings-emnlp.29.pdf
- Key idea: prompt tuning helps, data augmentation can help, but self-training is not consistently reliable; cross-sentence and multi-triple contexts remain difficult.
- Relevance: cautions against naive self-training and supports targeted data augmentation/filtering.

### LAL-JER: Label-Aware Learning for Adaptive Joint Entity and Relation Extraction with LLM data augmentation

- Year: 2023
- DOI: `10.1145/3640912.3640993`
- Key idea: label-aware prompts incorporate semantic meaning of entity/relation types; LLM augmentation improves robustness and unseen label behavior.
- Relevance: ACCORD relation labels (`necessity`, `selection`, numeric comparisons) need explicit label descriptions, not only label IDs.

### LLM-ACNC: Aerospace Requirement Texts Knowledge Graph Construction Utilizing Large Language Model

- Year: 2025
- DOI: `10.3390/aerospace12060463`
- Open access: https://www.mdpi.com/2226-4310/12/6/463/pdf
- Key idea: requirement-text KG construction combines LLM data augmentation, BERTScore filtering, continual learning/token-index encoding, and dynamic few-shot prompting.
- Relevance: closest domain analogue to CODE-ACCORD regulatory/requirement text.

## Implementation Implication

The next best experiment should be data-volume and schema-awareness, not another head/loss tweak:

1. Generate ACCORD-style sentences anchored to gold `(head, relation, tail)` triples for under-recalled relations.
2. Include relation label descriptions in prompts, especially `necessity`, `selection`, `part-of`, `equal`, `greater-equal`.
3. Validate generated examples structurally: head and tail strings must appear, relation label must be from the ACCORD schema, and a lightweight verifier should reject ambiguous or missing-entity generations.
4. Start with targeted augmentation for `selection` and `part-of`, because oracle-entity diagnostics show those relations remain weak even when entity spans are gold.
