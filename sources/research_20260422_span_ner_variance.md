
================================================================================
Query 1: span-based joint entity and relation extraction training variance reduction multi-seed stability techniques 2024 2025
Timestamp: 2026-04-22 05:01:33
Backend: perplexity | Model: perplexity/sonar-pro-search
================================================================================
Span-based joint entity and relation extraction (ERE) models, which identify entity spans and their relations in a single pass, face training variance issues due to stochastic optimization in deep learning, leading to instability across random seeds. Recent research (2020-2026) emphasizes multi-seed evaluations for stability assessment and general techniques like variance-reduced gradients (e.g., SVRG, SAGA) or ensemble methods to mitigate this, though direct applications to span-based ERE remain limited.[1][8] High-impact studies prioritize reproducibility, with calls for 10-80 seeds depending on sample size to stabilize results.

## Foundational Span-Based ERE Models

SpERT, a seminal span-based model using BERT embeddings for joint entity recognition and relation classification, sets the benchmark but notes sensitivity to negative sampling and pre-training, implying seed-dependent variance in training.[1] Published in arXiv 2019 (highly cited, ~500+ citations estimated from influence; authors from established NLP groups like IBM Research; tier-2 venue via COLING presentation), it outperforms priors by 5% F1 via localized context but lacks explicit multi-seed analysis, a common limitation pre-2020.[1][3]

A 2020 attention-based span model enhances semantic representations for candidate spans, addressing overlap issues in joint extraction.[3] From COLING 2020 (ACL anthology; mid-tier NLP conference), it shares span semantics across tasks but reports no variance reduction, highlighting early field focus on accuracy over stability.[3]

## Recent Advances in Joint ERE (2024-2025)

A 2024 Nature Scientific Reports paper proposes a decomposition-based model splitting head/tail entity extraction with BERT, handling overlaps effectively on NYT-multi.[6] (Venue tier-1; F1 gains on overlapping triples; authors from Chinese institutions; DOI: s41598-024-51559-w). It improves robustness but omits multi-seed reporting, a noted gap amid rising reproducibility concerns.[6]

In CCL 2024, a bidirectional update with long-term memory gate fuses entity/relation info iteratively, achieving SOTA on Chinese datasets.[7] (Tier-3 conference; enhances fine-grained info without order prediction; no citation count yet). Limitations include potential info loss from over-focusing relations, unaddressed by seed variance.[7]

A 2025 PMC paper on entity-relation extraction with dependency parsing and graph NNs (MGRel) boosts F1 by 1-3% on NYT/DuIE via multi-feature fusion.[9] (Open-access; tier-3; DOI via PMC12852732). It mitigates boundary ambiguity but ignores training stability.[9]

## Variance Reduction and Stability Techniques

Direct variance reduction in ERE training is rare, but a 2022 hierarchical RL model for overlapping ERE uses advantage functions to reduce policy gradient variance. (Reduces estimator variance via value subtraction; experiments show superiority; obscure venue but relevant methodology). Key finding: Subtracting high/low-level policy values stabilizes gradients for joint tasks.

General deep learning variance reduction via momentum (SVRG/SAGA hybrids) accelerates non-convex optimization, applicable to ERE's Transformer training. (arXiv 2021; cited in ML optimization; authors from Mila/Quebec AI). Multi-momentum with SGD cuts variance without full batches, ideal for span enumeration's compute cost.

PET reconstruction (analogous stochastic task) shows SVRG outperforming SGD in run-to-run stability across seeds, with low NRMSE variance at optimal subsets.[8] (PMC 2025; tier-3 medical imaging; confirms stochastic methods' seed sensitivity). Implications for ERE: Subset-based gradients could stabilize span sampling.[8]

## Reproducibility and Multi-Seed Practices

A 2025 reproducibility study in NER stresses seed control (e.g., 40 epochs fixed seeds) for F1 stability. (Case study; highlights epoch/seed interplay; emerging NLP repro focus).

Causal inference work (2024 PMC, cited 10+ times) reveals seed-driven CI divergence in ML estimators, recommending 20-80 seeds/cross-fitting for <5% instability. (High-impact methodological; tier-1 PMC; authors clinical ML experts). For ERE's doubly robust-like losses, this implies multi-seed averaging essential; small samples need more seeds.

ReproNLP 2025 shared task quantifies poor system reproducibility (CV* ~60), urging standardized seeds in NLP evals. (ACL 2025; tier-2; exposes eval fragility). ERE benchmarks like NYT lack this, risking inflated SOTA claims.

| Technique | Description | ERE Applicability | Evidence/Source |
|-----------|-------------|-------------------|-----------------|
| Multi-Seed Averaging | Run 10-80 seeds, average F1 | Stabilizes CIs in small-sample ERE | Low variance at n=20  |
| SVRG/SAGA | Variance-reduced SGD | Reduces gradient noise in span training | Faster convergence, seed-stable [8] |
| Advantage Functions (RL) | Subtract baselines for low variance | Overlapping span extraction | Effective on unstructured text  |
| Momentum Clustering | Multi-momentum on datasets | Non-convex ERE optim | Zero-variance asymptotics  |

## Limitations and Gaps

No top-tier (Nature/Science) papers directly tackle span-based ERE variance; most prioritize F1 over stability.[6] Conflicts: Decomposition aids overlap but may amplify seed variance via subtasks.[6] Few 2024-2025 works report seeds, echoing NLP "reproducibility crisis."

**Future Directions:** Integrate SVRG into span models (e.g., SpERT++); mandate multi-seed benchmarks in ACL/EMNLP; explore seed-agnostic losses like bipartite matching. Gaps include low-resource ERE stability and cross-lingual seeds. High-citation repro guidelines could standardize practices.

Usage: {'prompt_tokens': 545, 'completion_tokens': 1278, 'total_tokens': 1823, 'cost': 0.0308, 'is_byok': False, 'prompt_tokens_details': {'cached_tokens': 0, 'cache_write_tokens': 0, 'audio_tokens': 0, 'video_tokens': 0}, 'cost_details': {'upstream_inference_cost': 0.0308, 'upstream_inference_prompt_cost': 0.001635, 'upstream_inference_completions_cost': 0.029165}, 'completion_tokens_details': {'reasoning_tokens': 0, 'image_tokens': 0, 'audio_tokens': 0}}
