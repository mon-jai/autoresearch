# Small-Dataset Joint NER+RE Methods (2023-2025)

Literature survey compiled 2026-04-25.
Context: CODE-ACCORD dataset (720 train examples), span-based (SpERT-like) architecture,
BERT-base encoder, Triple F1 ceiling = 0.329 after 48+ experiments.

---

## 1. LLM-Based Data Augmentation for Joint NER+RE

### 1.1 Enhancing Low-Resource Joint Entity and Relation Extraction Using LLMs within Semi-Supervised Learning

- **Authors**: (Multiple, see TKDD)
- **Year/Venue**: 2025, ACM Transactions on Knowledge Discovery from Data (TKDD)
- **URL**: https://dl.acm.org/doi/abs/10.1145/3799423

**Key technique**: Combines semi-supervised learning with LLM-generated augmentation.
LLMs generate semantically coherent augmented data from *unlabeled* samples. These,
together with limited labeled data, feed a consistency-regularization + pseudo-labeling
SSL framework. An iterative refinement loop uses SSL performance to improve LLM
parameter-efficient fine-tuning, progressively improving both augmentation quality
and extraction accuracy.

**Reported gains**: Significantly outperforms SOTA on four benchmark datasets when
labeled data is scarce. ~17% improvement on TACRED/RE-TACRED with only 3% labeled data.

**Applicability**: HIGH. Directly targets joint ERE with small labeled sets. The SSL
framework is model-agnostic -- could wrap around our SpERT pipeline. Requires access
to unlabeled in-domain text (building regulations, standards documents) which is
available for CODE-ACCORD domain.

---

### 1.2 PGA-SciRE: Harnessing LLM on Data Augmentation for Enhancing Scientific Relation Extraction

- **Authors**: Yang Zhou, Shimin Shan, Hongkui Wei, Zhehuan Zhao, Wenshuo Feng
- **Year/Venue**: 2024, arXiv:2405.20787 (also JCIP 2025)
- **URL**: https://arxiv.org/abs/2405.20787

**Key technique**: Two-mode LLM augmentation using GPT-3.5:
(1) **Paraphrase mode** -- rephrases original training sentences while preserving
entity/relation annotations.
(2) **Generation mode** -- generates entirely new sentences given relation type and
entity pair as seed, producing implicit-relation sentences.
Both augmented sets are mixed with original data for training.

**Reported gains**: F1 improvements on three mainstream scientific RE models. Specific
numbers vary by model; demonstrates consistent benefit of both augmentation modes.

**Applicability**: HIGH. Directly relevant to scientific domain RE. The paraphrase +
generation dual strategy is straightforward to apply: feed CODE-ACCORD triplets to
GPT-4o, collect paraphrased and generated sentences, re-annotate programmatically,
add to SpERT training set. Low implementation cost.

---

### 1.3 Making LLMs as Fine-Grained Relation Extraction Data Augmentor (ConsistRE)

- **Authors**: Yifan Zheng, Wenjun Ke, Qi Liu, Yuting Yang, Ruizhuo Zhao, Dacheng Feng, Jianwei Zhang, Zhi Fang
- **Year/Venue**: 2024, IJCAI-24
- **URL**: https://www.ijcai.org/proceedings/2024/736

**Key technique**: ConsistRE framework maintains *context consistency* when generating
augmented RE data with LLMs. The key insight is that naive LLM generation often
breaks relational dependencies -- ConsistRE enforces that generated sentences preserve
the relation semantics of the seed entity pair.

**Reported gains**: +1.48% F1 on SemEval, +5.48% on TACRED, +3.16% on TACREV over
best prior augmentation methods.

**Applicability**: MEDIUM-HIGH. The context-consistency constraint is relevant when
augmenting CODE-ACCORD data, where building regulation language has strict semantics.
However, the paper focuses on RE only (not joint NER+RE).

---

### 1.4 LLM-Based Data Annotation and Augmentation for NER and RE Models Enhancement

- **Authors**: (Multiple, see Springer)
- **Year/Venue**: 2025, Springer LNCS
- **URL**: https://link.springer.com/chapter/10.1007/978-3-031-96522-7_12

**Key technique**: Uses LLMs to both annotate and augment training data for NER and RE
models, reducing manual annotation bottleneck.

**Applicability**: MEDIUM. General framework, less specific methodology details
available from search.

---

### 1.5 Pushing the Limits of Low-Resource NER Using LLM Artificial Data Generation

- **Authors**: Joan Santoso, Patrick Sutanto, Billy Cahyadi, Esther Setiawan
- **Year/Venue**: 2024, Findings of ACL 2024
- **URL**: https://aclanthology.org/2024.findings-acl.575/

**Key technique**: Uses open-source LLMs to generate NER training data from only a few
labeled examples. Also addresses class imbalance by over-generating for rare entity types.

**Reported gains**: Significant improvement over baselines on diverse low-resource NER
datasets. Consistent macro-F1 improvements, especially for tail classes.

**Applicability**: MEDIUM. NER-only (not joint), but the class-imbalance augmentation
strategy is directly useful for CODE-ACCORD where some entity types are rare.

---

## 2. Semi-Supervised and Self-Training Approaches

### 2.1 CSJE: A Curriculum-Guided Semi-Supervised Learning Framework for Joint Entity and Relation Extraction

- **Authors**: (See Knowledge-Based Systems)
- **Year/Venue**: 2026, Knowledge-Based Systems
- **URL**: https://www.sciencedirect.com/science/article/abs/pii/S0950705126004867

**Key technique**: Two-phase framework:
(1) **Curriculum learning** on labeled data: partitions training examples by a difficulty
metric based on entity-relation overlap complexity (single entities -> overlapping
entities -> overlapping relations), training from simple to complex.
(2) **Self-training** with adaptive dynamic threshold on unlabeled data: generates
pseudo-labels, selects high-confidence ones, iteratively retrains.

**Reported gains**: +1-5% F1 over supervised baselines in low-resource settings. High
generalizability and compatibility with existing ERE models.

**Applicability**: VERY HIGH. This is the most directly applicable paper found:
- Joint NER+RE (not just RE)
- Explicitly targets low-resource settings
- Model-agnostic (can wrap around SpERT)
- Curriculum ordering by overlap complexity is clever for CODE-ACCORD where
  nested/overlapping entities exist
- Requires unlabeled domain text (available: raw building regulation PDFs)
- Self-training loop can leverage our existing SpERT checkpoint as teacher

---

### 2.2 ASPER: Answer Set Programming Enhanced Neural Network Models for Joint Entity-Relation Extraction

- **Authors**: (See Cambridge Core / TPLP)
- **Year/Venue**: 2023, Theory and Practice of Logic Programming (Cambridge)
- **URL**: https://arxiv.org/abs/2305.15374

**Key technique**: Combines neural ERE models with Answer Set Programming (ASP) for
iterative self-training. Trains initial model on limited data, then uses ASP rules
encoding domain knowledge (entity type constraints, relation argument types) to
*revise* pseudo-labels before retraining. Domain knowledge acts as a noise filter
for pseudo-labels.

**Reported gains**: On SciERC, outperforms all baselines on relation (R) and joint
entity+relation (ER) F1 at both micro and macro levels.

**Applicability**: HIGH. The ASP constraint idea is powerful for CODE-ACCORD where
domain rules are well-defined (e.g., "a Quantity can only relate to a Property via
hasValue"). We could encode building-regulation ontology constraints as ASP rules
to filter pseudo-labels during self-training.

---

## 3. Few-Shot and Meta-Learning Approaches

### 3.1 TKRE: Bridging Generative and Discriminative Learning -- Few-Shot Relation Extraction via Two-Stage Knowledge-Guided Pre-training

- **Authors**: Quanjiang Guo, Jinchuan Zhang, Sijie Wang, Ling Tian, Zhao Kang, Bin Yan, Weidong Xiao
- **Year/Venue**: 2025, IJCAI 2025
- **URL**: https://arxiv.org/abs/2505.12236

**Key technique**: Two innovations:
(1) LLMs generate *explanation-driven knowledge* and *schema-constrained synthetic data*
to address data scarcity.
(2) Two-stage pre-training: Masked Span Language Modeling (MSLM) + Span-Level
Contrastive Learning (SCL). MSLM teaches span-level semantics; SCL teaches
discriminative span representations for relation types.

**Reported gains**: Outperforms all baselines including TYP Marker and GenPT across
all metrics on FewRel and other benchmarks.

**Applicability**: HIGH. The span-level pre-training (MSLM + SCL) aligns well with
our span-based architecture. Could be used as a pre-training stage before SpERT
fine-tuning on CODE-ACCORD. The synthetic data generation also addresses our data
scarcity. Code available at https://github.com/UESTC-GQJ/TKRE.

---

### 3.2 MICRE: Meta In-Context Learning Makes LLMs Better Zero and Few-Shot Relation Extractors

- **Authors**: (See IJCAI 2024)
- **Year/Venue**: 2024, IJCAI 2024
- **URL**: https://arxiv.org/abs/2404.17807

**Key technique**: Meta-trains an LLM on diverse RE datasets to improve its in-context
learning ability for RE specifically. At inference, the LLM recovers relation semantics
from a few given examples.

**Reported gains**: Better zero-shot and few-shot RE compared to vanilla ICL with
GPT-3.5/4.

**Applicability**: MEDIUM. Useful for generating silver annotations on unlabeled
CODE-ACCORD text, but requires meta-training an LLM which is expensive. More
practical as a labeling oracle for data augmentation than as the final model.

---

### 3.3 Few-Shot Biomedical NER via LLM-Assisted Data Augmentation and Multi-Scale Feature Extraction

- **Authors**: (See BioData Mining)
- **Year/Venue**: 2025, BioData Mining
- **URL**: https://pmc.ncbi.nlm.nih.gov/articles/PMC11969866/

**Key technique**: Uses ChatGPT to generate enriched data with distinct semantics for
the same entities (not just word replacement). Combines with a multi-scale feature
extraction architecture using dynamic convolution to capture local patterns at
multiple granularities.

**Reported gains**: Up to +20% F1 improvement in 5-shot setting on NCBI and BC5CDR
datasets. Strong gains across 5-shot, 20-shot, and 50-shot scenarios.

**Applicability**: MEDIUM. NER-only, biomedical domain. But the multi-scale convolution
idea could complement our encoder -- adding local feature extraction on top of BERT
embeddings before the span classifier.

---

## 4. Zero-Shot and Transfer Learning Approaches

### 4.1 GLiREL: Generalist Model for Zero-Shot Relation Extraction

- **Authors**: Jack Boylan, Chris Hokamp, Demian Gholipour Ghalandari
- **Year/Venue**: 2025, NAACL 2025
- **URL**: https://aclanthology.org/2025.naacl-long.418/

**Key technique**: Efficient architecture for zero-shot relation classification.
Classifies multiple entity pairs in a single forward pass using natural-language
relation descriptions (no fixed label set). Builds on GLiNER (zero-shot NER).
Also contributes a protocol for synthetically generating diverse relation-labeled
datasets.

**Reported gains**: SOTA on FewRel and WikiZSL zero-shot relation classification.

**Applicability**: MEDIUM-HIGH. Two use cases:
(1) Use GLiREL as a silver-label annotator on unlabeled building regulation text.
(2) Adopt the synthetic dataset generation protocol to create augmented training data.
The single-forward-pass design is efficient. Code available on GitHub.

---

### 4.2 A Few-Shot Approach for Relation Extraction Domain Adaptation using LLMs (AECO Domain)

- **Authors**: (See DL4KG @ KDD 2024)
- **Year/Venue**: 2024, DL4KG Workshop at ACM SIGKDD KDD 2024
- **URL**: https://arxiv.org/abs/2408.02377

**Key technique**: Uses ChatGPT in-context learning with schema-constrained prompts
to generate labeled RE training data for the AECO (Architecture, Engineering,
Construction, Operations) domain. Trains **SpERT with SciBERT** on the generated data.

**Reported gains**: NER performance from LLM-generated data slightly outperforms
baseline in most configurations. RE gains are more limited -- LLM few-shot learning
is harder for RE than NER. **Best results achieved by combining LLM-generated labels
with curated out-of-domain SciERC labels.**

**Applicability**: EXTREMELY HIGH. This is essentially our exact setup (SpERT +
SciBERT + construction domain). Key lessons:
- ChatGPT-generated data helps NER more than RE
- Combining LLM-generated + out-of-domain curated data (SciERC) outperforms either alone
- More few-shot examples and explicit task definitions improve generation quality
- 3373 sentences were queried, providing a template for scale

**Direct action items**:
1. Replicate their approach: prompt GPT-4o with CODE-ACCORD schema + examples,
   generate silver-labeled construction-domain text
2. Mix generated data with CODE-ACCORD gold labels for SpERT training
3. Also try adding SciERC data as supplementary out-of-domain training

---

### 4.3 LoRE: Zero-Shot Framework for Low-Resource Relation Extraction via Distant Supervision and LLMs

- **Authors**: (See MDPI Electronics)
- **Year/Venue**: 2025, Electronics 14(3)
- **URL**: https://www.mdpi.com/2079-9292/14/3/593

**Key technique**: Blends distant supervision with LLM capabilities. Uses knowledge
bases to generate distant labels, then uses LLMs to filter noise from distant
supervision, addressing data sparsity without manual annotation.

**Applicability**: MEDIUM. Requires a suitable knowledge base for the construction
domain. Could work if building-regulation ontologies (e.g., IFC, buildingSMART)
are available as distant supervision sources.

---

## 5. Hybrid and Neuro-Symbolic Approaches

### 5.1 LLM + ASP Workflow for Joint Entity-Relation Extraction

- **Authors**: (See arXiv)
- **Year/Venue**: 2025, arXiv:2508.12611
- **URL**: https://arxiv.org/abs/2508.12611

**Key technique**: Generic JERE workflow combining LLM natural language understanding
with ASP (Answer Set Programming) reasoning. LLM extracts candidate entities/relations
from raw text; ASP encodes domain-specific type constraints and rules to validate and
refine extractions. YAML config files specify domain predicates -- no core program
modification needed for new domains.

**Reported gains**: With only 10% of SciERC training data, achieves 2.5x improvement
over SOTA in RE (35% vs 15% F1). Competitive with full-data methods using fraction
of training examples.

**Applicability**: HIGH. The 10%-data result is remarkable and directly relevant.
For CODE-ACCORD, building regulation rules (type constraints, valid relation
argument types) could be encoded as ASP rules. However, this is a fundamentally
different architecture from SpERT -- would require adopting the LLM+ASP pipeline
rather than improving the span-based model.

---

### 5.2 Reamend: LLM-Augmented Joint Learning Framework for Entity-Relation Extraction

- **Authors**: Haochen Zou et al.
- **Year/Venue**: 2025, Applied Soft Computing
- **URL**: https://www.sciencedirect.com/science/article/abs/pii/S1568494625014073

**Key technique**: Uses pre-trained LLM as encoder with specialized modules:
LLM encoder -> relation identification module -> entity recognition module ->
entity-relation amendment module (filters invalid triplets) + adversarial training.

**Applicability**: MEDIUM. Different architecture from SpERT. The amendment module
(post-hoc triplet validation) is an interesting idea that could be adapted as a
post-processing step for our span-based output.

---

## 6. Span-Level Efficiency and Training Improvements

### 6.1 CEFF: Span Contribution Evaluation and Focusing Framework

- **Authors**: (See Elsevier)
- **Year/Venue**: 2024, Neurocomputing (Elsevier)
- **URL**: https://www.sciencedirect.com/science/article/abs/pii/S088523082400127X

**Key technique**: Pre-training phase evaluates contribution of each non-entity span
to model performance via sampling. Assigns contribution scores; focuses training on
high-contribution spans, discarding low-value negative spans.

**Reported gains**: SOTA on five benchmark datasets. Simplified variant CEFF-s achieves
comparable performance with fewer spans, reducing training cost.

**Applicability**: HIGH. Directly addresses a known issue with span-based models:
the overwhelming number of negative (non-entity) spans. For CODE-ACCORD with short
sentences but many span candidates, focusing on high-contribution negatives could
improve training efficiency and generalization with limited data.

---

## 7. Synthetic Data Distillation

### 7.1 Synthetic Data Distillation Enables Clinical Information Extraction at Scale

- **Authors**: (See npj Digital Medicine)
- **Year/Venue**: 2025, npj Digital Medicine
- **URL**: https://www.nature.com/articles/s41746-025-01681-4

**Key technique**: Uses a large LLM (Llama-3.1-70B-Instruct) to generate synthetic
QA training pairs, then fine-tunes a smaller model (8B). The distilled 8B model
sometimes outperforms the 70B teacher. Fine-tuning with only the *most challenging*
questions still improves performance (targeted distillation).

**Reported gains**: 8B model matches or exceeds 70B teacher across three clinical
NER/extraction tasks.

**Applicability**: MEDIUM. The paradigm (LLM teacher -> small model student) is
relevant but applied to encoder-decoder/decoder models, not span-based. The insight
about targeted distillation (hardest examples matter most) is useful for curriculum
design.

---

### 7.2 Distilled BERT Models for Clinical NER (2025)

- **Year/Venue**: 2025, arXiv:2501.00031
- **URL**: https://arxiv.org/abs/2501.00031

**Key technique**: Uses SOTA LLMs (Gemini, GPT-4o) as teacher labelers + medical
ontologies to generate silver labels, then trains distilled BERT models (~1000x
smaller than LLMs). 12x faster inference than GPT-4o, up to 101x cheaper.

**Applicability**: HIGH. This is exactly the pattern we should consider: use GPT-4o
to label unlabeled building regulation text, then train our BERT-based SpERT on
the combined gold + silver data.

---

## 8. Data-Centric Analysis

### 8.1 LLM4RE: A Data-Centric Feasibility Study for Relation Extraction

- **Authors**: Anushka Swarup, Tianyu Pan, Ronald Wilson, Avanti Bhandarkar, Damon Woodard
- **Year/Venue**: 2025, COLING 2025
- **URL**: https://aclanthology.org/2025.coling-main.447/

**Key technique**: Exhaustive analysis of 5 SOTA LLMs across 2100+ experiments for RE.
Finds LLMs are NOT robust to complex RE scenarios: contextual ambiguity, correlating
relations, long-tail data, fine-grained relation distributions all degrade performance.

**Key insight for us**: LLM-generated silver labels will be noisier for RE than for
NER. Noise filtering (via ASP constraints, consistency checks, or confidence
thresholds) is essential when using LLM augmentation for relation extraction.

---

## Summary: Ranked Recommendations for CODE-ACCORD

### Tier 1: Highest impact, lowest risk (try first)

| Priority | Method | Expected effort | Expected gain |
|----------|--------|----------------|---------------|
| 1 | **LLM data augmentation** (PGA-SciRE style): GPT-4o paraphrase + generate new sentences for CODE-ACCORD triplets | 2-3 days | +2-5% F1 |
| 2 | **CSJE curriculum + self-training**: Curriculum order by overlap complexity, then self-train on unlabeled building regulation text | 3-5 days | +1-5% F1 |
| 3 | **AECO replication**: Follow the DL4KG 2024 recipe exactly (ChatGPT schema-constrained annotation -> SpERT+SciBERT), combine with SciERC data | 2-3 days | +2-4% F1 |
| 4 | **CEFF span focusing**: Score negative spans, train only on high-contribution negatives | 2-3 days | +1-2% F1 (efficiency) |

### Tier 2: Moderate impact, moderate effort

| Priority | Method | Expected effort | Expected gain |
|----------|--------|----------------|---------------|
| 5 | **ASPER-style ASP pseudo-label filtering**: Encode CODE-ACCORD type constraints as ASP rules, use for self-training noise filter | 3-5 days | +2-4% F1 |
| 6 | **GLiREL zero-shot silver labeling**: Use GLiREL to annotate unlabeled regulatory text, add to training | 2-3 days | +1-3% F1 |
| 7 | **TKRE span-level pre-training**: MSLM + SCL pre-training on construction text before SpERT fine-tuning | 5-7 days | +2-5% F1 |
| 8 | **Distilled BERT labeling**: Use GPT-4o as teacher labeler on raw regulatory text, train SpERT on gold + silver | 3-4 days | +2-4% F1 |

### Tier 3: High potential but architectural change required

| Priority | Method | Expected effort | Expected gain |
|----------|--------|----------------|---------------|
| 9 | **LLM+ASP workflow**: Full pipeline replacement (not incremental improvement to SpERT) | 7-10 days | potentially large (2.5x on SciERC with 10% data) |
| 10 | **Full SSL framework** (TKDD 2025): LLM augmentation + consistency regularization + pseudo-labeling | 5-7 days | +5-17% (but measured on RE-only benchmarks) |

### Key Insight

The encoder ceiling (0.329 Triple F1) may be breakable by **increasing effective training data** rather than architectural changes. The most promising path combines:
1. LLM-generated augmented training examples (paraphrase + generation)
2. Out-of-domain data mixing (SciERC as supplement)
3. Self-training on unlabeled building regulation text with domain-constraint filtering
4. Curriculum ordering (simple -> complex overlap patterns)

All four can be composed incrementally on top of the existing SpERT pipeline.
