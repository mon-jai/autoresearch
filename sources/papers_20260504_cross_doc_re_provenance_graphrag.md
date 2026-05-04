# Literature lookup - cross-document RE and provenance Graph RAG

Timestamp: 2026-05-04 23:14 CST

Query:

> Find papers 2023 2024 2025 cross-document joint entity relation extraction document-level relation extraction low-resource knowledge graph construction source provenance Graph RAG

Backend: research-lookup skill, Perplexity academic search.

## Most actionable result

Jain, Monika; Mutharaju, Raghava; Singh, Kuldeep; Kavuluru, Ramakanth. 2024.
"Knowledge-Driven Cross-Document Relation Extraction." Findings of ACL 2024.
DOI: 10.18653/v1/2024.findings-acl.227.

Reported method: KXDocRE combines text paths and knowledge paths between entity
pairs, uses entity/relevance filtering to select the top paths, then applies
cross-path attention for relation classification. The lookup reported CodRED
F1 gains of roughly 3-4 points over cross-document baselines, with filtering
ablations contributing materially to performance.

Actionable fit for CODE-ACCORD:

- Treat the current two-sentence window as only a context carrier, not the full
  retrieval strategy.
- Before relation classification or Graph RAG, rank candidate evidence paths by
  entity overlap, relation-schema compatibility, and source-sentence support.
- For ACCORD, replace Wikidata paths with document-local paths: same section ID,
  repeated entity mentions, source sentence provenance, and existing KG edges.

## Secondary directions

- MR.COD (ACL Findings 2023) uses multi-hop evidence retrieval over passage
  graphs for cross-document relation extraction. This supports a next probe that
  builds passage/entity paths before asking the RE head or Graph RAG retriever
  to reason over neighborhoods.
- Provenance-aware SemanticRAG / Graph RAG work from 2025 reinforces exposing
  source traces, but the A8 diagnostic shows provenance alone is insufficient
  when extracted KG coverage is low.

## Experiment decision

Do not expand data augmentation. Best next ROI is an evidence-path candidate
ranking diagnostic: for each gold or predicted ACCORD relation query, retrieve
same-document two-hop sentence/entity paths and measure whether the missing gold
tail is present before any LLM answer step. This isolates retrieval/evidence
coverage from model generation.
