# KXDocRE: Knowledge-Driven Cross-Document Relation Extraction

Paper: Monika Jain, Raghava Mutharaju, Kuldeep Singh, Ramakanth Kavuluru.
"Knowledge-Driven Cross-Document Relation Extraction." Findings of ACL 2024.
DOI: 10.18653/v1/2024.findings-acl.227.

## Why it matters

The Phase A8 two-sentence window preserves nearby context, but the KG/RAG
diagnostic still shows low KG-only accuracy. This paper suggests the missing
piece is not merely longer context; it is evidence path selection between
entities before relation prediction or retrieval.

## Method summary

KXDocRE builds candidate evidence paths between entity pairs using text paths
and external knowledge paths. It filters paths for entity relevance, encodes
selected paths, and applies cross-path attention for relation classification.
The lookup reported roughly 3-4 F1 improvement on CodRED-style cross-document
relation extraction, with path filtering contributing strongly in ablations.

## CODE-ACCORD translation

External Wikidata paths are unlikely to cover building-regulation entities.
Use document-local substitutes:

- same `doc_id` or section metadata,
- repeated normalized entity mentions,
- source sentences already preserved on KG edges,
- schema-compatible relation labels,
- one- or two-hop paths through extracted ACCORD triples.

## Next experiment idea

Run an evidence-path coverage diagnostic before any new model training:

1. Build a graph over test sentences and entity mentions using same-document
   adjacency and extracted KG edges.
2. For each gold relation question, retrieve top-k document-local paths from
   head to candidate tails.
3. Measure whether the gold tail and relation-compatible evidence sentence are
   present in top-k.

If top-k evidence recall is low, the bottleneck is extraction coverage. If it is
high but Graph RAG still fails, the bottleneck is ranking/prompt use of evidence.
