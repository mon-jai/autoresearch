
================================================================================
Query 1: Find papers on GraphRAG knowledge graph retrieval entity linking subgraph retrieval low-resource information extraction 2024 2025
Timestamp: 2026-05-02 06:19:45
Backend: perplexity | Model: perplexity/sonar-pro-search
================================================================================
GraphRAG enhances retrieval-augmented generation (RAG) by leveraging knowledge graphs (KGs) for structured retrieval, addressing limitations in traditional vector-based RAG for complex, multi-hop queries. Recent 2024-2025 papers focus on entity linking to map queries to KG nodes, subgraph retrieval for relevant evidence extraction, and adaptations for low-resource settings via lightweight models or dynamic construction. These advancements improve accuracy, reduce hallucinations, and scale to large corpora, with key works from Microsoft Research and arXiv preprints leading the field.[2][1]

## Foundational GraphRAG Framework

The seminal GraphRAG paper introduces a scalable method for query-focused summarization over private corpora by constructing entity KGs and community summaries. Authors Darren Edge et al. from Microsoft Research (established in AI and graph systems) propose a two-stage indexing: entity extraction into a KG followed by hierarchical community detection and LLM-generated summaries for global questions like "main themes in the dataset." This outperforms baseline RAG by improving comprehensiveness (72-83%) and diversity (62-82%) on million-token datasets, using local/global search modes; limitations include high indexing costs, mitigated by precomputation.[2]

No citation counts available yet due to recency, but from Microsoft Research (Tier 2 venue equivalent via arXiv/NeurIPS impact); DOI: 10.48550/arXiv.2404.16130.[2]

## Subgraph Retrieval Innovations

SubgraphRAG by Mufei Li et al. (2024, updated 2025) optimizes KG-based RAG via lightweight MLPs for parallel triple-scoring with directional-distance encoding (DDE), enabling flexible subgraph extraction tailored to query needs and LLM capacity. It balances retrieval efficiency and effectiveness on benchmarks like WebQSP and CWQ, where Llama3.1-8B matches larger models' accuracy with explainable reasoning, reducing hallucinations; key implication is scalability without fine-tuning.[1]

GRAG (Hu et al., 2024) employs divide-and-conquer for linear-time textual subgraph retrieval, integrating topology into LLMs via soft pruning of irrelevant entities. This enhances multi-hop reasoning in low-resource extraction by filtering noisy graphs.[8]

Both are arXiv preprints (Tier 3), recent (cited in 2025 surveys, ~10-50 citations estimated); DOIs: 10.48550/arXiv.2410.20724 (SubgraphRAG), GRAG referenced in GraphRAG repos.[1]

## Entity Linking in GraphRAG

Entity linking connects query mentions to KG nodes, crucial for accurate retrieval. A SemTab 2025 challenge paper adapts GRASP (zero-shot SPARQL translation) for interactive KG exploration via functions like SEN (search entities) and LST (list triples), ranking first in cell entity annotation by disambiguating candidates through triple inspection. This low-resource approach suits sparse KGs, with implications for biomedical/low-data domains; limitation: LLM dependency for function calls.[6]

KGMEL (2025) enhances multimodal entity linking with KGs, supporting GraphRAG in vision-text settings.

Tier 3 venues (challenge/workshop papers), authors from Univ. Freiburg (reputable in KGQA); no DOIs in snippets.[6]

## Low-Resource Information Extraction

Low-resource IE in GraphRAG emphasizes dynamic/zero-shot methods. GraphFlow (NeurIPS 2025 poster) retrieves diverse KG subgraphs without expensive supervision, outperforming GPT-4o by 10% hit rate/recall on STaRK benchmark across domains. It aligns retrieval to query needs via flow-based methods, ideal for low-data KGs; generalization to unseen graphs noted as strength.[1]

DrKGC (2025) uses lightweight embeddings and rule-guided bottom-up subgraph retrieval for KG completion in sparse settings, refining via GCN adapters before LLM prediction.

Relink (2026 arXiv) dynamically repairs sparse KGs by re-linking latent relations from text, outperforming static GraphRAG under sparsity.[10]

These are emerging (Tier 2-3: NeurIPS poster, arXiv), prioritizing efficiency; e.g., GraphFlow from NeurIPS (high-impact conference).[1]

## Key Comparisons

| Paper | Venue/Year | Core Innovation | Strengths | Limitations | Benchmark Gains |
|-------|------------|-----------------|-----------|-------------|-----------------|
| GraphRAG (Edge et al.) | arXiv 2024 | Entity KG + community summaries | Global QA, diversity | Indexing cost | +72-83% comprehensiveness [2] |
| SubgraphRAG (Li et al.) | arXiv 2024/25 | MLP triple-scoring + DDE | Efficiency, flexible size | Relies on KG quality | SOTA on WebQSP/CWQ [1] |
| GraphFlow | NeurIPS 2025 | Supervision-free flow retrieval | Low-resource, generalization | Domain-specific eval | +10% vs GPT-4o [1] |
| GRASP Entity Linking | SemTab 2025 | Interactive SPARQL functions | Zero-shot, disambiguation | LLM function calls | 1st in challenge [6] |

Influential due to Microsoft lead and benchmark leadership; no Tier-1 (Nature/Science) yet, as field is nascent.

Future directions include hybrid vector-graph systems for broader adoption, real-time dynamic KG updates for evolving data, and low-resource IE via self-supervised rules to close gaps in sparse domains like biomedicine. Conflicts arise in retrieval granularity (subgraph size vs. noise), with ongoing benchmarks favoring adaptive methods; no major controversies, but scalability debates persist.[7]

Additional References (1):
  [1] DOI: 10.48550/arXiv.2404.16130 - https://doi.org/10.48550/arXiv.2404.16130

Usage: {'prompt_tokens': 548, 'completion_tokens': 1240, 'total_tokens': 1788, 'cost': 0.03024, 'is_byok': False, 'prompt_tokens_details': {'cached_tokens': 0, 'cache_write_tokens': 0, 'audio_tokens': 0, 'video_tokens': 0}, 'cost_details': {'upstream_inference_cost': 0.03024, 'upstream_inference_prompt_cost': 0.001644, 'upstream_inference_completions_cost': 0.028596}, 'completion_tokens_details': {'reasoning_tokens': 0, 'image_tokens': 0, 'audio_tokens': 0}}
