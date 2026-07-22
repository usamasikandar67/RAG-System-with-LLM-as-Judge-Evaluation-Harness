# Roadmap - Cancer Clinical AI Evaluation Platform

## Chronological Release Plan

```mermaid
gantt
    title Clinical RAG Platform Release Roadmap
    dateFormat  YYYY-MM-DD
    section Implementation
    Phase 1: Basic Ingestion & Mock Pipeline :active, p1, 2026-07-16, 10d
    Phase 2: Retrieval Optimizations (Hybrid/Rerank) : p2, after p1, 14d
    Phase 3: Clinical NER Integration (SciSpacy) : p3, after p2, 14d
    Phase 4: ICD-10 Coding Mapping : p4, after p3, 7d
    Phase 5: Observability Traces (Langfuse) : p5, after p4, 10d
    Phase 6: Multi-Page dashboard & CI Gate : p6, after p5, 14d
    Phase 7: Enterprise Production Deployment : p7, after p6, 14d
```

### Phase Details

#### Phase 1 → Small Prototype
- Establish `datasets/manager.py` parser, recursive character splitter, mock embeddings (seeded hash), simple mock retrieval, and local SQLite log DB.

#### Phase 2 → Improve Retrieval
- Introduce hybrid BM25 + dense retrieval, custom overlap settings testing, and integration of cross-encoder rerankers.

#### Phase 3 → Add Medical NER
- Integrate SciSpaCy model extracts: cancer types, biomarkers, medication, chemotherapy drugs, and dosages. Structure as metadata.

#### Phase 4 → ICD Mapping
- Perform mapping of cancer entities to ICD-10 classification codes (e.g., Breast Cancer to C50).

#### Phase 5 → Langfuse Observability
- Configure spans and tracing tracking prompt tokens, completions, embedding latencies, and costs.

#### Phase 6 → Evaluation Platform
- Assemble the 11-page Streamlit analytical dashboard and CI/CD gate triggers.

#### Phase 7 → Production Architecture
- Migrate local storage to Qdrant or Pinecone vector DB and connect trusted NCI and WHO oncology guideline corpora.
