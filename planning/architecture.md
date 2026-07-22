# Architecture - Cancer Clinical AI Evaluation Platform

## Decoupled System Layers
The architecture is structured according to the Repository Pattern and SOLID principles to isolate clinical data operations from RAG pipeline runs and evaluation grading.

```mermaid
graph TD
    subgraph Data Layer
        A[CancerQA.csv] --> B[datasets/manager.py]
        B --> C[(knowledge_base text files)]
        B --> D[golden_dataset.json]
    end

    subgraph Retrieval Layer
        C --> E[Recursive Splitter]
        E --> F[SciSpacy NER & ICD Map]
        F --> G[(SQLite Metadata Index)]
        E --> H[BM25 Index]
        E --> I[Dense Vector Store]
        H & I --> J[Hybrid Search / RRF]
    end

    subgraph Execution & Evaluation Layer
        J --> K[RAGPipeline app/pipeline.py]
        D --> L[EvaluationEngine]
        K --> M[Mock/OpenAI Judge]
        M --> L
        L --> N[(SQLite Runs Database)]
    end

    subgraph Observability & Presentation
        N --> O[Streamlit Dashboard]
        K --> P[Langfuse Tracing Span]
    end
```

## SOLID Principles Design
- **Single Responsibility (SRP)**: Ingestion, NER, Embeddings generation, Retrieval, and Judge scoring are separated into distinct modules.
- **Open/Closed (OCP)**: Retrievers and Judges inherit from abstract base classes (`BaseRetriever`, `BaseJudge`), allowing addition of Pinecone or Claude-Judge without altering the pipeline runner.
- **Liskov Substitution (LSP)**: Mock generators and OpenAI pipelines share identical typing interfaces.
- **Interface Segregation (ISP)**: Clients import minimal APIs (e.g. `ingest_documents` or `evaluate_regression`).
- **Dependency Inversion (DIP)**: `RAGPipeline` depends on `BaseRetriever` interface rather than concrete database clients.
