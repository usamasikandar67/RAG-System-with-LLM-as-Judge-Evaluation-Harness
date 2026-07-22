# Database Schema - Cancer Clinical AI Evaluation Platform

## Relational SQLite Database Design

```mermaid
erDiagram
    experiments ||--o{ eval_runs : logs
    eval_runs ||--o{ traces : tracks
    eval_runs ||--o{ latency_metrics : records
    eval_runs ||--o{ token_analytics : details

    experiments {
        TEXT experiment_id PK
        TEXT name
        TEXT created_at
        TEXT rag_settings
    }

    eval_runs {
        TEXT run_id PK
        TEXT experiment_id FK
        TEXT test_id
        TEXT question
        TEXT response_text
        REAL latency
        REAL cost
        REAL hit_1
        REAL hit_3
        REAL hit_5
        REAL reciprocal_rank
        REAL average_precision
        REAL correctness
        REAL completeness
        REAL faithfulness
        REAL citation_accuracy
        REAL clinical_utility
        TEXT timestamp
    }

    traces {
        TEXT trace_id PK
        TEXT run_id FK
        TEXT question
        TEXT retrieved_documents
        TEXT prompt
        TEXT llm_response
        TEXT judge_response
    }

    latency_metrics {
        TEXT run_id PK
        REAL embedding_time
        REAL retrieval_time
        REAL reranking_time
        REAL generation_time
        REAL judge_time
        REAL total_time
    }

    token_analytics {
        TEXT run_id PK
        INTEGER input_tokens
        INTEGER output_tokens
        INTEGER total_tokens
        TEXT cancer_type
        TEXT model_name
    }
```

- **RAG Settings JSON**: Stores chunk size, overlap, embedding model details, and retriever configurations.
- **Traces Payload**: Connects logs to debugging interfaces.
