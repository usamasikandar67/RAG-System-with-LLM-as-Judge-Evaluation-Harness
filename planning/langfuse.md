# Langfuse Integration - Cancer Clinical AI Evaluation Platform

## Trace Metadata Mapping
Observability traces require mapping configuration parameters alongside queries:

```json
{
  "trace": {
    "name": "cancer_clinical_query",
    "user_id": "auditor_doc_01",
    "tags": ["experiment_hybrid_v1", "gpt-4o-mini"],
    "metadata": {
      "experiment_id": "exp_1784153929255",
      "retriever_type": "hybrid",
      "chunk_size": 800,
      "chunk_overlap": 150,
      "prompt_version": "v1.2",
      "embedding_model": "text-embedding-3-small"
    }
  }
}
```

## Trace Call Tree (Spans)
- **Trace ID**: `trace_root_abc123`
  - **Span 1: Extract Clinical Entities**: SciSpaCy NER extraction execution.
  - **Span 2: Retrieve Contexts**: Vector lookup + BM25 keyword matching latency.
  - **Span 3: Compile Prompt**: User instructions rendering.
  - **Span 4: Generation Completion**: LLM query execution tracking token usage and cost.
  - **Span 5: Judge Evaluation**: LLM-as-Judge validation grading.

## Cost and Token Tracking
Calculates costs based on actual API billing rates per million tokens. Logs input, output, and total token parameters.
