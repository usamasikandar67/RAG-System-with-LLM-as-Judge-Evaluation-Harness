# Runbook - Stage 04: Retrieval Strategies

## Local Development Setup
- Install dependencies: `pip install rank-bm25`.

## Query Execution CLI
Verify retrieval outputs directly from CLI:
```bash
python -m src.stage_04_retrieval_strategies.query --strategy hybrid --q "How do I configure MLflow experiment tracking?"
```

## Running Tests
Run unit tests verifying the correctness of RRF scoring, multi-query expansion mocks, and retrieval output schemas:
```bash
pytest tests/test_stage_04.py
```

## CI/CD Workflow
- Automatically checked on every push/PR.
- If changes are made to retrieval strategy files, `eval.yml` triggers the evaluation gate.

## Configuration and Environment
- Environment Variables:
  - `RETRIEVER_STRATEGY`: `dense`, `multi_query`, or `hybrid`.
  - `OPENAI_API_KEY`: Required for multi-query LLM expansion calls.

## Troubleshooting and Rollback
- Slow retrieval times: Verify BM25 index size or database connection times. Pre-tokenize text datasets.
- Query expansion failures: Ensure prompt handling catches openapi json exceptions or timeouts, falling back to a raw kNN retrieval query.
- Reset BM25: BM25 caches may be saved in `./data/processed/bm25.pkl`; delete this file to re-initialize BM25 statistics.
