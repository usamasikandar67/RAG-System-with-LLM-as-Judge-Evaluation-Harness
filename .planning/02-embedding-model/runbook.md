# Runbook - Stage 02: Embedding Model

## Local Development Setup
- Install dependencies: `pip install openai sentence-transformers`.
- Export API key: `export OPENAI_API_KEY="your-key-here"`.

## Testing Embeddings
Validate the embedding generation outputs:
```bash
python -c "from src.stage_02_embedding_model.openai_embedding import OpenAIEmbeddingModel; print(len(OpenAIEmbeddingModel().embed_queries(['test'])[0]))"
```

## Running Tests
Run unit tests with mocks:
```bash
pytest tests/test_stage_02.py
```

## CI/CD Workflow
- Unit tests automatically execute with pytest.
- Mocks simulate API calls so secrets aren't required to pass standard pipeline checks (except integration suites, if any).

## Configuration and Environment
- Environment Variables:
  - `OPENAI_API_KEY`: OpenAI API Key.
  - `EMBEDDING_PROVIDER`: Choose `openai` or `huggingface`.

## Troubleshooting and Rollback
- Rate limit errors: Increase batch delay or apply retry backoff in the client wrapper.
- Dimensions mismatches: Ensure `EMBEDDING_DIMENSION` environment variable matches the configured model capability (e.g. 3072 for text-embedding-3-large).
