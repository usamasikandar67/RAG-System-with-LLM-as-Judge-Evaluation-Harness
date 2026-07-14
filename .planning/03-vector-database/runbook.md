# Runbook - Stage 03: Vector Database

## Local Development Setup
- Install dependencies: `pip install chromadb faiss-cpu`.

## Running Indexing Pipeline
Build indices from chunks:
```bash
python -m src.stage_03_vector_database.index
```

## Running Tests
Run unit tests checking vector database initialization, document indexing, and retrieval matches:
```bash
pytest tests/test_stage_03.py
```

## CI/CD Workflow
- Automatically checked by GitHub Actions.
- FAISS/Chroma tests are executed in-memory to prevent local disk writes and test isolation issues.

## Configuration and Environment
- Environment Variables:
  - `VECTOR_DB_TYPE`: `chroma` or `faiss`.
  - `VECTOR_DB_PERSIST_PATH`: Path to store SQLite/Index binaries locally.

## Troubleshooting and Rollback
- Database locks: If Chroma raises a database lock warning (usually SQLite related), terminate any orphaned python processes.
- Reset index: Run cleanup CLI script or delete persistence folder:
```bash
rm -rf ./data/vector_store/
```
