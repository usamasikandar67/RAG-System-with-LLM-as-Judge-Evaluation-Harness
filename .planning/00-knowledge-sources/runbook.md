# Runbook - Stage 00: Knowledge Sources

## Local Development Setup
- Ensure dependencies are installed (e.g., `pip install httpx beautifulsoup4 playwright`).
- Run `playwright install chromium` to support dynamic JS pages if required.

## Running Ingestion
To run the ingestion command locally:
```bash
python -m src.stage_00_knowledge_sources.ingest --source all
```

## Running Tests
Run tests for stage 00 using pytest:
```bash
pytest tests/test_stage_00.py
```

## CI/CD Workflow
- Linting and unit tests will run on every push/PR.
- Scraper validation tests run under a mock web environment to avoid live network requests.

## Configuration and Environment
Secrets/credentials required:
- None for public documentation.
- Custom token access if internal resources require auth (e.g., `INTERNAL_DOCS_TOKEN`).

## Troubleshooting and Rollback
- Check internet connection if requests fail.
- Inspect `data/raw/manifest.json` for structure mismatches.
- Delete individual raw directories (e.g., `data/raw/databricks_docs/`) and re-run to reset a specific source.
