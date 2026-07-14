# Runbook - Stage 01: Document Processing

## Local Development Setup
- Ensure dependencies are installed (e.g., `pip install pypdf pdfplumber langchain-text-splitters`).

## Running Document Processing
To process ingested files:
```bash
python -m src.stage_01_document_processing.process
```

## Running Tests
Run parser and text splitter unit tests:
```bash
pytest tests/test_stage_01.py
```

## CI/CD Workflow
- Checked automatically by GitHub Actions CI pipeline.
- Uses mock files under `tests/fixtures/` to test parser outputs.

## Configuration and Environment
Secrets/credentials required:
- None.

## Troubleshooting and Rollback
- If PDF extraction fails, verify if PDF is scanned (requires OCR like Tesseract) or password-protected.
- If chunks overlap does not match specs, review recursive character splitter settings.
- To rollback, re-run processing on a previous manifest or clean output:
```bash
rm -rf data/processed/chunks.jsonl
python -m src.stage_01_document_processing.process
```
