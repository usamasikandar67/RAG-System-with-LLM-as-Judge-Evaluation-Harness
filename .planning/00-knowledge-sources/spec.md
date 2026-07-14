# Specification - Stage 00: Knowledge Sources

## Goal / Responsibility
The goal of this stage is to define the ingestion pathways and local layout for the raw data sources that populate our knowledge base. It is responsible for fetching or referencing documentation (Databricks Docs, Spark Docs, Delta Lake Docs, dbt Docs, Kafka Docs, Snowflake Docs, MLflow Docs) and internal PDFs, and storing them in an structured directory raw-format hierarchy for downstream processing.

## Inputs and Outputs

### Inputs
- **Public Documentation Sites**: URLs/sitemaps for the targeted doc portals.
- **Internal PDFs**: Local files or file paths containing internal documentation to be indexed.

### Outputs
- **Raw Storage Directory**: A local directory structure (e.g., `data/raw/`) divided by source category:
  - `data/raw/databricks_docs/` (HTML/Markdown)
  - `data/raw/spark_docs/` (HTML/Markdown)
  - `data/raw/delta_lake_docs/` (HTML/Markdown)
  - `data/raw/dbt_docs/` (HTML/Markdown)
  - `data/raw/kafka_docs/` (HTML/Markdown)
  - `data/raw/snowflake_docs/` (HTML/Markdown)
  - `data/raw/mlflow_docs/` (HTML/Markdown)
  - `data/raw/internal_pdfs/` (PDF files)
- **Manifest File**: `data/raw/manifest.json` containing metadata about downloaded sources (URLs, fetch timestamps, file hashes, source types).

## Interfaces/APIs
This stage does not expose a running API, but rather a directory layout and a CLI ingestion runner:
- `python -m src.stage_00_knowledge_sources.ingest --source [source_name|all]`
- Outputs static files consumed by downstream Stages (Stage 01: Document Processing).

## Tech Choices and Why
- **Python (httpx + BeautifulSoup4 + Playwright)**: To scrape or download public sitemaps and content. Playwright will serve as a swappable alternative if Javascript rendering is required.
- **Local Directory Structure**: Simple filesystem storage for low complexity, easy inspection, and reproducibility.
- **Alternatives**: AWS S3 or MinIO as storage targets (swappable if scaling out of a single local workspace is required).

## Config/Env Vars
- `INGESTION_DATA_DIR` (Default: `./data`)
- `PLAYWRIGHT_HEADLESS` (Default: `True`)

## Explicit Non-Goals
- Parsing PDF contents or HTML contents into text or chunks (handled in Stage 01).
- Incremental change tracking or database-level syncing.
- Cleaning of raw markdown/html noise.

## Acceptance Criteria
- Running `python -m src.stage_00_knowledge_sources.ingest --source test` downloads a sample document from each domain.
- Ingested files are stored in distinct directories with a valid `manifest.json`.
- Manifest file satisfies JSON schema verification (must contain source path, type, timestamp, hash).
- CLI fails gracefully (non-zero status code) if a source is unreachable, with detailed logging.
