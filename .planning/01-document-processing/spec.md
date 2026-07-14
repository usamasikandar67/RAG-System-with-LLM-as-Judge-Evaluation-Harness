# Specification - Stage 01: Document Processing

## Goal / Responsibility
The goal of this stage is to build the document processing pipeline. It is responsible for parsing raw ingestion files (HTML, markdown, and PDF) from the raw storage directory, cleaning them (removing boilerplate, HTML tags, or formatting noise), chunking them into semantic segments, extracting metadata (source URL, title, document type, chunk index, page number), and outputting a structured chunk manifest.

## Inputs and Outputs

### Inputs
- **Raw Files**: Located in `data/raw/*`
- **Ingestion Manifest**: `data/raw/manifest.json`

### Outputs
- **Chunks File**: `data/processed/chunks.jsonl` where each line is a JSON object with:
  - `chunk_id` (string, unique hash of content)
  - `document_id` (string, hash of source document)
  - `content` (string, clean chunk text)
  - `metadata`:
    - `source` (string, original URL or filename)
    - `doc_type` (string, e.g., "pdf", "databricks_docs", "dbt_docs")
    - `title` (string, title of the page/doc)
    - `page_number` (int, if PDF, null otherwise)
    - `chunk_index` (int, 0-indexed position within doc)

## Interfaces/APIs
- CLI command to trigger processing:
  - `python -m src.stage_01_document_processing.process`
- Python class interface for downstream stages:
  - `src.stage_01_document_processing.loader.ChunkLoader` for reading `chunks.jsonl`.

## Tech Choices and Why
- **PyPDF/pdfplumber**: Python libraries for reliable local text extraction from PDFs.
- **Markdown / BeautifulSoup**: To strip HTML tags and scrape headers.
- **LangChain RecursiveCharacterTextSplitter (or custom sliding window tokenizer implementation)**: For chunking content dynamically based on characters (e.g. 500-1000 characters with 10% overlap) ensuring sentence/paragraph boundaries are preserved where possible.
- **Alternatives**: Unstructured.io (heavy, complex docker dependency, avoided for simpler local parsing).

## Config/Env Vars
- `CHUNK_SIZE` (Default: `1000` characters)
- `CHUNK_OVERLAP` (Default: `100` characters)

## Explicit Non-Goals
- Generating embeddings for chunks (Stage 02).
- Storing chunks in the vector database (Stage 03).
- Fine-tuning chunk size dynamically based on evaluation metrics in this stage.

## Acceptance Criteria
- Running the processing CLI script reads from `data/raw/` and outputs a valid `chunks.jsonl`.
- Zero raw HTML tags or PDF control characters in the final output `content` fields.
- Each chunk ID matches the SHA256 of its contents, avoiding duplicates.
- Processing handles corrupt/empty files gracefully by logging warning traces instead of crashing.
- Unit tests verify splitter bounds and overlap guarantees on sample mock text.
