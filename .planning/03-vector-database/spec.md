# Specification - Stage 03: Vector Database

## Goal / Responsibility
The goal of this stage is to design and implement a swappable Vector Database abstraction layer. It is responsible for indexing chunk vectors generated in Stage 02 and performing fast approximate nearest neighbor (ANN) searches. ChromaDB will be used as the default local, file-backed database, with interfaces kept fully swappable for cloud databases (such as Pinecone or Weaviate) or simple memory-based indexes (FAISS).

## Inputs and Outputs

### Inputs
- **Ingestion / Indexing**: Text chunks (`chunks.jsonl`) accompanied by their pre-computed dense embedding vectors (`List[List[float]]`).
- **Query Search**: A single dense query vector (`List[float]`) and a target integer `k` representing the number of neighbors to retrieve.

### Outputs
- **Search Results**: List of matching chunks with similarity scores:
  ```python
  List[Dict[str, Any]]  # containing chunk_id, content, metadata, and score
  ```

## Interfaces/APIs
This stage exposes a common python interface:
```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseVectorDatabase(ABC):
    @abstractmethod
    def index_documents(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        pass

    @abstractmethod
    def query_similarity(self, query_vector: List[float], k: int) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass
```

Implementations:
- `ChromaVectorDB` (using chromadb client SDK)
- `FAISSVectorDB` (using faiss-cpu library)
- `PineconeVectorDB` (placeholder implementation for scale testing)

## Tech Choices and Why
- **ChromaDB**: Native Python vector store, requires zero external server setup, runs file-backed, easy to spin up in tests.
- **FAISS (cpu)**: Extremely fast local CPU-based indexing library, optimal for high-speed offline pipeline evaluations.
- **Alternatives**: Pinecone, Weaviate, Milvus (good for production cluster deployments, but introduce heavy operational overhead for local testing).

## Config/Env Vars
- `VECTOR_DB_TYPE` (e.g. `chroma`, `faiss`, `pinecone`)
- `VECTOR_DB_PERSIST_PATH` (Default: `./data/vector_store/`)
- `PINECONE_API_KEY` (Optional)
- `PINECONE_ENVIRONMENT` (Optional)

## Explicit Non-Goals
- Text pre-processing, chunking, or token length validations (Stage 01).
- Direct query expansion or hybrid fusion calculations (Stage 04).
- Hosting database nodes as part of local orchestration (databases run in-process or serverless).

## Acceptance Criteria
- Code implements the `BaseVectorDatabase` interface correctly.
- Indexing 1,000 documents with Chroma or FAISS finishes within 10 seconds.
- Similarity queries return exactly `k` nearest neighbors with correct metadata structures.
- Vector database operations are completely isolated and swappable through environment configuration.
- Cleanup utility exists to clear indices during integration and unit test setups.
