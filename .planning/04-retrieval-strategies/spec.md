# Specification - Stage 04: Retrieval Strategies

## Goal / Responsibility
The goal of this stage is to construct three distinct, independently testable retrieval strategies behind a unified retriever interface. This allows downstream generation and evaluation harnesses to interchange the retrieval system without changing generation logic.
The strategies are:
1. **Vector Search (V1)**: Dense kNN retrieval utilizing similarity search in the vector DB.
2. **Multi-Query RAG (V2)**: Query expansion utilizing an LLM to formulate multiple search queries, executing dense searches for each, and combining matching documents.
3. **Hybrid BM25 + Dense (V3)**: Combining BM25 keyword matching (sparse) with dense vector search (dense) using Reciprocal Rank Fusion (RRF).

## Inputs and Outputs

### Inputs
- **User Query**: String query.
- **Top K**: Integer specifying maximum number of documents to return.

### Outputs
- **Retrieved Contexts**: List of matching chunks (with scores and contents) that downstream generation uses.
  ```python
  List[Dict[str, Any]]
  ```

## Interfaces/APIs
All strategies inherit from a common `BaseRetriever` interface:
```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int) -> List[Dict[str, Any]]:
        pass
```

Implementations:
- `DenseRetriever`
- `MultiQueryRetriever`
- `HybridRetriever`

## Tech Choices and Why
- **Rank-BM25**: Pure-python BM25 implementation for sparse token retrieval, requiring no extra infrastructure.
- **Reciprocal Rank Fusion (RRF)**: A parameter-free rank fusion strategy that scores documents based on their rank in sparse and dense search results:
  $$RRF\_Score(d) = \sum_{m \in M} \frac{1}{k_{rrf} + r_m(d)}$$ (typically $k_{rrf} = 60$).
- **LLM Query Expansion**: Simple call to GPT-3.5-turbo or GPT-4o to generate 3 alternative queries.
- **Alternatives**: Cohere Rerank API (swappable, but omitted from V1/V2/V3 to limit network latency and costs).

## Config/Env Vars
- `RETRIEVER_STRATEGY` (e.g. `dense`, `multi_query`, `hybrid`)
- `RRF_K_CONSTANT` (Default: `60`)
- `MULTI_QUERY_COUNT` (Default: `3`)

## Explicit Non-Goals
- Prompt generation for downstream LLM answering (Stage 05).
- Evaluation of retrieval precision (Stage 07).
- Generating embedding databases or chunking documents (Stages 01, 02, 03).

## Acceptance Criteria
- Unified interface exposes a `.retrieve()` method returning list of dictionaries with matching formats.
- `MultiQueryRetriever` handles API failures gracefully by falling back to the original single query.
- `HybridRetriever` implements correct RRF ranking algorithm (verified by unit tests with explicit rank expectations).
- Retrievers support execution against empty databases or out-of-vocabulary terms without crashing.
- Unit tests cover each of the three retriever strategies individually under isolated mock databases.
