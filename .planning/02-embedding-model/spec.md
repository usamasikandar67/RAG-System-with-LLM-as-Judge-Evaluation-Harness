# Specification - Stage 02: Embedding Model

## Goal / Responsibility
The goal of this stage is to establish a pluggable, unified interface for generating vector embeddings from text chunks. It handles interactions with API-based embedding providers (defaulting to OpenAI's `text-embedding-3-large`) and supports local/open-source alternative models (such as BGE, E5, or Instructor models) with strict formatting validation.

## Inputs and Outputs

### Inputs
- **Text Chunk or Batch of Text Chunks**: A single string or a list of strings to embed.
- **Model Specification**: Parameters detailing the target model name and dimensions.

### Outputs
- **Embedding Vectors**: List of lists of floats (`List[List[float]]`), representing the high-dimensional vector space.
- For `text-embedding-3-large`, default dimension is `3072` (or explicitly truncated to `1536` if specified).

## Interfaces/APIs
This stage exposes a python interface:
```python
from abc import ABC, abstractmethod
from typing import List

class BaseEmbeddingModel(ABC):
    @abstractmethod
    def embed_queries(self, queries: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass
```

Implementations:
- `OpenAIEmbeddingModel` (using OpenAI `openai` SDK client)
- `HuggingFaceEmbeddingModel` (using `sentence-transformers` for local model executions)

## Tech Choices and Why
- **OpenAI Client SDK**: High-quality, fast inference using `text-embedding-3-large`.
- **HuggingFace SentenceTransformers**: Easy pluggability to switch to open-source BGE (`BAAI/bge-large-en-v1.5`) or Multilingual-E5 (`intfloat/multilingual-e5-large`) running locally on CPU or GPU.
- **Alternatives**: Cohere Embed API, AWS Bedrock Titan Embeddings (swappable behind `BaseEmbeddingModel`).

## Config/Env Vars
- `EMBEDDING_PROVIDER` (e.g. `openai`, `huggingface`)
- `EMBEDDING_MODEL_NAME` (Default: `text-embedding-3-large` or `BAAI/bge-large-en-v1.5`)
- `EMBEDDING_DIMENSION` (Default: `3072`)
- `OPENAI_API_KEY` (Required for OpenAI provider)

## Explicit Non-Goals
- Vector database indexing or storage (handled in Stage 03).
- Token checking for prompt lengths (responsibility of the caller, although tokenizer validation helpers are allowed).
- Embedding caching/memoization layers (out of scope for initial prototype).

## Acceptance Criteria
- Code implements the `BaseEmbeddingModel` interface correctly.
- Invoking `embed_documents` returns correct vector dimensions matching configuration (e.g., 3072 dimensions).
- Support batching to avoid API rate limits or out-of-memory errors for large local models.
- Unit tests run with mocks for the OpenAI API to prevent test suite API charges.
- Local execution fallback runs correctly without GPUs when running tests.
