import os
import hashlib
import random
import math
from abc import ABC, abstractmethod
from typing import List

class BaseEmbeddings(ABC):
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass

class MockEmbeddings(BaseEmbeddings):
    def __init__(self, dimension: int = 1536):
        self.dimension = dimension

    def embed_query(self, text: str) -> List[float]:
        # Hash text to get a stable integer seed
        hasher = hashlib.md5(text.encode("utf-8"))
        seed = int(hasher.hexdigest(), 16) % (2**32)
        
        # Use Python's built-in thread-safe random generator instance
        rng = random.Random(seed)
        vec = [rng.gauss(0, 1) for _ in range(self.dimension)]
        
        # Calculate magnitude in pure Python
        magnitude = math.sqrt(sum(x * x for x in vec))
        if magnitude > 0:
            vec = [x / magnitude for x in vec]
            
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(t) for t in texts]

class OpenAIEmbeddings(BaseEmbeddings):
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name
        # pyrefly: ignore [missing-import]
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        self.client = OpenAI(api_key=api_key)

    def embed_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            input=[text],
            model=self.model_name
        )
        return response.data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            input=texts,
            model=self.model_name
        )
        return [item.embedding for item in response.data]

def get_embedding_model(model_name: str) -> BaseEmbeddings:
    # If key is missing or model is mock, use MockEmbeddings
    if model_name == "mock" or not os.getenv("OPENAI_API_KEY"):
        print("[Embeddings] Using MockEmbeddings (offline mode or key missing)")
        return MockEmbeddings()
    else:
        print(f"[Embeddings] Using OpenAIEmbeddings with model: {model_name}")
        return OpenAIEmbeddings(model_name=model_name)
