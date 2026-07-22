from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseRetriever(ABC):
    @abstractmethod
    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Indexes a list of documents/chunks.
        Each document is a dictionary containing:
        - "id": unique string ID
        - "text": string text content
        - "metadata": dict containing source file, topic, etc.
        """
        pass

    @abstractmethod
    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves top k documents matching the query.
        Returns a list of documents, each enriched with a "score" key.
        """
        pass
