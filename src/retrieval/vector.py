import math
from typing import List, Dict, Any
from retrieval.base import BaseRetriever
from embeddings.factory import BaseEmbeddings

class SimpleVectorRetriever(BaseRetriever):
    """
    MVP (Minimum Viable Product) in-memory vector retriever.
    This implementation uses a simple Python list and basic math for cosine similarity.
    For production workloads, use ChromaRetriever instead.
    """
    def __init__(self, embedding_model: BaseEmbeddings):
        self.embedding_model = embedding_model
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: List[List[float]] = []

    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        self.documents = documents
        if not documents:
            self.embeddings = []
            return
        
        texts = [doc["text"] for doc in documents]
        self.embeddings = self.embedding_model.embed_documents(texts)

    def _cosine_similarity(self, u: List[float], v: List[float]) -> float:
        dot_product = sum(x * y for x, y in zip(u, v))
        magnitude_u = math.sqrt(sum(x * x for x in u))
        magnitude_v = math.sqrt(sum(y * y for y in v))
        if magnitude_u == 0 or magnitude_v == 0:
            return 0.0
        return dot_product / (magnitude_u * magnitude_v)

    def retrieve(self, query: str, k: int = 3, filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        if not self.documents or not self.embeddings:
            return []

        query_vec = self.embedding_model.embed_query(query)
        
        # Apply metadata pre-filter
        candidate_indices = range(len(self.documents))
        if filter_metadata:
            candidate_indices = [
                i for i in candidate_indices
                if all(self.documents[i].get("metadata", {}).get(k_) == v
                       for k_, v in filter_metadata.items())
            ]
        
        # Calculate scores on filtered candidates
        scored_docs = []
        for idx in candidate_indices:
            score = self._cosine_similarity(self.embeddings[idx], query_vec)
            scored_docs.append((idx, score))
            
        # Sort descending by score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in scored_docs[:k]:
            doc = self.documents[idx].copy()
            doc["score"] = float(score)
            results.append(doc)
            
        return results
