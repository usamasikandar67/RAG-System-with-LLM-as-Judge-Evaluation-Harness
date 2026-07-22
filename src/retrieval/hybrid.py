from typing import List, Dict, Any
from retrieval.base import BaseRetriever
from retrieval.vector import SimpleVectorRetriever
from retrieval.lexical import BM25Retriever

class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever: SimpleVectorRetriever, lexical_retriever: BM25Retriever, rrf_constant: int = 60):
        self.vector_retriever = vector_retriever
        self.lexical_retriever = lexical_retriever
        self.rrf_constant = rrf_constant

    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        self.vector_retriever.index_documents(documents)
        self.lexical_retriever.index_documents(documents)

    def retrieve(self, query: str, k: int = 3, filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        # Fetch slightly more candidates from each stream to allow fusion
        search_k = max(20, k * 2)
        vector_results = self.vector_retriever.retrieve(query, k=search_k, filter_metadata=filter_metadata)
        lexical_results = self.lexical_retriever.retrieve(query, k=search_k, filter_metadata=filter_metadata)

        fused_scores = {}
        doc_lookup = {}

        # Add vector ranks
        for rank, doc in enumerate(vector_results):
            doc_id = doc["id"]
            doc_lookup[doc_id] = doc
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (self.rrf_constant + (rank + 1))

        # Add lexical ranks
        for rank, doc in enumerate(lexical_results):
            doc_id = doc["id"]
            doc_lookup[doc_id] = doc
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (self.rrf_constant + (rank + 1))

        # Sort descending by fused RRF score
        sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in sorted_docs[:k]:
            doc = doc_lookup[doc_id].copy()
            doc["score"] = float(score)
            results.append(doc)

        return results
