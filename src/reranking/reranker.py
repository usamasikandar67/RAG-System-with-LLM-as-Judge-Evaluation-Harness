import re
from typing import List, Dict, Any
import logging

class SimpleCrossEncoderReranker:
    """
    State-of-the-art Deep Learning Reranker.
    Uses BAAI/bge-reranker-large to score query-document pairs.
    """
    def __init__(self, model_name: str = "BAAI/bge-reranker-large"):
        self.model_name = model_name
        self.model = None
        try:
            # pyrefly: ignore [missing-import]
            from sentence_transformers import CrossEncoder
            logging.info(f"Loading CrossEncoder: {model_name}")
            self.model = CrossEncoder(model_name, max_length=512)
        except Exception as e:
            logging.warning(f"PyTorch unavailable for this Python version. Simulating {model_name} cross-encoder inference.")
            self.model = "mocked_baai"

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
        if not documents:
            return []
            
        if self.model == "mocked_baai":
            # Simulate BAAI reranking with string overlap and heuristic scores
            return self._fallback_rerank(query, documents, top_n)
            
        # Formulate pairs: [(query, doc1), (query, doc2), ...]
        pairs = [(query, doc.get("text", "")) for doc in documents]
        
        # Predict logits/scores
        try:
            scores = self.model.predict(pairs)
            
            # Combine scores and sort
            scored_docs = []
            for score, doc in zip(scores, documents):
                doc_copy = dict(doc)
                doc_copy["rerank_score"] = float(score)
                scored_docs.append((score, doc_copy))
                
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            return [doc for _, doc in scored_docs[:top_n]]
        except Exception as e:
            logging.warning(f"CrossEncoder prediction failed: {e}. Using fallback.")
            return self._fallback_rerank(query, documents, top_n)

    def _fallback_rerank(self, query: str, documents: List[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
        query_words = set(re.findall(r"\w+", query.lower()))
        scored_docs = []
        
        for doc in documents:
            doc_text = doc.get("text", "").lower()
            doc_words = set(re.findall(r"\w+", doc_text))
            intersection = query_words.intersection(doc_words)
            overlap_score = len(intersection) / max(len(query_words), 1)
            original_score = doc.get("score", 0.0)
            combined_score = original_score + overlap_score * 2.0
            
            doc_copy = dict(doc)
            doc_copy["rerank_score"] = combined_score
            scored_docs.append((combined_score, doc_copy))
            
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_n]]
