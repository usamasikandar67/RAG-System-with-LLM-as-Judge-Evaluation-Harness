import re
import math
from typing import List, Dict, Any
from retrieval.base import BaseRetriever

class BM25Retriever(BaseRetriever):
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[Dict[str, Any]] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_len: float = 0.0
        self.corpus_size: int = 0
        self.doc_term_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        # Clean alphanumeric lowercased tokens
        return re.findall(r"\w+", text.lower())

    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        self.documents = documents
        self.corpus_size = len(documents)
        if self.corpus_size == 0:
            self.avg_doc_len = 0.0
            return

        self.doc_lengths = []
        self.doc_term_freqs = []
        doc_contains_term = {}

        for doc in documents:
            tokens = self._tokenize(doc["text"])
            self.doc_lengths.append(len(tokens))
            
            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.doc_term_freqs.append(tf)

            for t in tf.keys():
                doc_contains_term[t] = doc_contains_term.get(t, 0) + 1

        self.avg_doc_len = sum(self.doc_lengths) / self.corpus_size

        # Compute standard IDF
        for term, freq in doc_contains_term.items():
            numerator = self.corpus_size - freq + 0.5
            denominator = freq + 0.5
            self.idf[term] = math.log((numerator / denominator) + 1.0)

    def retrieve(self, query: str, k: int = 3, filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        if self.corpus_size == 0:
            return []

        query_tokens = self._tokenize(query)
        scores = []

        for idx, doc in enumerate(self.documents):
            # Apply metadata pre-filter
            if filter_metadata:
                meta = doc.get("metadata", {})
                if not all(meta.get(fk) == fv for fk, fv in filter_metadata.items()):
                    continue

            score = 0.0
            doc_len = self.doc_lengths[idx]
            tf = self.doc_term_freqs[idx]

            for term in query_tokens:
                if term not in self.idf:
                    continue
                term_tf = tf.get(term, 0)
                idf_val = self.idf[term]
                numerator = term_tf * (self.k1 + 1)
                denominator = term_tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                score += idf_val * (numerator / denominator)

            scores.append((idx, score))

        # Sort descending by BM25 score
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:k]

        results = []
        for idx, score in sorted_scores:
            doc = self.documents[idx].copy()
            doc["score"] = float(score)
            results.append(doc)

        return results
