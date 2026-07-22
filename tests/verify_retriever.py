import unittest
# pyrefly: ignore [missing-import]
from retrieval.vector import SimpleVectorRetriever
# pyrefly: ignore [missing-import]
from retrieval.lexical import BM25Retriever
# pyrefly: ignore [missing-import]
from retrieval.hybrid import HybridRetriever
from embeddings.factory import MockEmbeddings

DOCS = [
    {
        "id": "doc_chemo",
        "text": "chemotherapy is a standard treatment used for lung cancer.",
        "metadata": {"topic": "Lung Cancer"}
    },
    {
        "id": "doc_watchful",
        "text": "watchful waiting is the primary monitoring method for MGUS.",
        "metadata": {"topic": "MGUS"}
    }
]

class TestRetrieverStrategies(unittest.TestCase):
    def test_lexical_retriever(self):
        retriever = BM25Retriever()
        retriever.index_documents(DOCS)
        
        # Matches chemotherapy keyword
        results = retriever.retrieve("chemotherapy", k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "doc_chemo")
        
        # Matches watchful waiting keyword
        results_2 = retriever.retrieve("waiting monitor", k=1)
        self.assertEqual(len(results_2), 1)
        self.assertEqual(results_2[0]["id"], "doc_watchful")

    def test_hybrid_retriever(self):
        embed = MockEmbeddings()
        vector_ret = SimpleVectorRetriever(embedding_model=embed)
        lexical_ret = BM25Retriever()
        
        hybrid_ret = HybridRetriever(vector_ret, lexical_ret)
        hybrid_ret.index_documents(DOCS)
        
        # Matches exact string to trigger high ranking in both streams
        results = hybrid_ret.retrieve("chemotherapy is a standard treatment used for lung cancer.", k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "doc_chemo")

if __name__ == "__main__":
    unittest.main()
