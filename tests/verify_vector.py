import unittest
# pyrefly: ignore [missing-import]
from retrieval.vector import SimpleVectorRetriever
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

class TestVectorRetriever(unittest.TestCase):
    def test_simple_vector_retriever_retrieve(self):
        embed = MockEmbeddings()
        retriever = SimpleVectorRetriever(embedding_model=embed)
        retriever.index_documents(DOCS)
        
        # Query related to chemo (exact match to guarantee score 1.0)
        results = retriever.retrieve("chemotherapy is a standard treatment used for lung cancer.", k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "doc_chemo")
        self.assertTrue(results[0]["score"] > 0.99)
        
        # Query related to watchful waiting (exact match)
        results_2 = retriever.retrieve("watchful waiting is the primary monitoring method for MGUS.", k=1)
        self.assertEqual(len(results_2), 1)
        self.assertEqual(results_2[0]["id"], "doc_watchful")
        self.assertTrue(results_2[0]["score"] > 0.99)


    def test_simple_vector_retriever_empty(self):
        embed = MockEmbeddings()
        retriever = SimpleVectorRetriever(embedding_model=embed)
        results = retriever.retrieve("query text", k=3)
        self.assertEqual(len(results), 0)

if __name__ == "__main__":
    unittest.main()
