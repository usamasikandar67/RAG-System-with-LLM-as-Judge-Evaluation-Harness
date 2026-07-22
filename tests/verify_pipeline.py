import unittest
# pyrefly: ignore [missing-import]
from ingestion.pipeline import RAGPipeline
# pyrefly: ignore [missing-import]
from retrieval.lexical import BM25Retriever

DOCS = [
    {
        "id": "doc_chemo",
        "text": "chemotherapy is a standard treatment used for lung cancer.",
        "metadata": {"source_file": "cancer_doc_0.txt", "clinical_category": "Oncology"}
    }
]

class TestRAGPipeline(unittest.TestCase):
    def test_rag_pipeline_generation(self):
        retriever = BM25Retriever()
        retriever.index_documents(DOCS)
        
        pipeline = RAGPipeline(retriever=retriever, generator_model="mock")
        
        results = pipeline.generate("chemotherapy treatment for lung cancer", k=1)
        
        self.assertEqual(results["question"], "chemotherapy treatment for lung cancer")
        self.assertIn("cancer_doc_0.txt", results["response_text"])
        self.assertEqual(len(results["retrieved_contexts"]), 1)
        self.assertTrue(results["latency"] > 0.0)
        self.assertTrue(results["token_usage"]["input_tokens"] > 0)
        self.assertTrue(results["token_usage"]["output_tokens"] > 0)
        self.assertTrue(results["cost"] >= 0.0)

if __name__ == "__main__":
    unittest.main()
