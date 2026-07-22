# pyrefly: ignore [missing-import]
from ingestion.pipeline import RAGPipeline
# pyrefly: ignore [missing-import]
from retrieval.lexical import BM25Retriever
import os

DOCS = [
    {
        "id": "doc_chemo",
        "text": "chemotherapy is a standard treatment used for lung cancer.",
        "metadata": {"source_file": "cancer_doc_0.txt", "clinical_category": "Oncology"}
    }
]

def test_rag_pipeline_generation():
    retriever = BM25Retriever()
    retriever.index_documents(DOCS)
    
    pipeline = RAGPipeline(retriever=retriever, generator_model="mock")
    
    # Query matching indexed document text
    results = pipeline.generate("chemotherapy treatment for lung cancer", k=1)
    
    assert results["question"] == "chemotherapy treatment for lung cancer"
    assert "cancer_doc_0.txt" in results["response_text"]
    assert len(results["retrieved_contexts"]) == 1
    assert results["latency"] > 0.0
    assert results["token_usage"]["input_tokens"] > 0
    assert results["token_usage"]["output_tokens"] > 0
    assert results["cost"] >= 0.0
