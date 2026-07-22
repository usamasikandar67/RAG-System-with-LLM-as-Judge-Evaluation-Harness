# pyrefly: ignore [missing-import]
from retrieval.vector import SimpleVectorRetriever
# pyrefly: ignore [missing-import]
from retrieval.chroma import ChromaRetriever
# pyrefly: ignore [missing-import]
from retrieval.lexical import BM25Retriever
# pyrefly: ignore [missing-import]
from retrieval.hybrid import HybridRetriever
from embeddings.factory import MockEmbeddings
import uuid

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

def test_lexical_retriever():
    retriever = BM25Retriever()
    retriever.index_documents(DOCS)
    
    # Matches chemotherapy keyword
    results = retriever.retrieve("chemotherapy", k=1)
    assert len(results) == 1
    assert results[0]["id"] == "doc_chemo"
    
    # Matches watchful waiting keyword
    results_2 = retriever.retrieve("waiting monitor", k=1)
    assert len(results_2) == 1
    assert results_2[0]["id"] == "doc_watchful"

def test_chroma_retriever():
    embed = MockEmbeddings()
    collection_name = f"test_collection_{uuid.uuid4().hex}"
    retriever = ChromaRetriever(embedding_model=embed, collection_name=collection_name)
    retriever.index_documents(DOCS)
    
    results = retriever.retrieve("chemotherapy is a standard treatment used for lung cancer.", k=1)
    assert len(results) == 1
    assert results[0]["id"] == "doc_chemo"
    assert "score" in results[0]

def test_hybrid_retriever():
    embed = MockEmbeddings()
    vector_ret = SimpleVectorRetriever(embedding_model=embed)
    lexical_ret = BM25Retriever()
    
    hybrid_ret = HybridRetriever(vector_ret, lexical_ret)
    hybrid_ret.index_documents(DOCS)
    
    # Matches exact string to trigger high ranking in both streams
    results = hybrid_ret.retrieve("chemotherapy is a standard treatment used for lung cancer.", k=1)
    assert len(results) == 1
    assert results[0]["id"] == "doc_chemo"
