import uuid
# pyrefly: ignore [missing-import]
import chromadb
from typing import List, Dict, Any, Optional
from retrieval.base import BaseRetriever
from embeddings.factory import BaseEmbeddings

class ChromaRetriever(BaseRetriever):
    """
    Production-grade Vector Retriever using ChromaDB.
    Supports in-memory or persistent storage.
    """
    def __init__(
        self, 
        embedding_model: BaseEmbeddings, 
        persist_directory: Optional[str] = None, 
        collection_name: str = "documents"
    ):
        self.embedding_model = embedding_model
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        if not documents:
            return
            
        texts = [doc["text"] for doc in documents]
        embeddings = self.embedding_model.embed_documents(texts)
        ids = [doc.get("id", str(uuid.uuid4())) for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

    def retrieve(self, query: str, k: int = 3, filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        query_vec = self.embedding_model.embed_query(query)
        
        where_clause = filter_metadata if filter_metadata else None
        
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=k,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )
        
        retrieved_docs = []
        if not results["ids"] or len(results["ids"][0]) == 0:
            return retrieved_docs
            
        for i in range(len(results["ids"][0])):
            distance = float(results["distances"][0][i])
            doc = {
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"][0] else {},
                # Convert distance to a similarity score where higher is better
                "score": 1.0 / (1.0 + distance)
            }
            retrieved_docs.append(doc)
            
        return retrieved_docs
