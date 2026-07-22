import math
# pyrefly: ignore [missing-import]
from embeddings.factory import MockEmbeddings

def test_mock_embeddings_dimension():
    embed = MockEmbeddings(dimension=1536)
    vec = embed.embed_query("test query")
    assert len(vec) == 1536
    
    vecs = embed.embed_documents(["doc1", "doc2"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 1536

def test_mock_embeddings_determinism():
    embed = MockEmbeddings()
    vec1 = embed.embed_query("identical clinical phrase")
    vec2 = embed.embed_query("identical clinical phrase")
    vec3 = embed.embed_query("different clinical phrase")
    
    assert vec1 == vec2
    assert vec1 != vec3

def test_mock_embeddings_normalization():
    embed = MockEmbeddings()
    vec = embed.embed_query("calculate magnitude")
    # Magnitude should be exactly 1.0
    magnitude = math.sqrt(sum(x * x for x in vec))
    assert abs(magnitude - 1.0) < 1e-9
