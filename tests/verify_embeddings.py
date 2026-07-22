import unittest
import math
from embeddings.factory import MockEmbeddings

class TestEmbeddings(unittest.TestCase):
    def test_mock_embeddings_dimension(self):
        embed = MockEmbeddings(dimension=1536)
        vec = embed.embed_query("test query")
        self.assertEqual(len(vec), 1536)
        
        vecs = embed.embed_documents(["doc1", "doc2"])
        self.assertEqual(len(vecs), 2)
        self.assertEqual(len(vecs[0]), 1536)

    def test_mock_embeddings_determinism(self):
        embed = MockEmbeddings()
        vec1 = embed.embed_query("identical clinical phrase")
        vec2 = embed.embed_query("identical clinical phrase")
        vec3 = embed.embed_query("different clinical phrase")
        
        self.assertEqual(vec1, vec2)
        self.assertNotEqual(vec1, vec3)

    def test_mock_embeddings_normalization(self):
        embed = MockEmbeddings()
        vec = embed.embed_query("calculate magnitude")
        magnitude = math.sqrt(sum(x * x for x in vec))
        self.assertTrue(abs(magnitude - 1.0) < 1e-9)

if __name__ == "__main__":
    unittest.main()
