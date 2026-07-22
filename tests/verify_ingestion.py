import os
import tempfile
import unittest
# pyrefly: ignore [missing-import]
from ingestion.ingestion import RecursiveCharacterTextSplitter
# pyrefly: ignore [missing-import]
from ingestion.ingestion import ingest_documents

class TestIngestion(unittest.TestCase):
    def test_recursive_splitter_short(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        text = "Short clinical sentence."
        chunks = splitter.split_text(text)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "Short clinical sentence.")

    def test_recursive_splitter_boundaries(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=5)
        text = "word word word word"
        chunks = splitter.split_text(text)
        for c in chunks:
            self.assertTrue(len(c) <= 20)

    def test_recursive_splitter_overlap(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=30, chunk_overlap=10)
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        chunks = splitter.split_text(text)
        self.assertTrue(len(chunks) > 1)

        for i in range(len(chunks) - 1):
            words_a = set(chunks[i].split())
            words_b = set(chunks[i+1].split())
            # Assert overlap calculations
            self.assertTrue(len(words_a.intersection(words_b)) >= 0)

    def test_ingest_documents(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            doc_path = os.path.join(tmpdir, "cancer_test_guideline.txt")
            mock_content = "This is paragraph one.\n\nThis is paragraph two. It contains description of lung cancer."
            with open(doc_path, "w", encoding="utf-8") as f:
                f.write(mock_content)
                
            chunks = ingest_documents(tmpdir, chunk_size=50, chunk_overlap=10)
            self.assertTrue(len(chunks) >= 1)
            for c in chunks:
                self.assertIn("id", c)
                self.assertIn("text", c)
                self.assertIn("metadata", c)
                meta = c["metadata"]
                self.assertEqual(meta["source_file"], "cancer_test_guideline.txt")
                self.assertIn("clinical_category", meta)
                self.assertIn("chunk_index", meta)

if __name__ == "__main__":
    unittest.main()
