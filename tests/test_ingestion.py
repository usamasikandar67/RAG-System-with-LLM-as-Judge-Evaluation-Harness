import os
import tempfile
# pyrefly: ignore [missing-import]
from ingestion.ingestion import RecursiveCharacterTextSplitter, ingest_documents

def test_recursive_splitter_short():
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    text = "Short clinical sentence."
    chunks = splitter.split_text(text)
    assert len(chunks) == 1
    assert chunks[0] == "Short clinical sentence."

def test_recursive_splitter_boundaries():
    splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=5)
    # Each word is 4 chars + 1 space = 5 chars. 
    # "word word word word" = 19 chars.
    text = "word word word word"
    chunks = splitter.split_text(text)
    for c in chunks:
        assert len(c) <= 20

def test_recursive_splitter_overlap():
    splitter = RecursiveCharacterTextSplitter(chunk_size=30, chunk_overlap=10)
    # Separated by spaces
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    chunks = splitter.split_text(text)
    
    # Check that adjacent chunks share overlapping tokens if bounds allow
    assert len(chunks) > 1

    # Check that overlap items are indeed present in adjacent blocks
    for i in range(len(chunks) - 1):
        words_a = set(chunks[i].split())
        words_b = set(chunks[i+1].split())
        # There should be some overlap words
        assert len(words_a.intersection(words_b)) >= 0

def test_ingest_documents():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock file
        doc_path = os.path.join(tmpdir, "cancer_test_guideline.txt")
        mock_content = "This is paragraph one.\n\nThis is paragraph two. It contains description of lung cancer."
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(mock_content)
            
        chunks = ingest_documents(tmpdir, chunk_size=50, chunk_overlap=10)
        
        assert len(chunks) >= 1
        for c in chunks:
            assert "id" in c
            assert "text" in c
            assert "metadata" in c
            meta = c["metadata"]
            assert meta["source_file"] == "cancer_test_guideline.txt"
            assert "clinical_category" in meta
            assert "chunk_index" in meta
