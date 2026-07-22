# pyrefly: ignore [missing-import]
from ingestion.prompt_templates import format_user_prompt

def test_format_user_prompt():
    question = "What are the key points of cancer doc?"
    contexts = [
        {
            "id": "chunk_0",
            "text": "This is sample text about cancer treatment.",
            "metadata": {"source_file": "cancer_doc_0.txt", "clinical_category": "Oncology"}
        }
    ]
    
    prompt = format_user_prompt(question, contexts)
    
    assert "cancer_doc_0.txt" in prompt
    assert "Oncology" in prompt
    assert "This is sample text about cancer treatment." in prompt
    assert "What are the key points of cancer doc?" in prompt
