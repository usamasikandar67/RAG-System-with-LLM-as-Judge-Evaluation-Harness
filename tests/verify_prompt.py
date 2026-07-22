import unittest
# pyrefly: ignore [missing-import]
from ingestion.prompt_templates import format_user_prompt

class TestPromptTemplates(unittest.TestCase):
    def test_format_user_prompt(self):
        question = "What are the key points of cancer doc?"
        contexts = [
            {
                "id": "chunk_0",
                "text": "This is sample text about cancer treatment.",
                "metadata": {"source_file": "cancer_doc_0.txt", "clinical_category": "Oncology"}
            }
        ]
        
        prompt = format_user_prompt(question, contexts)
        
        self.assertIn("cancer_doc_0.txt", prompt)
        self.assertIn("Oncology", prompt)
        self.assertIn("This is sample text about cancer treatment.", prompt)
        self.assertIn("What are the key points of cancer doc?", prompt)

if __name__ == "__main__":
    unittest.main()
