# tests/verify_chatbot.py

import os
import unittest
import shutil
import sqlite3
# pyrefly: ignore [missing-import]
from experiments.db import init_db, save_chat_message, load_chat_history
# pyrefly: ignore [missing-import]
from ingestion.ingestion import ingest_documents

TEST_DB = "experiments/test_chatbot_results.db"
TEST_KB_DIR = "datasets/test_kb_chatbot"

class TestChatbotAndGuidelines(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Setup test paths
        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)
        os.makedirs(TEST_KB_DIR, exist_ok=True)
        init_db(TEST_DB)
        
    @classmethod
    def tearDownClass(cls):
        # Cleanup
        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)
        if os.path.exists(TEST_KB_DIR):
            shutil.rmtree(TEST_KB_DIR)
            
    def test_01_chat_history_persistence(self):
        session_id = "test_doc_session_123"
        contexts = [
            {"text": "Sample lymphoma guidelines text.", "metadata": {"source_file": "doc_a.txt", "clinical_category": "Hematology"}}
        ]
        
        # 1. Save messages
        save_chat_message(TEST_DB, session_id, "user", "What is follicular lymphoma treatment?", [])
        save_chat_message(TEST_DB, session_id, "assistant", "Observation is standard first-line approach.", contexts)
        
        # 2. Retrieve history
        history = load_chat_history(TEST_DB, session_id)
        
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[0]["content"], "What is follicular lymphoma treatment?")
        self.assertEqual(history[1]["role"], "assistant")
        self.assertEqual(history[1]["content"], "Observation is standard first-line approach.")
        self.assertEqual(len(history[1]["contexts"]), 1)
        self.assertEqual(history[1]["contexts"][0]["metadata"]["source_file"], "doc_a.txt")
        print("[Pass] Chat history persistence verified in SQLite.")
        
    def test_02_clinical_data_update_and_reindex(self):
        doc_name = "test_temp_lymphoma.txt"
        filepath = os.path.join(TEST_KB_DIR, doc_name)
        
        # Write guideline file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("Clinical protocol target keyword: R-CHOP regimen for advanced stages.")
            
        chunks = ingest_documents(TEST_KB_DIR, chunk_size=800, chunk_overlap=150)
        self.assertTrue(len(chunks) > 0)
        
        found = False
        for c in chunks:
            if c["metadata"]["source_file"] == doc_name:
                self.assertIn("R-CHOP", c["text"])
                found = True
                break
        self.assertTrue(found)
        print("[Pass] Guidelines upload and parsing index pipeline verified.")

if __name__ == "__main__":
    unittest.main()
