import unittest
# pyrefly: ignore [missing-import]
import tempfile
import os
# pyrefly: ignore [missing-import]
from experiments.db import init_db, create_experiment, log_run, get_runs_for_experiment

class TestExperimentsDB(unittest.TestCase):
    def test_sqlite_logging_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_evals.db")
            
            # 1. Initialize schema
            init_db(db_path)
            self.assertTrue(os.path.exists(db_path))
            
            # 2. Register experiment
            settings = {"chunk_size": 800, "chunk_overlap": 150, "retriever": "hybrid"}
            exp_id = create_experiment(db_path, "Test Ingestion Run", settings)
            self.assertTrue(exp_id.startswith("exp_"))
            
            # 3. Log query evaluation run
            run_report = {
                "test_id": "TC-001",
                "clinical_category": "Oncology",
                "latency": 0.15,
                "cost": 0.0002,
                "hit_1": 1.0,
                "hit_3": 1.0,
                "hit_5": 1.0,
                "reciprocal_rank": 1.0,
                "average_precision": 1.0,
                "correctness": 0.8,
                "completeness": 0.7,
                "faithfulness": 0.9,
                "citation_accuracy": 1.0,
                "clinical_utility": 0.85
            }
            query_details = {
                "question": "What is NSCLC protocol?",
                "response_text": "NSCLC protocol references [cancer_doc_0.txt] guidelines."
            }
            
            run_id = log_run(db_path, exp_id, run_report, query_details)
            self.assertTrue(run_id.startswith("run_"))
            
            # 4. Fetch metrics
            runs = get_runs_for_experiment(db_path, exp_id)
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["test_id"], "TC-001")
            self.assertEqual(runs[0]["correctness"], 0.8)
            self.assertEqual(runs[0]["clinical_utility"], 0.85)
            self.assertEqual(runs[0]["question"], "What is NSCLC protocol?")

if __name__ == "__main__":
    unittest.main()
