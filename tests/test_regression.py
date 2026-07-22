import tempfile
import os
# pyrefly: ignore [missing-import]
from experiments.db import init_db, create_experiment, log_run
# pyrefly: ignore [missing-import]
from evaluation.regression_gate import evaluate_regression

def test_regression_gate_states():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_regression.db")
        init_db(db_path)
        
        # 1. Setup Baseline
        base_exp = create_experiment(db_path, "Baseline Exp", {"chunk_size": 800})
        base_report = {
            "test_id": "TC-001",
            "latency": 0.1,
            "cost": 0.0001,
            "hit_1": 1.0, "hit_3": 1.0, "hit_5": 1.0, "reciprocal_rank": 1.0, "average_precision": 1.0,
            "correctness": 0.9,
            "completeness": 0.9,
            "faithfulness": 0.9,
            "citation_accuracy": 1.0,
            "clinical_utility": 0.9
        }
        log_run(db_path, base_exp, base_report, {"question": "q", "response_text": "r"})
        
        # 2. Setup PASS Candidate (similar metrics)
        pass_exp = create_experiment(db_path, "Pass Candidate Exp", {"chunk_size": 800})
        pass_report = base_report.copy()
        pass_report["correctness"] = 0.88 # -0.02 delta (threshold is -0.05)
        log_run(db_path, pass_exp, pass_report, {"question": "q", "response_text": "r"})
        
        report_pass = evaluate_regression(db_path, pass_exp, base_exp)
        assert report_pass["status"] == "PASS"
        assert report_pass["metrics"]["correctness"]["status"] == "PASS"

        # 3. Setup FAIL Candidate (regression in correctness)
        fail_exp = create_experiment(db_path, "Fail Candidate Exp", {"chunk_size": 800})
        fail_report = base_report.copy()
        fail_report["correctness"] = 0.80 # -0.10 delta (fails since < -0.05)
        log_run(db_path, fail_exp, fail_report, {"question": "q", "response_text": "r"})
        
        report_fail = evaluate_regression(db_path, fail_exp, base_exp)
        assert report_fail["status"] == "FAIL"
        assert report_fail["metrics"]["correctness"]["status"] == "FAIL"

        # 4. Setup WARN Candidate (latency increase > 50%)
        warn_exp = create_experiment(db_path, "Warn Candidate Exp", {"chunk_size": 800})
        warn_report = base_report.copy()
        warn_report["latency"] = 0.2 # 2.0x increase (threshold is > 1.5x)
        log_run(db_path, warn_exp, warn_report, {"question": "q", "response_text": "r"})
        
        report_warn = evaluate_regression(db_path, warn_exp, base_exp)
        assert report_warn["status"] == "WARN"
        assert report_warn["metrics"]["latency"]["status"] == "WARN"
