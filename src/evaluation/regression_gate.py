# evaluation/regression_gate.py

import sys
import argparse
from typing import Dict, Any, List
from experiments.db import get_db_connection

def evaluate_regression(db_path: str, candidate_exp_id: str, baseline_exp_id: str) -> Dict[str, Any]:
    def fetch_averages(exp_id: str) -> Dict[str, float]:
        conn = get_db_connection(db_path)
        cursor = conn.cursor()
        
        # Get averages of all evaluation metrics
        cursor.execute("""
            SELECT e.metric_name, AVG(e.metric_value) as avg_val
            FROM evaluations e
            JOIN experiment_runs r ON e.eval_id = r.eval_id
            WHERE r.experiment_id = ?
            GROUP BY e.metric_name
        """, (exp_id,))
        rows = cursor.fetchall()
        
        # Get averages of generation metrics (latency_ms, cost)
        # Note: Cost isn't logged to ERD by default in our current setup, but latency is.
        cursor.execute("""
            SELECT AVG(g.latency_ms) as avg_latency
            FROM generations g
            JOIN experiment_runs r ON g.generation_id = r.generation_id
            WHERE r.experiment_id = ?
        """, (exp_id,))
        gen_row = cursor.fetchone()
        
        conn.close()
        
        if not rows and not gen_row:
             return {}
             
        averages = {row["metric_name"]: float(row["avg_val"]) for row in rows}
        averages["latency"] = float(gen_row["avg_latency"]) if gen_row and gen_row["avg_latency"] is not None else 0.0
        averages["cost"] = 0.0 # Placeholder since cost was removed from generation table
        
        # Ensure all metrics exist to prevent KeyError
        fields = [
            "hit_1", "hit_3", "hit_5", "reciprocal_rank", "average_precision",
            "correctness", "completeness", "faithfulness", "citation_accuracy",
            "clinical_utility"
        ]
        for f in fields:
            if f not in averages:
                averages[f] = 0.0
                
        return averages

    b_avg = fetch_averages(baseline_exp_id)
    c_avg = fetch_averages(candidate_exp_id)
    
    if not b_avg:
        raise ValueError(f"No runs found for baseline experiment {baseline_exp_id}")
    if not c_avg:
        raise ValueError(f"No runs found for candidate experiment {candidate_exp_id}")

    metrics = {}
    status = "PASS"

    # Delta based checks: threshold is a drop of > 0.05 (i.e. delta < -0.05 is a FAIL)
    delta_metrics = ["average_precision", "correctness", "faithfulness"]
    for m in delta_metrics:
        b_val = b_avg[m]
        c_val = c_avg[m]
        delta = c_val - b_val
        
        m_status = "PASS"
        if delta < -0.05:
            m_status = "FAIL"
            status = "FAIL"
            
        metrics[m] = {
            "baseline": round(b_val, 4),
            "candidate": round(c_val, 4),
            "delta": round(delta, 4),
            "status": m_status
        }

    # Ratio based checks (latency, cost) -> threshold is > 50% increase (ratio > 1.5 is a WARN)
    ratio_metrics = ["latency", "cost"]
    for m in ratio_metrics:
        b_val = b_avg[m]
        c_val = c_avg[m]
        ratio = (c_val / b_val) if b_val > 0.0 else 1.0
        
        m_status = "PASS"
        if ratio > 1.5:
            m_status = "WARN"
            if status != "FAIL":
                status = "WARN"
                
        metrics[m] = {
            "baseline": round(b_val, 4),
            "candidate": round(c_val, 4),
            "ratio": round(ratio, 4),
            "status": m_status
        }

    return {
        "status": status,
        "baseline_exp_id": baseline_exp_id,
        "candidate_exp_id": candidate_exp_id,
        "metrics": metrics
    }

def main():
    parser = argparse.ArgumentParser(description="Clinical RAG Regression Testing Gate")
    parser.add_argument("--db-path", default="src/experiments/evaluation_results.db", help="Path to SQLite evaluations DB")

    parser.add_argument("--baseline", required=True, help="Baseline experiment ID")
    parser.add_argument("--candidate", required=True, help="Candidate experiment ID")
    args = parser.parse_args()

    try:
        report = evaluate_regression(args.db_path, args.candidate, args.baseline)
        print("\n=== CLINICAL RAG REGRESSION TESTING GATE REPORT ===")
        print(f"Status: {report['status']}")
        print(f"Baseline ID: {report['baseline_exp_id']}")
        print(f"Candidate ID: {report['candidate_exp_id']}")
        print("---------------------------------------------------")
        for metric, data in report["metrics"].items():
            if "delta" in data:
                print(f"{metric:20}: Baseline={data['baseline']:.4f}, Candidate={data['candidate']:.4f}, Delta={data['delta']:.4f} [{data['status']}]")
            else:
                print(f"{metric:20}: Baseline={data['baseline']:.4f}, Candidate={data['candidate']:.4f}, Ratio={data['ratio']:.2f}x [{data['status']}]")
        print("===================================================\n")
        
        if report["status"] == "FAIL":
            sys.exit(1)
        else:
            sys.exit(0)
    except Exception as e:
        print(f"Error executing regression gate: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()
