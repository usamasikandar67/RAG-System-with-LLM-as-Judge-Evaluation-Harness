# pyrefly: ignore [missing-import]
from evaluation.evaluator import EvaluationEngine
# pyrefly: ignore [missing-import]
from judge.factory import MockJudge

def test_retrieval_metrics_calculation():
    engine = EvaluationEngine(judge_model=MockJudge())
    
    # Test doc list and target expected doc
    retrieved = [
        {"metadata": {"source_file": "doc_a.txt"}},
        {"metadata": {"source_file": "doc_b.txt"}},
    ]
    expected = ["doc_b.txt"]
    
    metrics = engine.calculate_retrieval_metrics(retrieved, expected)
    
    assert metrics["hit_1"] == 0.0
    assert metrics["hit_3"] == 1.0
    assert metrics["reciprocal_rank"] == 0.5 # rank 2 -> 1/2 = 0.5
    assert metrics["average_precision"] == 0.5

def test_pipeline_evaluation_reporting():
    engine = EvaluationEngine(judge_model=MockJudge())
    
    query_result = {
        "question": "What is treatment of lung cancer?",
        "response_text": "Recommended protocol is [cancer_doc_0.txt]. Standard chemotherapy is advised.",
        "retrieved_contexts": [
            {"text": "Chemotherapy is standard treatment for lung cancer.", "metadata": {"source_file": "cancer_doc_0.txt"}}
        ],
        "latency": 0.12,
        "cost": 0.0001
    }
    
    golden_case = {
        "test_id": "TC-001",
        "question": "What is treatment of lung cancer?",
        "ground_truth_answer": "Chemotherapy is the standard treatment.",
        "expected_documents": ["cancer_doc_0.txt"],
        "clinical_category": "Oncology"
    }
    
    report = engine.evaluate(query_result, golden_case)
    
    assert report["test_id"] == "TC-001"
    assert report["clinical_category"] == "Oncology"
    assert report["hit_1"] == 1.0
    assert report["reciprocal_rank"] == 1.0
    assert report["citation_accuracy"] == 1.0
    assert report["clinical_utility"] > 0.0
