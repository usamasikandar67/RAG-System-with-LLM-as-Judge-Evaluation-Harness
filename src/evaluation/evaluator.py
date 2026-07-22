# evaluation/evaluator.py

from typing import Dict, Any, List
from judge.factory import BaseJudge

class EvaluationEngine:
    def __init__(self, judge_model: BaseJudge):
        self.judge_model = judge_model

    def calculate_retrieval_metrics(self, retrieved_contexts: List[Dict[str, Any]], expected_documents: List[str]) -> Dict[str, float]:
        retrieved_sources = [doc.get("metadata", {}).get("source_file") for doc in retrieved_contexts]
        
        # 1. Hits@K calculations
        hit_1 = 1.0 if any(expected in retrieved_sources[:1] for expected in expected_documents) else 0.0
        hit_3 = 1.0 if any(expected in retrieved_sources[:3] for expected in expected_documents) else 0.0
        hit_5 = 1.0 if any(expected in retrieved_sources[:5] for expected in expected_documents) else 0.0

        # 2. Reciprocal Rank (RR)
        rr = 0.0
        for rank, src in enumerate(retrieved_sources):
            if src in expected_documents:
                rr = 1.0 / (rank + 1)
                break

        # 3. Average Precision (AP)
        hits = 0
        sum_precisions = 0.0
        for rank, src in enumerate(retrieved_sources):
            if src in expected_documents:
                hits += 1
                precision_at_rank = hits / (rank + 1)
                sum_precisions += precision_at_rank
        ap = sum_precisions / max(1, len(expected_documents))

        return {
            "hit_1": float(hit_1),
            "hit_3": float(hit_3),
            "hit_5": float(hit_5),
            "reciprocal_rank": float(rr),
            "average_precision": float(ap)
        }

    def evaluate(self, query_result: Dict[str, Any], golden_case: Dict[str, Any], tracer=None, trace_id: str=None) -> Dict[str, Any]:
        retrieved_contexts = query_result.get("retrieved_contexts", [])
        expected_documents = golden_case.get("expected_documents", [])
        
        # Compute retrieval metrics
        ret_metrics = self.calculate_retrieval_metrics(retrieved_contexts, expected_documents)

        # Prepare judge contexts as strings
        contexts_texts = [c.get("text", "") for c in retrieved_contexts]
        
        # Call LLM-as-Judge
        judge_metrics = {}
        if tracer and trace_id:
            with tracer.span(trace_id, "Evaluation", {"judge": self.judge_model.__class__.__name__}) as span:
                judge_metrics = self.judge_model.evaluate_response(
                    question=query_result.get("question", ""),
                    response=query_result.get("response_text", ""),
                    ground_truth=golden_case.get("ground_truth_answer", ""),
                    contexts=contexts_texts
                )
                span["output"] = judge_metrics
        else:
            judge_metrics = self.judge_model.evaluate_response(
                question=query_result.get("question", ""),
                response=query_result.get("response_text", ""),
                ground_truth=golden_case.get("ground_truth_answer", ""),
                contexts=contexts_texts
            )

        # Merge results
        eval_report = {
            "test_id": golden_case.get("test_id", "TC-unknown"),
            "clinical_category": golden_case.get("clinical_category", "Oncology"),
            "latency": query_result.get("latency", 0.0),
            "cost": query_result.get("cost", 0.0),
            **ret_metrics,
            **judge_metrics
        }

        # Calculate a unified clinical utility score
        # Weighted blend: 30% correctness, 30% faithfulness, 20% completeness, 20% citation_accuracy
        clinical_utility = (
            judge_metrics.get("correctness", 0.0) * 0.3 +
            judge_metrics.get("faithfulness", 0.0) * 0.3 +
            judge_metrics.get("completeness", 0.0) * 0.2 +
            judge_metrics.get("citation_accuracy", 0.0) * 0.2
        )
        eval_report["clinical_utility"] = round(clinical_utility, 2)

        return eval_report
