# run_pipeline.py

import os
import sys
import json
import argparse
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv()
from typing import Dict, Any

# Ensure src directory is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
# pyrefly: ignore [missing-import]
from ingestion.ingestion import ingest_documents
# pyrefly: ignore [missing-import]
from embeddings.factory import get_embedding_model  
# pyrefly: ignore [missing-import]
from retrieval.vector import SimpleVectorRetriever
# pyrefly: ignore [missing-import]
from retrieval.lexical import BM25Retriever
# pyrefly: ignore [missing-import]
from retrieval.hybrid import HybridRetriever
# pyrefly: ignore [missing-import]
from ingestion.pipeline import RAGPipeline
# pyrefly: ignore [missing-import]
from judge.factory import get_judge_model
# pyrefly: ignore [missing-import]
from evaluation.evaluator import EvaluationEngine
# pyrefly: ignore [missing-import]
from experiments.db import (
    init_db, create_experiment, log_query, log_retrieval, 
    log_generation, log_evaluation, log_experiment_run, get_db_connection
)
# pyrefly: ignore [missing-import]
from evaluation.regression_gate import evaluate_regression
# pyrefly: ignore [missing-import]
from langfuse.tracer import LangfuseTracer

def run_evaluation_pipeline(args) -> str:
    print("\n=== STARTING CLINICAL RAG EVALUATION PIPELINE RUN ===")
    
    # 1. Initialize SQLite tracker DB
    init_db(args.db_path)
    
    # 2. Ingest Reference Documents
    kb_dir = "datasets/knowledge_base"
    print(f"[1/6] Ingesting documents from {kb_dir} (size={args.chunk_size}, overlap={args.chunk_overlap}, chunking={args.chunking})...")
    chunks = ingest_documents(kb_dir, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap, chunking_enabled=args.chunking)
    if not chunks:
        print("[Error] No document chunks found. Please run datasets/manager.py first.")
        sys.exit(2)
        
    # 3. Instantiate Retriever
    print(f"[2/6] Initializing embeddings and retriever type: {args.retriever}...")
    embed = get_embedding_model("mock")
    
    if args.retriever == "vector":
        retriever = SimpleVectorRetriever(embedding_model=embed)
    elif args.retriever == "lexical":
        retriever = BM25Retriever()
    else: # hybrid
        v_ret = SimpleVectorRetriever(embedding_model=embed)
        l_ret = BM25Retriever()
        retriever = HybridRetriever(vector_retriever=v_ret, lexical_retriever=l_ret)
        
    # Index document chunks
    retriever.index_documents(chunks)
    
    # 4. Instantiate Pipeline & Evaluator Engine
    print(f"[3/6] Setting up generation pipeline (generator={args.generator}) and judge...")
    pipeline = RAGPipeline(retriever=retriever, generator_model=args.generator)
    judge = get_judge_model("mock")
    evaluator = EvaluationEngine(judge_model=judge)
    
    # 5. Create Experiment Log Entry
    settings = {
        "retriever": args.retriever,
        "generator": args.generator,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap
    }
    exp_id = create_experiment(args.db_path, f"Pipeline {args.retriever.upper()} Run", args.retriever, settings)
    print(f"[4/6] Registered experiment ID: {exp_id}")
    
    # 6. Load Golden Dataset cases
    golden_path = "datasets/golden_dataset.json"
    if not os.path.exists(golden_path):
        print(f"[Error] Golden dataset {golden_path} does not exist. Run datasets/manager.py first.")
        sys.exit(2)
        
    with open(golden_path, "r", encoding="utf-8") as f:
        golden_cases = json.load(f)
        
    print(f"[5/6] Running evaluation over {len(golden_cases)} golden test cases...")
    
    # Initialize Tracer
    tracer = LangfuseTracer(args.db_path)

    for idx, case in enumerate(golden_cases):
        # 1. Log Query
        query_id = log_query(args.db_path, case["question"])

        # Start Trace
        trace_id = tracer.create_trace(
            name=f"Eval_{args.retriever}_{args.generator}",
            session_id=exp_id,
            user_id="eval_runner",
            input_text=case["question"]
        )

        # 2. Run Generation
        result = pipeline.generate(case["question"], k=3, tracer=tracer, trace_id=trace_id)
        
        # 3. Log Generation
        gen_id = log_generation(args.db_path, query_id, "prompt", args.generator, result["response_text"], result["latency"])
        
        # 4. Log Retrievals
        for i, ctx in enumerate(result.get("retrieved_contexts", [])):
            chunk_id = ctx.get("metadata", {}).get("chunk_id", "unknown_chunk")
            log_retrieval(args.db_path, query_id, chunk_id, rank=i+1, score=1.0)
            
        # 5. Evaluate
        report = evaluator.evaluate(result, case, tracer=tracer, trace_id=trace_id)
        
        # End Trace
        tracer.end_trace(
            trace_id=trace_id,
            output_text=result["response_text"],
            total_latency_ms=result["latency"] * 1000,
            total_cost=result["cost"],
            input_tokens=result.get("token_usage", {}).get("input_tokens", 0),
            output_tokens=result.get("token_usage", {}).get("output_tokens", 0),
            metadata={"report": report}
        )

        # 6. Log Evaluations & Experiment Runs
        for metric in ["hit_1", "hit_3", "hit_5", "reciprocal_rank", "average_precision", "correctness", "completeness", "faithfulness", "citation_accuracy", "clinical_utility"]:
            eval_id = log_evaluation(args.db_path, gen_id, None, metric, report.get(metric, 0.0), "judge_model")
            log_experiment_run(args.db_path, exp_id, gen_id, eval_id)
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(golden_cases)} test cases...")

    # 7. Print summary metrics
    conn = get_db_connection(args.db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT e.metric_name, AVG(e.metric_value) as avg_val
        FROM evaluations e
        JOIN experiment_runs r ON e.eval_id = r.eval_id
        WHERE r.experiment_id = ?
        GROUP BY e.metric_name
    """, (exp_id,))
    rows = cursor.fetchall()
    conn.close()
    
    print("\n[6/6] Run completed. Summary Averages:")
    print("---------------------------------------------------")
    for row in rows:
        print(f"  Average {row['metric_name']:20}: {row['avg_val']:.4f}")
    print("---------------------------------------------------")
    
    return exp_id

def main():
    parser = argparse.ArgumentParser(description="End-to-End Clinical RAG Evaluation Engine")
    parser.add_argument("--db-path", default="src/experiments/evaluation_results.db", help="Path to SQLite DB")

    parser.add_argument("--retriever", choices=["vector", "lexical", "hybrid"], default="hybrid", help="Retriever strategy")
    parser.add_argument("--generator", default="mock", help="Inference generator model name")
    parser.add_argument("--chunk-size", type=int, default=800, help="Doc chunk char size")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Doc overlap char size")
    parser.add_argument("--chunking", type=lambda x: x.lower() in ('true', '1', 'yes'), default=True, help="Enable document chunking (True/False)")
    parser.add_argument("--baseline-exp", help="Baseline experiment ID to trigger regression gating check")
    args = parser.parse_args()

    # 1. Run Pipeline
    exp_id = run_evaluation_pipeline(args)
    
    # 2. Check Regression Gating if baseline ID is provided
    if args.baseline_exp:
        print(f"\n=== RUNNING REGRESSION TESTING GATE VS BASELINE: {args.baseline_exp} ===")
        try:
            report = evaluate_regression(args.db_path, exp_id, args.baseline_exp)
            print(f"Gate Status: {report['status']}")
            for metric, data in report["metrics"].items():
                if "delta" in data:
                    print(f"  {metric:20}: Baseline={data['baseline']:.4f}, Candidate={data['candidate']:.4f}, Delta={data['delta']:.4f} [{data['status']}]")
                else:
                    print(f"  {metric:20}: Baseline={data['baseline']:.4f}, Candidate={data['candidate']:.4f}, Ratio={data['ratio']:.2f}x [{data['status']}]")
            print("=================================================================\n")
            
            if report["status"] == "FAIL":
                print("[Gate] Failed regression verification. Exiting with error.")
                sys.exit(1)
            else:
                print("[Gate] Regression gate check passed.")
                sys.exit(0)
        except Exception as e:
            print(f"[Gate] Error performing regression gate check: {e}")
            sys.exit(2)
            
    sys.exit(0)

if __name__ == "__main__":
    main()
