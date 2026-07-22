# src/experiments/db.py

import os
import sqlite3
import json
import time
import uuid
from typing import Dict, Any, List, Optional

def init_db(db_path: str = "src/experiments/evaluation_results.db") -> None:
    # Ensure parent directories exist
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON;")

    # Drop old tables if this is the first migration step
    cursor.execute("DROP TABLE IF EXISTS eval_runs")

    # 1. Documents
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        doc_id TEXT PRIMARY KEY,
        source TEXT,
        title TEXT,
        url TEXT,
        doc_type TEXT,
        ingested_at TEXT,
        metadata TEXT -- JSONB equivalent
    )
    """)

    # 2. Chunks
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id TEXT PRIMARY KEY,
        doc_id TEXT NOT NULL,
        chunk_index INTEGER,
        content TEXT NOT NULL,
        token_count INTEGER,
        metadata TEXT, -- JSONB equivalent
        FOREIGN KEY (doc_id) REFERENCES documents (doc_id)
    )
    """)

    # 3. Embeddings
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS embeddings (
        embedding_id TEXT PRIMARY KEY,
        chunk_id TEXT NOT NULL,
        vector_ref TEXT,
        model_name TEXT,
        dims INTEGER,
        created_at TEXT,
        FOREIGN KEY (chunk_id) REFERENCES chunks (chunk_id)
    )
    """)

    # 4. Queries
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS queries (
        query_id TEXT PRIMARY KEY,
        session_id TEXT,
        user_id TEXT,
        query_text TEXT NOT NULL,
        created_at TEXT
    )
    """)

    # 5. Retrieval Results
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS retrieval_results (
        retrieval_id TEXT PRIMARY KEY,
        query_id TEXT NOT NULL,
        chunk_id TEXT NOT NULL,
        pipeline TEXT,
        rank INTEGER,
        score REAL,
        FOREIGN KEY (query_id) REFERENCES queries (query_id),
        FOREIGN KEY (chunk_id) REFERENCES chunks (chunk_id)
    )
    """)

    # 6. Generations
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS generations (
        generation_id TEXT PRIMARY KEY,
        query_id TEXT NOT NULL,
        prompt_text TEXT,
        llm_model TEXT,
        response_text TEXT,
        latency_ms REAL,
        created_at TEXT,
        FOREIGN KEY (query_id) REFERENCES queries (query_id)
    )
    """)

    # 7. Citations
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS citations (
        citation_id TEXT PRIMARY KEY,
        generation_id TEXT NOT NULL,
        chunk_id TEXT NOT NULL,
        citation_order INTEGER,
        FOREIGN KEY (generation_id) REFERENCES generations (generation_id),
        FOREIGN KEY (chunk_id) REFERENCES chunks (chunk_id)
    )
    """)

    # 8. Golden Dataset
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS golden_dataset (
        golden_id TEXT PRIMARY KEY,
        query_text TEXT NOT NULL,
        expected_answer TEXT,
        expected_chunk_ids TEXT, -- CSV list of chunk IDs
        created_at TEXT
    )
    """)

    # 9. Evaluations
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS evaluations (
        eval_id TEXT PRIMARY KEY,
        generation_id TEXT NOT NULL,
        golden_id TEXT,
        metric_name TEXT NOT NULL,
        metric_value REAL NOT NULL,
        judge_model TEXT,
        evaluated_at TEXT,
        FOREIGN KEY (generation_id) REFERENCES generations (generation_id),
        FOREIGN KEY (golden_id) REFERENCES golden_dataset (golden_id)
    )
    """)

    # 10. Experiments
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS experiments (
        experiment_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        pipeline TEXT,
        config TEXT, -- JSONB equivalent
        created_at TEXT
    )
    """)

    # 11. Experiment Runs
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS experiment_runs (
        run_id TEXT PRIMARY KEY,
        experiment_id TEXT NOT NULL,
        generation_id TEXT NOT NULL,
        eval_id TEXT,
        FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id),
        FOREIGN KEY (generation_id) REFERENCES generations (generation_id),
        FOREIGN KEY (eval_id) REFERENCES evaluations (eval_id)
    )
    """)


    # Backwards compatibility tables for Chatbot
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chatbot_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        contexts_json TEXT,
        timestamp TEXT NOT NULL
    )
    """)

    conn.commit()
    conn.close()

# --- HELPER FUNCTIONS ---

def get_db_connection(db_path: str = "src/experiments/evaluation_results.db") -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def create_experiment(db_path: str, name: str, pipeline: Any = "hybrid", config: Optional[Dict[str, Any]] = None) -> str:
    if config is None and isinstance(pipeline, dict):
        config = pipeline
        pipeline = "hybrid"
    if config is None:
        config = {}
    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    exp_id = f"exp_{int(time.time() * 1000)}"
    created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    
    cursor.execute(
        "INSERT INTO experiments (experiment_id, name, pipeline, config, created_at) VALUES (?, ?, ?, ?, ?)",
        (exp_id, name, str(pipeline), json.dumps(config), created_at)
    )
    conn.commit()
    conn.close()
    return exp_id

def log_query(db_path: str, query_text: str, session_id: str = "eval_session") -> str:
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    query_id = f"q_{uuid.uuid4().hex[:8]}"
    created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    cursor.execute("""
    INSERT INTO queries (query_id, session_id, user_id, query_text, created_at)
    VALUES (?, ?, ?, ?, ?)
    """, (query_id, session_id, "system", query_text, created_at))
    conn.commit()
    conn.close()
    return query_id

def log_document(db_path: str, doc_id: str, source: str, title: str, doc_type: str, metadata: Dict[str, Any]) -> None:
    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    ingested_at = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    try:
        cursor.execute("""
        INSERT OR IGNORE INTO documents (doc_id, source, title, url, doc_type, ingested_at, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (doc_id, source, title, "", doc_type, ingested_at, json.dumps(metadata)))
        conn.commit()
    except Exception as e:
        print(f"Failed to log document: {e}")
    finally:
        conn.close()

def log_chunk(db_path: str, chunk_id: str, doc_id: str, chunk_index: int, content: str, token_count: int, metadata: Dict[str, Any]) -> None:
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("""
        INSERT OR IGNORE INTO chunks (chunk_id, doc_id, chunk_index, content, token_count, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (chunk_id, doc_id, chunk_index, content, token_count, json.dumps(metadata)))
        conn.commit()
    except Exception as e:
        print(f"Failed to log chunk: {e}")
    finally:
        conn.close()

def log_retrieval(db_path: str, query_id: str, chunk_id: str, rank: int, score: float, pipeline: str = "hybrid") -> str:
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    retrieval_id = f"ret_{uuid.uuid4().hex[:8]}"
    cursor.execute("""
    INSERT INTO retrieval_results (retrieval_id, query_id, chunk_id, pipeline, rank, score)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (retrieval_id, query_id, chunk_id, pipeline, rank, score))
    conn.commit()
    conn.close()
    return retrieval_id

def log_generation(db_path: str, query_id: str, prompt_text: str, llm_model: str, response_text: str, latency_ms: float) -> str:
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    gen_id = f"gen_{uuid.uuid4().hex[:8]}"
    created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    cursor.execute("""
    INSERT INTO generations (generation_id, query_id, prompt_text, llm_model, response_text, latency_ms, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (gen_id, query_id, prompt_text, llm_model, response_text, latency_ms, created_at))
    conn.commit()
    conn.close()
    return gen_id

def log_evaluation(db_path: str, generation_id: str, golden_id: Optional[str], metric_name: str, metric_value: float, judge_model: str) -> str:
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    eval_id = f"eval_{uuid.uuid4().hex[:8]}"
    evaluated_at = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    cursor.execute("""
    INSERT INTO evaluations (eval_id, generation_id, golden_id, metric_name, metric_value, judge_model, evaluated_at)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (eval_id, generation_id, golden_id, metric_name, metric_value, judge_model, evaluated_at))
    conn.commit()
    conn.close()
    return eval_id

def log_experiment_run(db_path: str, experiment_id: str, generation_id: str, eval_id: str) -> str:
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    cursor.execute("""
    INSERT INTO experiment_runs (run_id, experiment_id, generation_id, eval_id)
    VALUES (?, ?, ?, ?)
    """, (run_id, experiment_id, generation_id, eval_id))
    conn.commit()
    conn.close()
    return run_id

def log_run(db_path: str, experiment_id: str, run_report: Dict[str, Any], query_details: Dict[str, Any]) -> str:
    init_db(db_path)
    query_id = log_query(db_path, query_details.get("question", ""))
    gen_id = log_generation(
        db_path,
        query_id,
        query_details.get("prompt", "prompt"),
        query_details.get("generator", "mock"),
        query_details.get("response_text", ""),
        query_details.get("latency", run_report.get("latency", 0.0))
    )
    last_run_id = ""
    for metric_name in ["hit_1", "hit_3", "hit_5", "reciprocal_rank", "average_precision", "correctness", "completeness", "faithfulness", "citation_accuracy", "clinical_utility"]:
        if metric_name in run_report:
            val = float(run_report[metric_name])
            eval_id = log_evaluation(db_path, gen_id, None, metric_name, val, "judge_model")
            last_run_id = log_experiment_run(db_path, experiment_id, gen_id, eval_id)
    return last_run_id

def get_runs_for_experiment(db_path: str, experiment_id: str) -> List[Dict[str, Any]]:
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT q.query_text, g.generation_id, g.latency_ms, e.metric_name, e.metric_value
        FROM experiment_runs r
        JOIN generations g ON r.generation_id = g.generation_id
        JOIN queries q ON g.query_id = q.query_id
        JOIN evaluations e ON r.eval_id = e.eval_id
        WHERE r.experiment_id = ?
    """, (experiment_id,))
    rows = cursor.fetchall()
    conn.close()
    
    runs_map: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        gen_id = r["generation_id"]
        if gen_id not in runs_map:
            runs_map[gen_id] = {
                "test_id": "TC-001",
                "question": r["query_text"],
                "latency": r["latency_ms"]
            }
        runs_map[gen_id][r["metric_name"]] = r["metric_value"]
    return list(runs_map.values())

# Chatbot legacy helpers
def save_chat_message(db_path: str, session_id: str, role: str, content: str, contexts: List[Dict[str, Any]]) -> None:
    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    contexts_json = json.dumps(contexts) if contexts else None
    cursor.execute("""
    INSERT INTO chatbot_history (session_id, role, content, contexts_json, timestamp)
    VALUES (?, ?, ?, ?, ?)
    """, (session_id, role, content, contexts_json, timestamp))
    conn.commit()
    conn.close()

def load_chat_history(db_path: str, session_id: str) -> List[Dict[str, Any]]:
    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT role, content, contexts_json, timestamp FROM chatbot_history WHERE session_id = ? ORDER BY id ASC", (session_id,))
    rows = cursor.fetchall()
    conn.close()
    history = []
    for row in rows:
        history.append({
            "role": row["role"],
            "content": row["content"],
            "timestamp": row["timestamp"],
            "contexts": json.loads(row["contexts_json"]) if row["contexts_json"] else []
        })
    return history
