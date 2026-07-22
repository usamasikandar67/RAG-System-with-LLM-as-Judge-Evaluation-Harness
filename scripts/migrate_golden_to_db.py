import json
import sqlite3
import uuid
import time
import os

def migrate_golden_to_db(db_path="src/experiments/evaluation_results.db", json_path="datasets/golden_dataset.json"):
    print(f"Migrating {json_path} to {db_path}...")
    
    if not os.path.exists(json_path):
        print(f"Error: {json_path} does not exist.")
        return
        
    with open(json_path, "r", encoding="utf-8") as f:
        cases = json.load(f)
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Ensure golden_dataset table exists
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS golden_dataset (
        golden_id TEXT PRIMARY KEY,
        query_text TEXT NOT NULL,
        expected_answer TEXT,
        expected_chunk_ids TEXT,
        created_at TEXT
    )
    """)
    
    count = 0
    for case in cases:
        golden_id = f"gold_{uuid.uuid4().hex[:8]}"
        query_text = case.get("question", "")
        expected_answer = case.get("ground_truth", "")
        
        # In a real scenario, we might map these to real chunk IDs from the FAISS ingestion, 
        # but for this script we just store them as CSV
        expected_chunks = ",".join(case.get("expected_sources", []))
        
        created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        
        try:
            cursor.execute("""
            INSERT INTO golden_dataset (golden_id, query_text, expected_answer, expected_chunk_ids, created_at)
            VALUES (?, ?, ?, ?, ?)
            """, (golden_id, query_text, expected_answer, expected_chunks, created_at))
            count += 1
        except Exception as e:
            print(f"Failed to insert golden case: {e}")
            
    conn.commit()
    conn.close()
    print(f"Successfully migrated {count} cases to golden_dataset table.")

if __name__ == "__main__":
    migrate_golden_to_db()
