import os
import csv
import json
import hashlib
from typing import List, Dict, Any

CSV_PATH = "cancer/CancerQA.csv"
KB_DIR = "datasets/knowledge_base"
GOLDEN_PATH = "datasets/golden_dataset.json"
MANIFEST_PATH = "datasets/manifest.json"

def clean_text(text: str) -> str:
    # Clean whitespace and strip leading/trailing spaces
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join([line for line in lines if line]).strip()

def process_cancer_data(limit: int = 50):
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Source file {CSV_PATH} not found.")

    os.makedirs(KB_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)

    print(f"[Ingestion] Reading clinical guidelines from {CSV_PATH}...")
    
    golden_dataset = []
    manifest_entries = {}

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        # Standard CSV reader handles multi-line fields correctly
        reader = csv.reader(f)
        
        # Read header
        header = next(reader)
        # Expected: Question, Answer, topic, split
        
        count = 0
        for idx, row in enumerate(reader):
            if count >= limit:
                break
                
            if len(row) < 2:
                continue
                
            question = clean_text(row[0])
            answer = clean_text(row[1])
            topic = clean_text(row[2]) if len(row) > 2 else "Cancer QA"
            split = clean_text(row[3]) if len(row) > 3 else "train"

            # Skip empty entries
            if not question or not answer:
                continue

            doc_filename = f"cancer_doc_{count}.txt"
            doc_path = os.path.join(KB_DIR, doc_filename)

            # Write document
            with open(doc_path, "w", encoding="utf-8") as doc_file:
                doc_file.write(answer)

            # Compute hash
            hasher = hashlib.sha256()
            hasher.update(answer.encode("utf-8"))
            doc_hash = hasher.hexdigest()

            # Register manifest entry
            manifest_entries[doc_filename] = {
                "topic": topic,
                "split": split,
                "hash": doc_hash,
                "char_length": len(answer)
            }

            # Register golden dataset test case
            golden_dataset.append({
                "test_id": f"TC-{count+1:03d}",
                "question": question,
                "ground_truth_answer": answer,
                "expected_documents": [doc_filename],
                "clinical_category": "Oncology",
                "difficulty": "Medium",
                "metadata": {
                    "topic": topic,
                    "split": split
                }
            })

            count += 1

    # Write manifest.json
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f_manifest:
        json.dump(manifest_entries, f_manifest, indent=2)

    # Write golden_dataset.json
    with open(GOLDEN_PATH, "w", encoding="utf-8") as f_golden:
        json.dump(golden_dataset, f_golden, indent=2)

    print(f"[Ingestion] Completed. Generated:")
    print(f"  - {count} document files in {KB_DIR}/")
    print(f"  - {MANIFEST_PATH} manifest registry")
    print(f"  - {GOLDEN_PATH} golden evaluation dataset")

if __name__ == "__main__":
    process_cancer_data(50)
