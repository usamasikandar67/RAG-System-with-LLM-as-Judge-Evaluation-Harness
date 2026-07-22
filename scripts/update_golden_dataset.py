import os
import json
import csv
import random

def safe_name(focus_area):
    name = "".join(c if c.isalnum() else "_" for c in focus_area).strip("_")
    return name if name else "General"

def main():
    csv_file = "/Users/d-23-10623/Desktop/LLM-as-Judge Evaluation Harness/medquad 2.csv"
    golden_dataset_path = "/Users/d-23-10623/Desktop/LLM-as-Judge Evaluation Harness/datasets/golden_dataset.json"
    
    print(f"Reading {csv_file}...")
    qa_pairs = []
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get("question", "").strip()
            a = row.get("answer", "").strip()
            focus = row.get("focus_area", "General").strip()
            if q and a:
                qa_pairs.append({
                    "question": q,
                    "answer": a,
                    "focus_area": focus
                })
                
    print(f"Loaded {len(qa_pairs)} QA pairs.")
    
    # Process all samples and shuffle them
    random.seed(42) # For reproducibility
    random.shuffle(qa_pairs)
    selected_samples = qa_pairs
    
    golden_dataset = []
    total_samples = len(selected_samples)
    train_cutoff = int(total_samples * 0.8)
    
    for i, sample in enumerate(selected_samples, 1):
        focus = sample["focus_area"]
        doc_name = f"{safe_name(focus)}.txt"
        
        # Calculate difficulty heuristically by answer length
        word_count = len(sample["answer"].split())
        if word_count < 20:
            diff = "Easy"
        elif word_count < 100:
            diff = "Medium"
        else:
            diff = "Hard"
            
        test_case = {
            "test_id": f"TC-{i:05d}",
            "question": sample["question"],
            "ground_truth_answer": sample["answer"],
            "expected_documents": [doc_name],
            "clinical_category": "General Medicine",
            "difficulty": diff,
            "metadata": {
                "topic": focus,
                "split": "train" if i <= train_cutoff else "test"
            }
        }
        golden_dataset.append(test_case)
        
    print(f"Writing {len(golden_dataset)} test cases to {golden_dataset_path}...")
    with open(golden_dataset_path, "w", encoding="utf-8") as out:
        json.dump(golden_dataset, out, indent=2)
        
    print("Successfully updated the golden dataset!")

if __name__ == "__main__":
    main()
