import os
import csv
import glob
from collections import defaultdict

def main():
    csv_file = "/Users/d-23-10623/Desktop/LLM-as-Judge Evaluation Harness/medquad 2.csv"
    kb_dir = "/Users/d-23-10623/Desktop/LLM-as-Judge Evaluation Harness/datasets/knowledge_base"
    
    # 1. Clear existing cancer_doc files
    old_files = glob.glob(os.path.join(kb_dir, "cancer_doc_*.txt"))
    for f in old_files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"Warning: Could not remove {f} - {e}")
    print(f"Removed {len(old_files)} old cancer documents.")
    
    if not os.path.exists(kb_dir):
        os.makedirs(kb_dir)
        
    print(f"Reading {csv_file}...")
    focus_areas = defaultdict(list)
    
    # Read the CSV
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                q = row.get("question", "").strip()
                a = row.get("answer", "").strip()
                focus = row.get("focus_area", "General").strip()
                if not focus:
                    focus = "General"
                if q and a:
                    focus_areas[focus].append(f"Q: {q}\nA: {a}")
                    count += 1
            print(f"Read {count} QA pairs.")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
                
    print(f"Found {len(focus_areas)} unique focus areas. Writing to {kb_dir}...")
    
    # Write out files
    for focus, qas in focus_areas.items():
        # Sanitize filename
        safe_name = "".join(c if c.isalnum() else "_" for c in focus).strip("_")
        if not safe_name:
            safe_name = "General"
            
        file_path = os.path.join(kb_dir, f"{safe_name}.txt")
        try:
            with open(file_path, "w", encoding="utf-8") as out:
                out.write(f"Topic: {focus}\n\n")
                out.write("\n\n---\n\n".join(qas))
        except Exception as e:
            print(f"Error writing to {file_path}: {e}")
            
    print(f"Successfully wrote {len(focus_areas)} documents to {kb_dir}")

if __name__ == "__main__":
    main()
