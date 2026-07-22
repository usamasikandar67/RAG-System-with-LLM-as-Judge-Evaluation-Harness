import json
import os
# pyrefly: ignore [missing-import]
import google.generativeai as genai
# pyrefly: ignore [missing-import]
from openai import OpenAI
# pyrefly: ignore [missing-import]
import chromadb
# pyrefly: ignore [missing-import]
from chromadb.config import Settings
from typing import List, Dict, Any

# ==========================================
# Configuration
# ==========================================
EVALUATION_JSON_PATH = "evaluation_dataset.json" # Update with your actual path
CHROMA_DB_PATH = "./chroma_db"

def init_clients():
    """Initializes LLM clients."""
    # Ensure API keys are present
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY environment variable is missing.")
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is missing.")

    # Configure Gemini
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    
    # Configure OpenAI
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    return openai_client

def generate_question_variations(original_question: str) -> List[str]:
    """Uses Gemini to generate 3-4 variations of the question."""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        You are a search intent specialist. Given the following clinical question, generate 3 to 4 alternative variations representing different ways a clinician might phrase this query.
        Output ONLY the questions, separated by newlines, with no bullet points, numbering, or introductory text.
        
        Original Question: {original_question}
        """
        response = model.generate_content(prompt)
        variations = [q.strip() for q in response.text.split('\n') if q.strip()]
        
        # Ensure we return a list including the original question
        return [original_question] + variations[:4]
    except Exception as e:
        print(f"Error generating variations for '{original_question}': {e}")
        return [original_question]

def get_embeddings(texts: List[str], openai_client: OpenAI) -> List[List[float]]:
    """Generates text-embedding-3-large embeddings for a list of texts."""
    try:
        response = openai_client.embeddings.create(
            input=texts,
            model="text-embedding-3-large"
        )
        return [data.embedding for data in response.data]
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return []

def main():
    print("Starting Q2Q Ingestion Process...")
    
    # Initialize Clients
    try:
        openai_client = init_clients()
    except ValueError as e:
        print(f"Initialization Error: {e}")
        return

    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_or_create_collection(
        name="qa_index",
        metadata={"hnsw:space": "cosine"} # Use cosine similarity
    )

    # Load Dataset
    if not os.path.exists(EVALUATION_JSON_PATH):
        print(f"Error: Dataset {EVALUATION_JSON_PATH} not found.")
        print("Please create an 'evaluation_dataset.json' file to run this script.")
        return

    with open(EVALUATION_JSON_PATH, "r") as f:
        try:
            dataset = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: {EVALUATION_JSON_PATH} contains invalid JSON.")
            return

    # Process each entry
    for i, entry in enumerate(dataset):
        # 1. Validation Block
        try:
            question = entry["question"]
            ground_truth = entry["ground_truth_answer"]
            expected_docs = entry["expected_documents"]
            
            if not question or not ground_truth or expected_docs is None:
                raise ValueError("Null values detected.")
        except (KeyError, ValueError) as e:
            print(f"Skipping malformed entry at index {i}: {e}")
            continue

        print(f"\nProcessing Entry {i+1}: {question[:50]}...")

        # 2. Synthesize Questions (Q2Q Enrichment)
        variations = generate_question_variations(question)
        print(f"Generated {len(variations)} variations.")

        # 3. Embedding
        embeddings = get_embeddings(variations, openai_client)
        
        if not embeddings or len(embeddings) != len(variations):
            print("Embedding failed. Skipping entry.")
            continue

        # 4. Database Ingestion
        ids = [f"doc_{i}_var_{j}" for j in range(len(variations))]
        metadatas = [
            {
                "ground_truth_answer": ground_truth,
                "expected_documents": json.dumps(expected_docs),
                "original_question": question,
                "is_variation": (j > 0)
            }
            for j in range(len(variations))
        ]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=variations, # We store the variation text as the document
            metadatas=metadatas
        )
        print("Successfully ingested into ChromaDB.")

    print("\nIngestion Complete!")

if __name__ == "__main__":
    main()
