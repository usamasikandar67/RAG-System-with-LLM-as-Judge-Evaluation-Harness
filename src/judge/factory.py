# judge/factory.py

import os
import re
import json
from abc import ABC, abstractmethod
from typing import List, Dict

class BaseJudge(ABC):
    @abstractmethod
    def evaluate_response(self, question: str, response: str, ground_truth: str, contexts: List[str]) -> Dict[str, float]:
        """
        Grades response using retrieved contexts and ground truth.
        Returns score metrics: correctness, completeness, faithfulness, citation_accuracy.
        """
        pass

class MockJudge(BaseJudge):
    def evaluate_response(self, question: str, response: str, ground_truth: str, contexts: List[str]) -> Dict[str, float]:
        # 1. Citation Accuracy Check (checks for brackets like [cancer_doc_0.txt])
        has_bracket_citation = bool(re.search(r"\[[a-zA-Z0-9_\-\.]+\.txt\]", response))
        citation_score = 1.0 if has_bracket_citation else 0.0

        # 2. Correctness and Completeness (lexical overlap check)
        gt_tokens = set(re.findall(r"\w+", ground_truth.lower()))
        resp_tokens = set(re.findall(r"\w+", response.lower()))
        
        overlap_size = len(gt_tokens.intersection(resp_tokens))
        overlap_ratio = overlap_size / max(1, len(gt_tokens))
        
        correctness = min(1.0, overlap_ratio * 1.2) # scale slightly
        completeness = overlap_ratio

        # 3. Faithfulness (checks if facts in response exist in context)
        context_text = " ".join(contexts).lower()
        context_tokens = set(re.findall(r"\w+", context_text))
        
        # Stopwords to filter out basic syntax
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "to", "in", "on", "at", "for", "with", "of", "and", "or", "based", "according"}
        significant_resp_tokens = resp_tokens - stopwords
        
        if not significant_resp_tokens:
            faithfulness = 1.0
        else:
            unsupported_tokens = significant_resp_tokens - context_tokens
            # Discount numbers/citations that might be valid format edits
            unsupported_tokens = {t for t in unsupported_tokens if not t.isdigit() and not t.endswith(".txt")}
            unsupported_ratio = len(unsupported_tokens) / len(significant_resp_tokens)
            faithfulness = max(0.0, 1.0 - unsupported_ratio)

        return {
            "correctness": round(float(correctness), 2),
            "completeness": round(float(completeness), 2),
            "faithfulness": round(float(faithfulness), 2),
            "citation_accuracy": round(float(citation_score), 2)
        }

class HuggingFaceJudge(BaseJudge):
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        self.model_name = model_name
        # pyrefly: ignore [missing-import]
        from huggingface_hub import InferenceClient
        token = os.getenv("HF_TOKEN")
        self.client = InferenceClient(token=token)

    def evaluate_response(self, question: str, response: str, ground_truth: str, contexts: List[str]) -> Dict[str, float]:
        system_prompt = "You are a medical AI judge. Evaluate the user's response to the medical query based on the ground truth. Return ONLY a raw JSON string with numeric keys (0.0 to 1.0): correctness, completeness, faithfulness, citation_accuracy."
        user_prompt = f"Question: {question}\nResponse: {response}\nGround Truth: {ground_truth}\nContexts: {contexts}"
        
        prompt = f"<s>[INST] {system_prompt}\n\nUser: {user_prompt} [/INST]"
        try:
            res = self.client.text_generation(prompt, model=self.model_name, max_new_tokens=200).strip()
            
            # Find json block
            match = re.search(r"\{.*\}", res, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                return {
                    "correctness": float(data.get("correctness", 0.0)),
                    "completeness": float(data.get("completeness", 0.0)),
                    "faithfulness": float(data.get("faithfulness", 0.0)),
                    "citation_accuracy": float(data.get("citation_accuracy", 0.0))
                }
            else:
                raise ValueError("No JSON block found in response.")
        except Exception as e:
            print(f"[Judge] HF Evaluation failed, falling back to Mock: {e}")
            fallback = MockJudge()
            return fallback.evaluate_response(question, response, ground_truth, contexts)

def get_judge_model(model_name: str) -> BaseJudge:
    if model_name.startswith("huggingface"):
        actual_model = model_name.replace("huggingface:", "").strip() or "mistralai/Mistral-7B-Instruct-v0.2"
        print(f"[Judge] Using HuggingFace API with model: {actual_model}")
        return HuggingFaceJudge(model_name=actual_model)
    elif model_name == "mock":
        print("[Judge] Using MockJudge (offline mode or default)")
        return MockJudge()
    else:
        print(f"[Judge] Unknown model {model_name}. Falling back to MockJudge.")
        return MockJudge()
