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

class OpenAIJudge(BaseJudge):
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        # pyrefly: ignore [missing-import]
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIJudge")
        self.client = OpenAI(api_key=api_key)

    def evaluate_response(self, question: str, response: str, ground_truth: str, contexts: List[str]) -> Dict[str, float]:
        eval_prompt = f"""You are a medical peer-review judge. Grade the generated response against the ground truth answer and retrieved contexts.

Question: {question}
Generated Response: {response}
Ground Truth: {ground_truth}
Retrieved Contexts: {" | ".join(contexts)}

Grade the following four criteria from 0.0 (Worst) to 1.0 (Best):
1. correctness: Is the answer clinically correct compared to ground truth?
2. completeness: Does the answer cover all key clinical points in the ground truth?
3. faithfulness: Are all claims in the response fully grounded in the retrieved contexts without hallucination?
4. citation_accuracy: Does the response correctly cite source guidelines using inline [filename.txt] labels?

Return ONLY a valid JSON object matching this schema:
{{
  "correctness": float,
  "completeness": float,
  "faithfulness": float,
  "citation_accuracy": float
}}
"""
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": eval_prompt}],
                response_format={"type": "json_object"}
            )
            data = json.loads(completion.choices[0].message.content)
            return {
                "correctness": float(data.get("correctness", 0.0)),
                "completeness": float(data.get("completeness", 0.0)),
                "faithfulness": float(data.get("faithfulness", 0.0)),
                "citation_accuracy": float(data.get("citation_accuracy", 0.0))
            }
        except Exception as e:
            print(f"[Judge] OpenAI Evaluation failed, falling back to Mock: {e}")
            fallback = MockJudge()
            return fallback.evaluate_response(question, response, ground_truth, contexts)

def get_judge_model(model_name: str) -> BaseJudge:
    if model_name == "mock" or not os.getenv("OPENAI_API_KEY"):
        print("[Judge] Using MockJudge (offline mode or key missing)")
        return MockJudge()
    else:
        print(f"[Judge] Using OpenAIJudge with model: {model_name}")
        return OpenAIJudge(model_name=model_name)
