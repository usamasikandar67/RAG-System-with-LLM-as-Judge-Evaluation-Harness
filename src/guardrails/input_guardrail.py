import os
import re
from typing import Dict, Any

class QueryGuardrail:
    """
    Evaluates incoming queries to ensure they are clinically relevant and safe to process.
    Blocks out-of-domain queries (e.g. recipes, coding, casual chat).
    """
    
    def __init__(self):
        # A simple heuristic approach for offline/fast blocking
        # In a production setting, this could be backed by an LLM classification call or specialized model.
        self.banned_keywords = [
            r"\b(recipe|bake|cook|cake|pizza|food)\b",
            r"\b(code|python|java|javascript|html|css)\b",
            r"\b(movie|music|game|sports)\b",
            r"\b(weather|politics)\b",
            r"\b(joke|funny)\b"
        ]

    def check(self, query: str) -> Dict[str, Any]:
        """
        Check if the query is relevant to the clinical domain.
        Returns {"is_relevant": bool, "reason": str}
        """
        query_lower = query.lower()
        
        # 1. Check for blatantly out-of-domain topics
        for pattern in self.banned_keywords:
            if re.search(pattern, query_lower):
                return {
                    "is_relevant": False,
                    "reason": f"Out of domain: detected non-medical topic."
                }
                
        # Intent Classification Prompt
        intent_prompt = f"""You are a strict intent classifier for a Clinical AI.
Classify the user's query into exactly one of the following intents:
- MEDICAL_QUERY: Questions asking for clinical guidelines, cancer info, treatments, or medical advice.
- WEATHER: Questions asking about temperature, climate, or weather forecasts.
- SPORTS: Questions asking about sports games or teams.
- FINANCE: Questions asking about stocks, investing, or finance.
- MOVIES: Questions asking about films or cinema.
- CODING: Questions asking for programming help or software engineering.
- RECIPES: Questions asking for cooking recipes or food.
- TRAVEL: Questions asking about flights, hotels, or travel.
- POLITICS: Questions asking about elections, government, or politics.
- SHOPPING: Questions asking about buying goods, e-commerce, or shopping.
- CASUAL_CHAT: General greetings, jokes, or casual conversation.
- OUT_OF_DOMAIN: Anything else not strictly related to clinical medicine.

Output a valid JSON object with the following schema:
{{
  "intent": "string",
  "confidence": float (0.0 to 1.0),
  "reason": "string (brief explanation)"
}}

Query: {query}"""

        def parse_and_validate(response_text: str) -> Dict[str, Any]:
            import json
            try:
                # Clean markdown formatting if present
                clean_text = response_text.replace("```json", "").replace("```", "").strip()
                result = json.loads(clean_text)
                intent = result.get("intent", "").upper()
                confidence = float(result.get("confidence", 0.0))
                reason = result.get("reason", "No reason provided")
                
                if intent != "MEDICAL_QUERY":
                    return {"is_relevant": False, "reason": f"Out of domain: LLM classified query intent as {intent} ({confidence:.2f}). Reason: {reason}"}
                if confidence < 0.8:
                    return {"is_relevant": False, "reason": f"Ambiguous query: Confidence ({confidence:.2f}) is below the 0.8 threshold. Reason: {reason}"}
                
                return {"is_relevant": True, "reason": "Passed LLM check", "confidence": confidence, "intent_details": result}
            except Exception as e:
                return {"is_relevant": False, "reason": f"Failed to parse LLM classification: {str(e)}"}

        # 2. LLM-based check if API key is present
        if os.getenv("OPENAI_API_KEY"):
            # pyrefly: ignore [missing-import]
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    completion = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": intent_prompt}],
                        response_format={"type": "json_object"},
                        max_tokens=150,
                        temperature=0
                    )
                    content = completion.choices[0].message.content
                    eval_result = parse_and_validate(content)
                    
                    if eval_result.get("is_relevant") or attempt == max_retries - 1:
                        return eval_result
                    # If parsing failed, loop and retry
                except Exception as e:
                    if attempt == max_retries - 1:
                        pass # Fallback to huggingface if API fails

        # 3. Try Free HuggingFace LLM-based check
        try:
            # pyrefly: ignore [missing-import]
            from huggingface_hub import InferenceClient
            client = InferenceClient(token=os.getenv("HF_TOKEN"))
            res = client.chat_completion(
                model="mistralai/Mistral-7B-Instruct-v0.2",
                messages=[{"role": "user", "content": intent_prompt}],
                max_tokens=150,
                temperature=0.1
            ).choices[0].message.content.strip()
            eval_result = parse_and_validate(res)
            if "Failed to parse" not in eval_result["reason"]:
                return eval_result
        except Exception:
            pass # Fallback to heuristic
            
        return {
            "is_relevant": True,
            "reason": "Query passed guardrail heuristics."
        }
