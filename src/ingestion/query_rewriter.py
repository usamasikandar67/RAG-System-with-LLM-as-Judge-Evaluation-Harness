import os
import json
from typing import List, Dict, Any

class QueryRewriter:
    """
    Rewrites conversational queries into standalone queries using chat history.
    This resolves pronouns and implied context (e.g. "What about stage 2?")
    so the retrieval engine has full context.
    """
    
    def __init__(self):
        self.rewrite_prompt = """You are a clinical query resolver.
Given a conversation history and a new user query, rewrite the user query to be a standalone question that contains all necessary context for a medical database search.
If the query is already standalone, return it as-is.
Do not answer the query. Just rewrite it.

Output a valid JSON object with the following schema:
{{
  "rewritten_query": "string"
}}

Conversation History:
{history_text}

New User Query: {query}"""

    def rewrite(self, query: str, chat_history: List[Dict[str, str]] = None) -> str:
        if not chat_history:
            return query
            
        # Only take the last 3 exchanges (6 messages) to avoid overflowing context
        recent_history = chat_history[-6:]
        history_text = "\n".join([f"{entry.get('role', 'user').capitalize()}: {entry.get('content', '')}" for entry in recent_history])
        
        prompt = self.rewrite_prompt.format(history_text=history_text, query=query)
        
        if os.getenv("OPENAI_API_KEY"):
            # pyrefly: ignore [missing-import]
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            try:
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    max_tokens=150,
                    temperature=0
                )
                
                content = completion.choices[0].message.content
                clean_text = content.replace("```json", "").replace("```", "").strip()
                result = json.loads(clean_text)
                return result.get("rewritten_query", query)
            except Exception as e:
                import logging
                logging.warning(f"OpenAI rewriting failed: {e}")
                pass # Fallback to HF or original
                
        # Fallback to HF or original query
        try:
            # pyrefly: ignore [missing-import]
            from huggingface_hub import InferenceClient
            client = InferenceClient(token=os.getenv("HF_TOKEN"))
            res = client.chat_completion(
                model="mistralai/Mistral-7B-Instruct-v0.2",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.1
            )
            clean_text = res.choices[0].message.content.replace("```json", "").replace("```", "").strip()
            result = json.loads(clean_text)
            import logging
            logging.info(f"[QueryRewriter] HF rewriting output: {result.get('rewritten_query', query)}")
            return result.get("rewritten_query", query)
        except Exception as e:
            import logging
            logging.warning(f"HF rewriting failed: {e}. Falling back to original query.")
            return query
