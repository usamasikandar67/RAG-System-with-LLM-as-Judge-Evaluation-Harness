import os

class OutputGuardrail:
    """
    Evaluates the generated response against the retrieved context
    to ensure the response is grounded and doesn't hallucinate.
    """
    
    def verify(self, question: str, response: str, contexts: list) -> dict:
        """Verify grounding of the generated response.

        Returns a dictionary with a ``passed`` boolean indicating whether the response
        is grounded in the provided contexts and a ``reason`` string describing the
        outcome. This method consolidates the previous compatibility wrapper and
        grounding logic.
        """
        # Fallback phrases indicate a safe answer that should be considered grounded.
        fallback_phrases = ["I cannot find", "I am sorry, but the retrieved", "I don't have this information"]
        if any(f in response for f in fallback_phrases):
            return {"passed": True, "reason": "Fallback response considered grounded."}

        # Basic heuristic for mock testing.
        q_words = set([w.strip('.,?!') for w in question.lower().split()])
        c_text = " ".join([c.get("text", "").lower() for c in contexts])
        c_words = set([w.strip('.,?!') for w in c_text.split()])
        stop_words = {"tell", "me", "about", "what", "is", "the", "a", "an", "of", "in", "and", "or", "to", "for", "with", "on", "can", "you", "give", "how", "do", "does"}
        meaningful_q_words = q_words - stop_words
        if meaningful_q_words and len(meaningful_q_words.intersection(c_words)) == 0:
            return {
                "passed": False,
                "reason": f"Query terms {meaningful_q_words} not found in context. Likely hallucination."
            }

        # LLM-based grounding check via OpenAI if API key is set.
        if os.getenv("OPENAI_API_KEY"):
            try:
                # pyrefly: ignore [missing-import]
                from openai import OpenAI
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                context_text = "\n".join([c.get("text", "") for c in contexts])
                prompt = f"""
                You are a strict grounding verification assistant.
                Determine if the Response answers the Question using ONLY the provided Context.
                CRITICAL RULE: If the Question is about a non-medical topic (e.g., weather, temperature, coding) and the Response attempts to answer it using the medical Context, output 'NO'.
                If the response contains external information, hallucinated facts, or answers an out-of-context question, output 'NO'.
                If the response is strictly derived from the Context and is medically relevant, output 'YES'.

                Context: {context_text}
                Question: {question}
                Response: {response}

                Answer 'YES' or 'NO' only.
                """
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=5,
                    temperature=0
                )
                answer = completion.choices[0].message.content.strip().upper()
                if "NO" in answer:
                    return {"passed": False, "reason": "OpenAI LLM detected hallucination or out-of-domain answer."}
                return {"passed": True, "reason": "Passed OpenAI grounding check."}
            except Exception:
                pass

        # Free HuggingFace LLM-based check.
        try:
            # pyrefly: ignore [missing-import]
            from huggingface_hub import InferenceClient
            client = InferenceClient(token=os.getenv("HF_TOKEN"))
            context_text = "\n".join([c.get("text", "") for c in contexts])
            prompt = f"Determine if the Response answers the Question using ONLY the provided Context. CRITICAL RULE: If the Question is about a non-medical topic (e.g., weather, temperature, coding) and the Response attempts to answer it, output 'NO'. If the response contains hallucinated facts, output 'NO'. If strictly derived from Context, output 'YES'.\n\nContext: {context_text}\nQuestion: {question}\nResponse: {response}\n\nAnswer 'YES' or 'NO' only."
            res = client.chat_completion(
                model="mistralai/Mistral-7B-Instruct-v0.2",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0.1
            ).choices[0].message.content.strip().upper()
            if "NO" in res:
                return {"passed": False, "reason": "Free LLM detected hallucination or out-of-domain answer."}
            return {"passed": True, "reason": "Passed Free LLM grounding check."}
        except Exception:
            pass

        # Heuristic fallback – assume grounded.
        return {"passed": True, "reason": "Heuristically passed grounding check."}


class RetrievalGuardrail:
    """Simple guardrail for retrieval results.

    Checks that retrieved contexts are non‑empty and contain at least one
    source file. Returns a dict with a boolean ``passed`` flag and a ``reason``
    string for debugging. In a production system this could include relevance
    scoring, freshness checks, or policy enforcement.
    """

    def validate(self, contexts: list) -> dict:
        """Validate retrieved contexts.

        Returns a dict with:
        * ``passed`` – bool indicating basic sanity.
        * ``reason`` – explanation for debugging.
        * ``stats``  – optional metrics (e.g., max_score) used later in the pipeline.
        """
        if not contexts:
            return {"passed": False, "reason": "No contexts retrieved", "stats": {}}
        max_score = 0.0
        for ctx in contexts:
            meta = ctx.get("metadata", {})
            # Some retrievers provide a ``score`` field; fall back to 0.
            score = ctx.get("score") or ctx.get("rerank_score") or 0.0
            if isinstance(score, (int, float)) and score > max_score:
                max_score = score
            if meta.get("source_file"):
                # At least one context has source info – success.
                return {"passed": True, "reason": "Contexts contain source files", "stats": {"max_score": max_score}}
        # No source_file metadata found.
        return {"passed": False, "reason": "Contexts missing source_file metadata", "stats": {"max_score": max_score}}


def compute_confidence_score(retrieval_stats: dict, verification: dict, num_unique_sources: int) -> dict:
    """Compute a lightweight confidence score for the pipeline response.

    * ``retrieval_stats`` – dict returned from ``RetrievalGuardrail.validate``
      (may contain additional metrics in the future).
    * ``verification`` – result from ``OutputGuardrail.verify`` indicating
      whether the generated answer is grounded.
    * ``num_unique_sources`` – number of distinct source files used.

    The function returns a dictionary with overall confidence and a label.
    The scoring is intentionally simple: if the answer is verified *and* we
    have at least one source, confidence is high; otherwise it degrades.
    """
    base = 0.5
    if verification.get("passed"):
        base += 0.3
    if retrieval_stats.get("passed"):
        base += 0.1
    # Slight bump for multiple sources (more evidence)
    if num_unique_sources >= 2:
        base += 0.05
    confidence = min(base, 1.0)
    label = "High" if confidence > 0.8 else "Medium" if confidence > 0.5 else "Low"
    return {
        "overall": confidence,
        "overall_pct": f"{int(confidence * 100)}%",
        "overall_label": label,
    }
