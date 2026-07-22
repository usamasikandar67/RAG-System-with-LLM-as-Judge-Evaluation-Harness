# src/ingestion/pipeline.py
"""
pipeline.py — Task 20: Full Production Pipeline

Architecture:
  User Query
    → Intent Guardrail          (Task 1)
    → PHI Detection + Masking   (Task 14)
    → Query Rewriter            (Task 3)
    → Medical NER               (Task 4)
    → ICD Mapping               (Task 5)
    → Hybrid Retrieval + RRF    (Task 6)
    → Metadata Filtering        (Task 7)
    → Cross-Encoder Reranking   (Task 6)
    → Retrieval Guardrail       (Task 2)
    → PromptBuilder             (Tasks 9, 13)
    → LLM Generation            
    → Grounding Check           (Task 10)
    → Hallucination Detection   (Task 11)
    → Confidence Scoring        (Task 12)
    → Latency Metrics           (Task 16)
    → Cost Tracking             (Task 17)
    → Langfuse Logging          (Task 15)
"""

import os
import time
import json
import logging
from typing import Dict, Any, List, Optional

# pyrefly: ignore [missing-import]
from ingestion.prompt_templates import CLINICAL_SYSTEM_PROMPT, PromptBuilder
# pyrefly: ignore [missing-import]
from ingestion.query_rewriter import QueryRewriter
from ner.ner_pipeline import extract_clinical_entities
from icd_mapping.icd_resolver import resolve_icd_code
from reranking.reranker import SimpleCrossEncoderReranker
from classical_ai.classifier import ClinicalTextClassifier
from phi.phi_detector import PHIDetector
from pih.pih_detector import PIHRiskScorer
from complaint.complaint_extractor import ComplaintExtractor
from services.service_tagger import ServiceTagger
from langfuse.tracer import LangfuseTracer
from guardrails.input_guardrail import QueryGuardrail
from guardrails.output_guardrail import OutputGuardrail, RetrievalGuardrail, compute_confidence_score

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self, retriever=None, generator_model: str = "mock"):
        """Initialize the RAG pipeline.

        Args:
            retriever: An object with a `retrieve(query, k, filter_metadata)` method.
                If None, a dummy retriever returning empty contexts is used.
            generator_model: Identifier for the LLM model (mock by default).
        """
        if retriever is None:
            class DummyRetriever:
                def retrieve(self, query, k=3, filter_metadata=None):
                    return []
            self.retriever = DummyRetriever()
        else:
            self.retriever = retriever
        self.generator_model = generator_model

        # ── Core components ────────────────────────────────────────────────
        self.classifier        = ClinicalTextClassifier()
        self.phi_detector      = PHIDetector()
        self.pih_scorer        = PIHRiskScorer()
        self.complaint_extractor = ComplaintExtractor()
        self.service_tagger    = ServiceTagger()
        self.input_guardrail   = QueryGuardrail()
        self.output_guardrail  = OutputGuardrail()
        self.retrieval_guard   = RetrievalGuardrail()
        self.reranker          = SimpleCrossEncoderReranker()
        self.query_rewriter    = QueryRewriter()
        self.prompt_builder    = PromptBuilder()

        # ── Golden dataset for mock generator ─────────────────────────────
        self.golden_cases: List[Dict[str, Any]] = []
        golden_path = "datasets/golden_dataset.json"
        if os.path.exists(golden_path):
            try:
                with open(golden_path, "r", encoding="utf-8") as f:
                    self.golden_cases = json.load(f)
            except Exception as e:
                logger.warning(f"[Pipeline] Failed to load golden cases: {e}")

    # ── Mock generator ─────────────────────────────────────────────────────
    def _find_mock_response(self, question: str, retrieved_contexts: List[Dict[str, Any]]) -> str:
        q_cleaned = question.strip().lower()
        best_match = None
        best_score = 0.0

        for case in self.golden_cases:
            case_q = case.get("question", "").strip().lower()
            if case_q == q_cleaned:
                return case.get("ground_truth_answer", "")
            q_words = set(q_cleaned.split())
            case_words = set(case_q.split())
            if q_words and case_words:
                overlap = len(q_words & case_words) / max(len(q_words), len(case_words))
                if overlap > best_score:
                    best_score = overlap
                    best_match = case

        if best_match and best_score > 0.6:
            return best_match.get("ground_truth_answer", "")

        if retrieved_contexts:
            first_src = retrieved_contexts[0].get("metadata", {}).get("source_file", "cancer_doc_0.txt")
            return (
                f"Based on the clinical guidelines in [{first_src}], "
                "The retrieved evidence provides clinical guidelines for this condition."
            )
        return "I cannot find this recommendation in the provided clinical guidelines."

    # ── Main generate ──────────────────────────────────────────────────────
    def generate(
        self,
        question: str,
        k: int = 3,
        tracer: Optional[LangfuseTracer] = None,
        trace_id: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:

        latency: Dict[str, float] = {}
        total_start = time.time()

        # ═══════════════════════════════════════════════════════════════════
        # STEP 1 — Intent Guardrail (Task 1)
        # ═══════════════════════════════════════════════════════════════════
        t0 = time.time()
        guardrail_result = self.input_guardrail.check(question)
        latency["guardrail_ms"] = round((time.time() - t0) * 1000, 2)

        if not guardrail_result["is_relevant"]:
            return self._blocked_response(question, guardrail_result["reason"], time.time() - total_start)

        # ═══════════════════════════════════════════════════════════════════
        # STEP 2 — PHI Detection & Masking (Task 14)
        # ═══════════════════════════════════════════════════════════════════
        phi_scan = self.phi_detector.detect(question)
        safe_question = self.phi_detector.redact(question) if phi_scan["has_phi"] else question
        if phi_scan["has_phi"]:
            logger.warning(f"[PHI] Detected {phi_scan['phi_count']} PHI items — masking before processing.")

        # ═══════════════════════════════════════════════════════════════════
        # STEP 3 — Query Rewriting (Task 3)
        # ═══════════════════════════════════════════════════════════════════
        t0 = time.time()
        rewritten_query = self.query_rewriter.rewrite(safe_question, chat_history)
        latency["rewrite_ms"] = round((time.time() - t0) * 1000, 2)
        logger.info(f"[Rewriter] '{safe_question}' → '{rewritten_query}'")

        # ═══════════════════════════════════════════════════════════════════
        # STEP 4 — Medical NER (Task 4) + ICD Mapping (Task 5)
        # ═══════════════════════════════════════════════════════════════════
        entities = extract_clinical_entities(rewritten_query)
        icd_data = {"code": "R69", "category": "Unknown", "description": ""}
        all_diseases = entities.get("cancer_types", []) + entities.get("diseases", [])
        if all_diseases:
            icd_data = resolve_icd_code(all_diseases[0])

        # Build metadata filter from extracted entities for smarter retrieval
        metadata_filter: Optional[Dict[str, Any]] = None
        if icd_data["code"] != "R69":
            metadata_filter = {"icd_10_code": icd_data["code"]}

        # ═══════════════════════════════════════════════════════════════════
        # STEP 5 — Hybrid Retrieval + Reranking (Tasks 6, 7)
        # ═══════════════════════════════════════════════════════════════════
        t0 = time.time()
        candidate_contexts = self.retriever.retrieve(rewritten_query, k=k * 3, filter_metadata=metadata_filter)
        # Fall back without filter if metadata filter returns nothing
        if not candidate_contexts and metadata_filter:
            candidate_contexts = self.retriever.retrieve(rewritten_query, k=k * 3)
        latency["retrieval_ms"] = round((time.time() - t0) * 1000, 2)

        t0 = time.time()
        contexts = self.reranker.rerank(rewritten_query, candidate_contexts, top_n=k)
        latency["reranking_ms"] = round((time.time() - t0) * 1000, 2)

        # ═══════════════════════════════════════════════════════════════════
        # STEP 6 — Retrieval Guardrail (Task 2)
        # ═══════════════════════════════════════════════════════════════════
        retrieval_validation = self.retrieval_guard.validate(contexts)
        if not retrieval_validation["passed"]:
            logger.warning(f"[RetrievalGuard] {retrieval_validation['reason']}")
            return self._no_evidence_response(question, retrieval_validation, entities, phi_scan, time.time() - total_start)

        retrieval_stats = retrieval_validation["stats"]

        # ═══════════════════════════════════════════════════════════════════
        # STEP 7 — PromptBuilder (Tasks 9, 13)
        # ═══════════════════════════════════════════════════════════════════
        user_prompt = self.prompt_builder.build_user_prompt(
            question=rewritten_query,
            context=contexts,
            chat_history=chat_history,
            ner_entities=entities,
        )

        # ═══════════════════════════════════════════════════════════════════
        # STEP 8 — LLM Generation
        # ═══════════════════════════════════════════════════════════════════
        response_text = ""
        input_tokens = output_tokens = 0
        cost = 0.0
        provider = "openai"

        def _do_generate():
            nonlocal response_text, input_tokens, output_tokens, cost, provider
            if self.generator_model.startswith("bedrock"):
                from bedrock.bedrock_client import BedrockGenerator  # pyrefly: ignore
                bedrock_model = self.generator_model.replace("bedrock:", "").strip() or "claude-3-haiku"
                bedrock = BedrockGenerator(model_name=bedrock_model)
                result = bedrock.generate(CLINICAL_SYSTEM_PROMPT, user_prompt)
                response_text, input_tokens, output_tokens, cost, provider = (
                    result["response_text"], result["input_tokens"], result["output_tokens"], result["cost"], result["provider"]
                )
            elif self.generator_model.startswith("huggingface"):
                from huggingface_hub import InferenceClient  # pyrefly: ignore
                hf_model = self.generator_model.replace("huggingface:", "").strip() or "mistralai/Mistral-7B-Instruct-v0.2"
                try:
                    client = InferenceClient(token=os.getenv("HF_TOKEN"))
                    res = client.chat_completion(
                        model=hf_model,
                        messages=[
                            {"role": "system", "content": CLINICAL_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt}
                        ],
                        max_tokens=800,
                        temperature=0.1
                    )
                    response_text = res.choices[0].message.content.strip()
                except Exception as e:
                    logger.error(f"HF generation failed: {e}")
                    raise RuntimeError(f"HF generation failed: {e}")
                input_tokens = int(len(user_prompt.split()) * 1.3)
                output_tokens = int(len(response_text.split()) * 1.3)
                cost = 0.0
                provider = f"huggingface ({hf_model})"
            elif self.generator_model.startswith("ollama"):
                ollama_model = self.generator_model.replace("ollama:", "").strip() or "mistral"
                from openai import OpenAI  # pyrefly: ignore
                client = OpenAI(
                    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                    api_key=os.getenv("OLLAMA_API_KEY", "ollama")
                )
                try:
                    completion = client.chat.completions.create(
                        model=ollama_model,
                        messages=[
                            {"role": "system", "content": CLINICAL_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                    )
                    response_text = completion.choices[0].message.content
                    usage = completion.usage
                    input_tokens = usage.prompt_tokens if usage else 0
                    output_tokens = usage.completion_tokens if usage else 0
                except Exception as e:
                    logger.error(f"Ollama generation failed: {e}")
                    raise RuntimeError(f"Ollama generation failed: {e}")
                
                # Simulated API cost based on standard Llama 3 8B API pricing (e.g., Groq/Together)
                # $0.05 per 1M input tokens, $0.08 per 1M output tokens
                cost = (input_tokens * 0.00000005) + (output_tokens * 0.00000008)
                provider = f"ollama ({ollama_model})"
            elif self.generator_model.startswith("xai"):
                xai_model = self.generator_model.replace("xai:", "").strip() or "grok-2-latest"
                from openai import OpenAI  # pyrefly: ignore
                client = OpenAI(
                    base_url="https://api.x.ai/v1",
                    api_key=os.getenv("XAI_API_KEY")
                )
                try:
                    completion = client.chat.completions.create(
                        model=xai_model,
                        messages=[
                            {"role": "system", "content": CLINICAL_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                    )
                    response_text = completion.choices[0].message.content
                    usage = completion.usage
                    input_tokens = usage.prompt_tokens if usage else 0
                    output_tokens = usage.completion_tokens if usage else 0
                except Exception as e:
                    error_msg = str(e)
                    if "invalid-argument" in error_msg or "Model not found" in error_msg:
                        logger.error(f"xAI API rejected the model name. This almost always happens when a new xAI account has $0.00 credits. Please add billing credits at console.x.ai. Raw error: {e}")
                        raise RuntimeError(f"xAI API Error (Check Billing Credits): {e}")
                    
                    logger.error(f"xAI generation failed: {e}")
                    raise RuntimeError(f"xAI generation failed: {e}")
                
                cost = 0.0
                provider = f"xai ({xai_model})"
            elif self.generator_model.startswith("gemini"):
                gemini_model = self.generator_model.replace("gemini:", "").strip() or "gemini-1.5-flash"
                import google.generativeai as genai  # pyrefly: ignore
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                try:
                    model = genai.GenerativeModel(
                        model_name=gemini_model,
                        system_instruction=CLINICAL_SYSTEM_PROMPT,
                    )
                    response = model.generate_content(user_prompt)
                    response_text = response.text
                    
                    # Usage parsing for Gemini
                    if hasattr(response, 'usage_metadata') and response.usage_metadata:
                        input_tokens = response.usage_metadata.prompt_token_count
                        output_tokens = response.usage_metadata.candidates_token_count
                    else:
                        input_tokens, output_tokens = 0, 0
                except Exception as e:
                    logger.error(f"Gemini generation failed: {e}")
                    raise RuntimeError(f"Gemini generation failed: {e}")
                
                if "flash" in gemini_model:
                    cost = (input_tokens * 0.000000075) + (output_tokens * 0.0000003)
                else:
                    cost = (input_tokens * 0.00000125) + (output_tokens * 0.000005)
                provider = f"gemini ({gemini_model})"
            elif self.generator_model.startswith("groq"):
                groq_model = self.generator_model.replace("groq:", "").strip() or "llama-3.1-8b-instant"
                from openai import OpenAI  # pyrefly: ignore
                client = OpenAI(
                    base_url="https://api.groq.com/openai/v1",
                    api_key=os.getenv("GROQ_API_KEY")
                )
                try:
                    completion = client.chat.completions.create(
                        model=groq_model,
                        messages=[
                            {"role": "system", "content": CLINICAL_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                    )
                    response_text = completion.choices[0].message.content
                    usage = completion.usage
                    input_tokens = usage.prompt_tokens if usage else 0
                    output_tokens = usage.completion_tokens if usage else 0
                except Exception as e:
                    logger.error(f"Groq generation failed: {e}")
                    raise RuntimeError(f"Groq generation failed: {e}")
                
                # Assume pricing for llama-3.1-8b/70b
                if "70b" in groq_model:
                    cost = (input_tokens * 0.00000059) + (output_tokens * 0.00000079)
                else:
                    cost = (input_tokens * 0.00000005) + (output_tokens * 0.00000008)
                provider = f"groq ({groq_model})"
            elif self.generator_model.startswith("longcat"):
                import requests
                
                url = os.getenv("LONGCAT_BASE_URL", "https://api.longcat.chat/openai/v1")
                if not url.endswith("/chat/completions"):
                    url = f"{url.rstrip('/')}/chat/completions"
                    
                headers = {
                    "Authorization": f"Bearer {os.getenv('LONGCAT_API_KEY')}",
                    "Content-Type": "application/json"
                }

                data = {
                    "model": "LongCat-2.0",
                    "messages": [
                        {"role": "system", "content": CLINICAL_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_tokens": 2048,
                    "temperature": 0.7
                }

                try:
                    response = requests.post(url, headers=headers, json=data)
                    response.raise_for_status()
                    result_json = response.json()
                    response_text = result_json["choices"][0]["message"]["content"]
                    usage = result_json.get("usage", {})
                    input_tokens = usage.get("prompt_tokens", 0)
                    output_tokens = usage.get("completion_tokens", 0)
                except Exception as e:
                    logger.error(f"LongCat generation failed: {e}")
                    raise RuntimeError(f"LongCat generation failed: {e}")
                
                cost = (input_tokens * 0.000005) + (output_tokens * 0.000015)
                provider = "longcat (LongCat-2.0)"
            elif self.generator_model == "mock":
                response_text = self._find_mock_response(question, contexts)
                input_tokens = len(CLINICAL_SYSTEM_PROMPT.split()) + len(user_prompt.split())
                output_tokens = len(response_text.split())
                cost = (input_tokens * 0.00000015) + (output_tokens * 0.0000006)
                time.sleep(0.05)
                provider = "mock"
            else:
                from openai import OpenAI  # pyrefly: ignore
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                completion = client.chat.completions.create(
                    model=self.generator_model,
                    messages=[
                        {"role": "system", "content": CLINICAL_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                response_text = completion.choices[0].message.content
                usage = completion.usage
                input_tokens, output_tokens = usage.prompt_tokens, usage.completion_tokens
                if "gpt-4o-mini" in self.generator_model:
                    cost = (input_tokens * 0.00000015) + (output_tokens * 0.0000006)
                elif "gpt-4o" in self.generator_model:
                    cost = (input_tokens * 0.000005) + (output_tokens * 0.000015)
                else:
                    cost = (input_tokens + output_tokens) * 0.000002

        t0 = time.time()
        if tracer and trace_id:
            with tracer.span(trace_id, "Generation", {"prompt": user_prompt, "model": self.generator_model}) as span:
                _do_generate()
                span["output"] = response_text
                span["input_tokens"] = input_tokens
                span["output_tokens"] = output_tokens
        else:
            _do_generate()
        latency["generation_ms"] = round((time.time() - t0) * 1000, 2)

        # ═══════════════════════════════════════════════════════════════════
        # STEP 9 — Grounding + Hallucination Verification (Tasks 10, 11)
        # ═══════════════════════════════════════════════════════════════════
        t0 = time.time()
        verification = self.output_guardrail.verify(rewritten_query, response_text, contexts)
        latency["verification_ms"] = round((time.time() - t0) * 1000, 2)

        if not verification["passed"]:
            logger.warning(f"[OutputGuard] Hallucination detected: {verification['reason']}")
            response_text = "I don't have information related to your question in the available knowledge base."

        # ═══════════════════════════════════════════════════════════════════
        # STEP 10 — Confidence Scoring (Task 12)
        # ═══════════════════════════════════════════════════════════════════
        num_unique_sources = len({c.get("metadata", {}).get("source_file", "") for c in contexts})
        confidence = compute_confidence_score(retrieval_stats, verification, num_unique_sources)

        # ═══════════════════════════════════════════════════════════════════
        # STEP 11 — Supplementary Analysis
        # ═══════════════════════════════════════════════════════════════════
        classical_ai     = self.classifier.classify_urgency(question)
        topic_class      = self.classifier.classify_topic(question)
        pih_risk         = self.pih_scorer.score(question)
        complaints       = self.complaint_extractor.extract(question)
        service_tag      = self.service_tagger.tag(question)

        latency["total_ms"] = round((time.time() - total_start) * 1000, 2)

        # ═══════════════════════════════════════════════════════════════════
        # STEP 12 — Build & return structured response
        # ═══════════════════════════════════════════════════════════════════
        return {
            "question": question,
            "rewritten_query": rewritten_query,
            "response_text": response_text,
            "retrieved_contexts": contexts,
            "latency": latency["total_ms"] / 1000,
            "latency_breakdown": latency,
            "token_usage": {
                "input_tokens": int(input_tokens),
                "output_tokens": int(output_tokens),
                "total_tokens": int(input_tokens + output_tokens),
            },
            "cost": float(cost),
            "provider": provider,
            "query_entities": entities,
            "icd_data": icd_data,
            "confidence": confidence,
            "verification": verification,
            "retrieval_validation": retrieval_validation,
            "classical_ai": {"urgency": classical_ai, "topic": topic_class},
            "phi_scan": phi_scan,
            "pih_risk": pih_risk,
            "complaints": complaints,
            "service_tag": service_tag,
        }

    # ── Helper response builders ───────────────────────────────────────────
    def _blocked_response(self, question: str, reason: str, latency: float) -> Dict[str, Any]:
        return {
            "question": question, "rewritten_query": question,
            "response_text": "I don't have this information.",
            "retrieved_contexts": [], "latency": latency,
            "latency_breakdown": {}, "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "cost": 0.0, "provider": "guardrail_blocked",
            "query_entities": {}, "icd_data": {}, "confidence": {},
            "verification": {"passed": False, "reason": reason, "hallucination_detected": False},
            "retrieval_validation": {"passed": False}, "classical_ai": {"urgency": {"urgency": "none", "confidence": 0.0}, "topic": {"topic": "blocked", "confidence": 0.0}},
            "phi_scan": {}, "pih_risk": 0.0, "complaints": [], "service_tag": "blocked",
            "guardrail_reason": reason,
        }

    def _no_evidence_response(self, question: str, retrieval_validation: Dict, entities: Dict, phi_scan: Dict, latency: float) -> Dict[str, Any]:
        return {
            "question": question, "rewritten_query": question,
            "response_text": "I don't have information related to your question in the available knowledge base.",
            "retrieved_contexts": [], "latency": latency,
            "latency_breakdown": {}, "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "cost": 0.0, "provider": "retrieval_blocked",
            "query_entities": entities, "icd_data": {},
            "confidence": {"overall": 0.0, "overall_pct": "0%", "overall_label": "Low"},
            "verification": {"passed": False, "reason": "Retrieval failed — no LLM called."},
            "retrieval_validation": retrieval_validation,
            "classical_ai": {"urgency": {"urgency": "none", "confidence": 0.0}, "topic": {"topic": "unknown", "confidence": 0.0}},
            "phi_scan": phi_scan, "pih_risk": 0.0, "complaints": [], "service_tag": "unknown",
        }
