# src/ingestion/prompt_templates.py
"""
prompt_templates.py — Task 9 & 13

PromptBuilder v2:
- Conversation history injection
- Rich evidence block with similarity scores + chunk metadata
- Enhanced citation instructions (Document | Source | Section | ICD | Similarity)
- Grounding enforcement directives
- Anti-hallucination system prompt
"""

from typing import List, Dict, Any, Optional


CLINICAL_SYSTEM_PROMPT = """\
You are ClinicalMind, a premier Clinical AI Research Assistant built for medical evidence synthesis.

MISSION: Produce exhaustive, well-cited, clinically accurate responses derived EXCLUSIVELY from the provided evidence.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT GROUNDING RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. CONTEXT-ONLY: Answer ONLY using facts explicitly stated in the <evidence> blocks. Never use prior training knowledge as the source of a factual claim.

2. CITATION MANDATORY: Every clinical statement MUST end with its citation in the format:
   [DocN | source_file | Section | Similarity: X.XX]
   Example: "Standard first-line treatment is carboplatin-paclitaxel [Doc2 | nsclc_guidelines.txt | Treatment | Similarity: 0.91]"

3. UNANSWERABLE: If the evidence is unrelated to the question or does not support an answer, reply with ONLY one of these exact responses (do not generate any additional text):
   "I don't have information related to your question in the available knowledge base."
   OR
   "I couldn't find relevant information to answer your question."
   OR
   "The available documents do not contain information needed to answer this question."
   Do NOT attempt to answer from memory.

4. OUT-OF-DOMAIN: If the query is not about clinical medicine, reply EXACTLY:
   "I don't have this information."

5. NO EXTRAPOLATION: Do not infer, extrapolate, or fill gaps. Report only what is written.

6. UNCERTAINTY: When evidence is partial, explicitly state "Evidence is limited to..." and cite what IS available.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Structure your response as follows:
### Summary
2-3 sentence clinical summary with citations.

### Detailed Evidence Synthesis
Multi-paragraph synthesis of all retrieved evidence. Cross-reference documents where relevant.

### Key Clinical Points
- Bullet point list of the most important findings with citations.

### Evidence Quality
Brief note on the strength and completeness of the retrieved evidence.
"""


class PromptBuilder:
    """
    Task 9 & 13 — Constructs grounded, heavily cited clinical prompts.
    
    Improvements over v1:
    - Rich chunk headers with ICD code, guideline name, year, page, section
    - Rerank score surfaced per chunk for citation
    - Conversation history injected for follow-up support
    - Grounding and citation instructions strengthened
    """

    def build_user_prompt(
        self,
        question: str,
        context: List[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, str]]] = None,
        ner_entities: Optional[Dict[str, Any]] = None,
    ) -> str:
        parts: List[str] = []

        # ── Conversation History ────────────────────────────────────────────
        if chat_history:
            parts.append("## Conversation History")
            for entry in chat_history[-6:]:   # last 3 exchanges
                role = entry.get("role", "").capitalize()
                content = entry.get("content", "")
                parts.append(f"{role}: {content}")
            parts.append("")

        # ── NER Clinical Context (helps LLM scope the answer) ──────────────
        if ner_entities:
            ner_lines = []
            for key, values in ner_entities.items():
                if values and isinstance(values, list):
                    display_key = key.replace('_', ' ').title()
                    ner_lines.append(f"{display_key}: {', '.join(values)}")
            if ner_lines:
                parts.append("## Extracted Clinical Context")
                parts.extend(ner_lines)
                parts.append("")

        # ── Retrieved Evidence Blocks ───────────────────────────────────────
        parts.append("## Retrieved Clinical Evidence")
        for i, chunk in enumerate(context, 1):
            text = chunk.get("text", "").strip()
            meta = chunk.get("metadata", {})

            source     = meta.get("source_file", "unknown_source.txt")
            category   = meta.get("clinical_category", "Oncology")
            icd_code   = meta.get("icd_10_code", "")
            icd_desc   = meta.get("icd_description", "")
            chunk_id   = chunk.get("id", f"chunk_{i}")
            guideline  = meta.get("guideline", "")
            year       = meta.get("year", "")
            section    = meta.get("section", "")
            page       = meta.get("page", "")
            score      = chunk.get("rerank_score") or chunk.get("score") or 0.0

            # Build rich header for citation
            header_parts = [f"[Doc{i}]", f"Source: {source}"]
            if guideline:  header_parts.append(f"Guideline: {guideline}")
            if year:       header_parts.append(f"Year: {year}")
            if section:    header_parts.append(f"Section: {section}")
            if page:       header_parts.append(f"Page: {page}")
            if icd_code:   header_parts.append(f"ICD-10: {icd_code} ({icd_desc})")
            header_parts.append(f"Similarity: {score:.3f}")
            header_parts.append(f"Chunk: {chunk_id}")
            header_parts.append(f"Category: {category}")

            parts.append(" | ".join(header_parts))
            parts.append(text)
            parts.append("─" * 60)

        # ── Question & Instruction ──────────────────────────────────────────
        parts.append(f"\n## Clinician Question\n{question}")
        parts.append(
            "\n## Clinical Response\n"
            "CRITICAL: Write a comprehensive, multi-section clinical response. "
            "Every factual claim MUST be cited using the format [DocN | source | Similarity: X.XX]. "
            "If the evidence is insufficient or unrelated, respond with ONLY one of these responses (no other text): "
            "'I don't have information related to your question in the available knowledge base.' OR "
            "'I couldn't find relevant information to answer your question.' OR "
            "'The available documents do not contain information needed to answer this question.' "
            "Do NOT use any knowledge outside the above evidence blocks."
        )

        return "\n".join(parts)

def format_user_prompt(question: str, context: List[Dict[str, Any]]) -> str:
    builder = PromptBuilder()
    return builder.build_user_prompt(question, context)

