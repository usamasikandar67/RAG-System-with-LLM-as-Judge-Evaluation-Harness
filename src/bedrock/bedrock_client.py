# src/bedrock/bedrock_client.py

import os
from typing import Dict, Any, List

class BedrockGenerator:
    def __init__(self, model_name: str = "claude-3-haiku"):
        self.model_name = model_name

    def generate(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        Mock generator for Bedrock inference. If AWS bedrock-runtime client is available,
        it could invoke AWS Bedrock. Falls back to structured mock response.
        """
        response_text = (
            f"Based on clinical evidence: Standard protocol treatment involves targeted oncology therapies. "
            f"[Doc1 | source_file | Section | Similarity: 0.95]"
        )
        return {
            "response_text": response_text,
            "cost": 0.00015,
            "provider": f"bedrock (mock:{self.model_name})",
            "token_usage": {"input_tokens": 120, "output_tokens": 45}
        }

class BedrockEmbeddings:
    def __init__(self, model_name: str = "amazon.titan-embed-text-v1"):
        self.model_name = model_name

    def embed_query(self, text: str) -> List[float]:
        # Return dummy vector of dimension 1536
        return [0.01] * 1536

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.01] * 1536 for _ in texts]
