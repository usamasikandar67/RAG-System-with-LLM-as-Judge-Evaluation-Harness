# Specification - Stage 06: Grounded Generation

## Goal / Responsibility
The goal of this stage is to execute grounded query answering using LLMs (supporting OpenAI's GPT models and Anthropic's Claude models). It acts as the final generation block, consuming the compiled chat messages from Stage 05, requesting model completion, handling API errors or rate-limiting, and verifying that the output contains valid source citations.

## Inputs and Outputs

### Inputs
- **Formatted Chat Messages**: List of messages (`List[Dict[str, str]]`) in role/content schema.
- **Generator Model Settings**: Target model string, temperature, max generation tokens.

### Outputs
- **LLM Response**: A dictionary containing:
  - `response_text` (string, raw generated answer)
  - `extracted_citations` (list of chunk IDs or document references parsed from inline brackets)
  - `token_usage` (dictionary with input, output, and total tokens)
  - `finish_reason` (string, e.g. "stop")

## Interfaces/APIs
This stage exposes a unified runner interface:
```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        pass
```

Implementations:
- `OpenAIGenerator`
- `AnthropicGenerator`

## Tech Choices and Why
- **OpenAI / Anthropic Client Libraries**: Official client SDKs provide native support for modern API parameters (e.g., system prompts, structured JSON outputs, temperature scaling).
- **Regex citation parsing**: Light parsing helper extracting citation identifiers (e.g., `\[Doc \d+\]`) from the generated response string to validate presence of citations.
- **Alternatives**: LangChain LCEL (adds complexity and wrapper debugging, direct SDK calls are simpler to trace).

## Config/Env Vars
- `GENERATOR_PROVIDER` (e.g. `openai`, `anthropic`)
- `GENERATOR_MODEL_NAME` (Default: `gpt-4o` or `claude-3-5-sonnet`)
- `GENERATOR_TEMPERATURE` (Default: `0.0` for maximum determination and reliability)
- `GENERATOR_MAX_TOKENS` (Default: `1000`)
- `OPENAI_API_KEY` (Required for OpenAI)
- `ANTHROPIC_API_KEY` (Required for Anthropic)

## Explicit Non-Goals
- Performing queries or merging document indices (handled in Stage 03/04).
- Formatting prompts or limiting contexts (handled in Stage 05).
- Running qualitative LLM-as-judge score evaluations (handled in Stage 07).

## Acceptance Criteria
- Implements `BaseGenerator` exposing the `.generate()` method with standardized output dictionaries.
- Incorporates API call retry mechanisms (with exponential backoff) for status codes 429 (rate limits) and 5xx (server errors).
- Correctly parses and returns lists of document citations present in text output.
- Unit tests run using HTTP mock frameworks (like `responses` or `pytest-mock`) to avoid actual client API hits.
- Exposes generation metadata (input/output tokens) correctly for ingestion by downstream logging (Stage 08).
