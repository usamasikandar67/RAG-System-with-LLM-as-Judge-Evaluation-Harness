# Specification - Stage 05: Prompt Construction

## Goal / Responsibility
The goal of this stage is to construct system and user prompt templates that inject retrieved contexts into the LLM prompt. It is responsible for formatting chunk citations, placing safety constraints, organizing the structure of the prompt dynamically within token boundaries, and ensuring source materials are presented clearly to the generator model.

## Inputs and Outputs

### Inputs
- **User Query**: String query.
- **Retrieved Contexts**: List of document chunks (`List[Dict[str, Any]]`) from Stage 04.
- **System Constraints**: Standard context-grounding instructions.

### Outputs
- **Formatted Prompt**: A list of messages (e.g. ChatML/OpenAI format) containing structured system, context, and user inputs:
  ```python
  List[Dict[str, str]]  # list of {"role": "...", "content": "..."}
  ```

## Interfaces/APIs
This stage exposes a prompt builder service:
```python
from typing import List, Dict, Any

class PromptBuilder:
    def __init__(self, system_template_path: str = None):
        pass

    def build_chat_messages(self, query: str, contexts: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        pass
```

## Tech Choices and Why
- **Jinja2 / String Formatting**: Standard templating frameworks are lightweight, highly readable, and allow complex conditional loops (e.g. iterating over contexts, formatting IDs, handling missing fields).
- **Tiktoken**: To count tokens dynamically and prune old or lower-ranking retrieved contexts if they exceed LLM context windows.
- **Alternatives**: LangChain Hub templates (creates unnecessary external dependencies and network queries to download prompts, violating local stability).

## Config/Env Vars
- `PROMPT_TEMPLATE_PATH` (Default: `./src/stage_05_prompt_construction/templates/`)
- `MAX_PROMPT_TOKENS` (Default: `4000` tokens)

## Explicit Non-Goals
- Executing the prompt against LLM API endpoints (handled in Stage 06).
- Reranking retrieved chunks or calculating semantic similarity (handled in Stage 04).
- Storing conversation history/sessions (this is a stateless retrieval-generation pass).

## Acceptance Criteria
- PromptBuilder formats contexts with clear labels (e.g., `[Document 1] title: ... \n content: ...`).
- If combined prompt length exceeds `MAX_PROMPT_TOKENS`, lowest ranked contexts are pruned using tiktoken counts until inside boundaries.
- Prompts include system instructions instructing the model to reply "I do not know" if context contains insufficient information.
- Prompts mandate inline citations (e.g., `[Doc 1]`) in the model's generated answer.
- Unit tests verify token-limiting rules and prompt assembly output formats.
