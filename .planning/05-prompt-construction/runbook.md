# Runbook - Stage 05: Prompt Construction

## Local Development Setup
- Install dependencies: `pip install tiktoken jinja2`.

## Testing Prompt Compilation
Verify how formatted messages look:
```bash
python -m src.stage_05_prompt_construction.compile_test
```

## Running Tests
Run prompt compiler unit tests and check token-limit overflow rules:
```bash
pytest tests/test_stage_05.py
```

## CI/CD Workflow
- Automatically run alongside unit tests.
- PRs modifying stage 05 prompt construction trigger the evaluation gate in `eval.yml` to prevent performance regression.

## Configuration and Environment
- Environment Variables:
  - `MAX_PROMPT_TOKENS`: Maximum allowed input context length.

## Troubleshooting and Rollback
- Truncation issues: If contexts are unexpectedly cut off, verify target model tokenizer setup in `tiktoken`.
- System prompt changes: Edit raw template files inside `./src/stage_05_prompt_construction/templates/system_prompt.txt`.
