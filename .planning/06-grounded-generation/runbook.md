# Runbook - Stage 06: Grounded Generation

## Local Development Setup
- Install dependencies: `pip install openai anthropic tenacity`.
- Set keys:
  ```bash
  export OPENAI_API_KEY="sk-..."
  export ANTHROPIC_API_KEY="sk-ant-..."
  ```

## Running Generation Verification
Test a simple query generation pass from terminal:
```bash
python -m src.stage_06_grounded_generation.test_generate --provider openai --q "What is Spark Structured Streaming?"
```

## Running Tests
Run unit tests with API client mocks:
```bash
pytest tests/test_stage_06.py
```

## CI/CD Workflow
- Automatically run in unit-test actions.
- PRs modifying generation code triggers the PR evaluation gate (`eval.yml`) to evaluate context alignment changes.

## Configuration and Environment
- Environment Variables:
  - `GENERATOR_PROVIDER`: `openai` or `anthropic`.
  - `GENERATOR_MODEL_NAME`: e.g. `gpt-4o`, `claude-3-5-sonnet`.
  - `GENERATOR_TEMPERATURE`: defaults to `0.0`.

## Troubleshooting and Rollback
- Authentication errors: Verify keys are correctly exported and active.
- Rate-limiting (429): Verify tenacity retry decorator configs. Consider checking organization limits.
- Empty outputs/hallucinations: Check input message compilation formatting from Stage 05.
