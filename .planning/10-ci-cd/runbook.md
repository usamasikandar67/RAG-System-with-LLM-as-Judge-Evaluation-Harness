# Runbook - Stage 10: CI/CD

## CI/CD Architecture Overview
The repository contains three automated pipelines defined under `.github/workflows/`:
1. `ci.yml`: Standard pull request checks (format, lint, unit tests).
2. `eval.yml`: Triggered on modifications to strategies, prompts, or generation stages. Executes evaluation runs against the golden dataset and writes metrics comments.
3. `deploy.yml`: Executed on merging to `main`. Builds and pushes the Streamlit Docker image and registers MLflow artifacts.

## Sourced Secrets and Environment Variables
To execute the workflows successfully in GitHub Actions, configure the following repository secrets under **Settings > Secrets and variables > Actions**:
- `OPENAI_API_KEY`: API key for OpenAI model and LLM-as-judge runs.
- `ANTHROPIC_API_KEY`: API key for alternative Claude generation tests.
- `DOCKER_REGISTRY_USER`: Target registry account.
- `DOCKER_REGISTRY_TOKEN`: Access credentials/token for pushing dashboard images.

## Local Emulation
You can verify formatting and run stage unit tests locally prior to pushing code by running:
```bash
# Verify formatting
ruff format --check src/ tests/
ruff check src/ tests/

# Execute all stage unit tests
pytest tests/
```

## Troubleshooting Failures
1. **Ruff lint failures**: Fix them automatically using `ruff format src/ tests/` or `ruff check --fix src/ tests/`.
2. **Evaluation Gate Regression failures**: If a build fails due to performance regressions:
   - Check if context alignment or prompt templates were changed.
   - Inspect failure queries in the run output JSON files locally.
   - If changes are expected and correct, update the baseline run records against which regressions are measured.
