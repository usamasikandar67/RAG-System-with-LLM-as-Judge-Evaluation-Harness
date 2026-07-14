# Specification - Stage 10: CI/CD

## Goal / Responsibility
The goal of this stage is to build the automated CI/CD engine for the repository using GitHub Actions workflows. It is responsible for orchestrating lint and style checks (Ruff), executing unit tests for each pipeline stage, checking type declarations, running the evaluation harness on code changes, gating pull request merges based on performance regression limits, and auto-deploying the containerized comparison dashboard to main.

## Inputs and Outputs

### Inputs
- **Code Changes / Pull Requests**: Triggers on code submissions.
- **Repository Secrets**: OpenAI/Anthropic API keys, LangSmith settings, container registry keys.

### Outputs
- **Pipeline Actions Log**: Execution feedback for commits and PR requests.
- **Evaluation Summary Comment**: Automatic markdown report detailing LLM-as-judge scores posted directly to GitHub PRs.
- **Deployed Dashboard**: Build and push trigger for Streamlit containerization.

## Interfaces/APIs
This stage defines GitHub Workflow definitions under `.github/workflows/`:
- `ci.yml`: Checks format, lints, runs unit tests, runs type checks.
- `eval.yml`: Runs evaluations against golden dataset on target PRs and computes regressions.
- `deploy.yml`: Deploys Streamlit container and registers MLflow artifacts to main.

Exposes a CLI script for local CI simulation:
```bash
bash scripts/run_ci.sh
```

## Tech Choices and Why
- **GitHub Actions**: Native GitHub orchestrator, requires no external infrastructure, supports robust secret environments and runner caching.
- **Ruff**: Extremely fast linting and formatting tool for Python, replacing Flake8/Black/Isort with significant speedups.
- **Docker**: For containerizing the Streamlit dashboard to ensure environment consistency upon deployment.
- **Alternatives**: GitLab CI/CD, Jenkins (unnecessary hosting costs, GitHub Actions is standard for simple git-based automation).

## Config/Env Vars
- Secrets configured in GitHub Repository Settings:
  - `OPENAI_API_KEY` (Required for evaluation gate)
  - `ANTHROPIC_API_KEY` (Optional)
  - `DOCKER_REGISTRY_TOKEN` (For deployment)
  - `DOCKER_REGISTRY_USER` (For deployment)

## Explicit Non-Goals
- Hosting physical runner infrastructure (uses GitHub-hosted runners).
- Running full ingestion tasks (`00-knowledge-sources`) inside CI (ingestion data is mocked to keep builds fast).
- Provisioning target production environments (deployment handles application delivery, not infrastructure setup like Terraform).

## Acceptance Criteria
- Code changes without proper formatting or failing tests block merges.
- Pull requests modifying retrieval, prompt, or generation components trigger the evaluation gate.
- Evaluation gate calculates aggregate metrics and fails the build if scores regress relative to a reference database by more than `0.05` points.
- Evaluation results are commented on the PR automatically using a GitHub API integration.
- Deployed workflow successfully builds a Docker container and registers MLflow models upon push to `main`.
