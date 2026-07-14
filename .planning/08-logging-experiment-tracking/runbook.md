# Runbook - Stage 08: Logging & Experiment Tracking

## Local Development Setup
- Install dependencies: `pip install mlflow sqlalchemy`.
- If using LangSmith, set configuration keys:
  ```bash
  export LANGCHAIN_TRACING_V2="true"
  export LANGCHAIN_API_KEY="ls__..."
  export LANGCHAIN_PROJECT="rag-eval-harness"
  ```

## Running MLflow Dashboard
To launch the local MLflow server UI:
```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```
Then navigate to `http://localhost:5000` in the browser.

## Running Tests
Run logging operations unit tests:
```bash
pytest tests/test_stage_08.py
```

## CI/CD Workflow
- Checked in standard PR tests.
- When running evaluation runs in CI (`eval.yml`), database runs are kept in-memory, and MLflow logging targets local file caches rather than remote HTTP servers.

## Configuration and Environment
- Environment Variables:
  - `DATABASE_URL`: Connection string. Defaults to `sqlite:///data/runs.db`.
  - `MLFLOW_TRACKING_URI`: Defaults to `./mlruns`.

## Troubleshooting and Rollback
- SQLite locked errors: Inspect concurrent write processes. Wrap db connection in transient transaction contexts.
- MLflow connection timed out: If using a remote MLflow tracking server, ensure it is reachable; if not, fall back to local `./mlruns` directory.
- Reset database structure:
  ```bash
  rm data/runs.db
  python -m src.stage_08_logging_experiment_tracking.init_db
  ```
