# Runbook - Stage 09: Streamlit Dashboard

## Local Development Setup
- Install dependencies: `pip install streamlit pandas plotly altair`.

## Running the Dashboard
To start the dashboard app locally:
```bash
streamlit run src/stage_09_streamlit_dashboard/app.py --server.port 8501
```
Open a browser at `http://localhost:8501`.

## Running Tests
Run UI integration mocks and component tests:
```bash
pytest tests/test_stage_09.py
```

## CI/CD Workflow
- Linting and unit tests will run on push/PRs.
- Continuous deployment builds a production container matching the Dockerfile defined in this directory and hosts it on a target environment.

## Configuration and Environment
- Environment Variables:
  - `DATABASE_URL`: Connection URL to database.
  - `MLFLOW_TRACKING_URI`: Path to MLflow run caches.

## Troubleshooting and Rollback
- App fails to start/port in use: Choose another port parameter:
  ```bash
  streamlit run src/stage_09_streamlit_dashboard/app.py --server.port 8502
  ```
- Missing run data: Verify that evaluations were run (Stage 07) and logged to the SQLite database (Stage 08) beforehand. Check paths.
