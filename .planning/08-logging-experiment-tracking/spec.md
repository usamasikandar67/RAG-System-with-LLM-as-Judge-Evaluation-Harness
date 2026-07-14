# Specification - Stage 08: Logging & Experiment Tracking

## Goal / Responsibility
The goal of this stage is to track experiment parameters, evaluate metrics over time, and log granular execution traces. It is responsible for saving raw query-level execution histories to a local SQLite database (migratable to Postgres), registering run-level metrics in MLflow for comparison, and enabling LangSmith tracing for visual, step-by-step debugging of retriever-generator calls.

## Inputs and Outputs

### Inputs
- **Execution Traces**: Prompt tokens, output tokens, retrieved document metadata, model configurations, latency.
- **Evaluation Outputs**: Faithfulness, Recall, Precision scores from Stage 07.

### Outputs
- **SQLite Database (`data/runs.db`)**: Structured tables representing run parameters, queries, and evaluation scores.
- **MLflow Runs**: Registered metrics and parameters in the configured MLflow Tracking URI.
- **LangSmith Traces**: Logged traces showing the exact context retrieved, latency, and generator answers.

## Interfaces/APIs
This stage exposes standard logging decorators and tracking managers:
```python
from typing import Dict, Any, List

class ExperimentTracker:
    def __init__(self, experiment_name: str):
        pass

    def start_run(self, strategy: str, parameters: Dict[str, Any]) -> str:
        """Starts a tracking run, returns Run ID."""
        pass

    def log_query_transaction(self, run_id: str, query: str, response: str, contexts: List[Dict[str, Any]], metrics: Dict[str, float]) -> None:
        """Saves detailed logs to SQL database."""
        pass

    def end_run(self, run_id: str, aggregate_metrics: Dict[str, float]) -> None:
        """Log aggregate metrics to MLflow and finalize."""
        pass
```

## Tech Choices and Why
- **SQLite**: Local, lightweight database requiring zero server configurations, ideal for developer isolation and offline run storage.
- **MLflow Client API**: Industry-standard open-source experiment tracker. Allows logging hyper-parameters (chunk size, retriever model, k-neighbors) and visualizing metrics.
- **LangSmith**: Built-in compatibility with LLM providers, offering trace visualization, chain debug views, and execution latency analysis.
- **Alternatives**: Weights & Biases (excellent, but MLflow is open-source and easy to host locally; LangSmith is optimized for RAG tracing).

## Config/Env Vars
- **SQL Database**:
  - `DATABASE_URL` (Default: `sqlite:///data/runs.db`)
- **MLflow**:
  - `MLFLOW_TRACKING_URI` (Default: `./mlruns`)
  - `MLFLOW_EXPERIMENT_NAME` (Default: `RAG_Evaluation_Harness`)
- **LangSmith**:
  - `LANGCHAIN_TRACING_V2` (Default: `false`)
  - `LANGCHAIN_API_KEY` (Optional)
  - `LANGCHAIN_PROJECT` (Default: `rag-eval-harness`)

## Explicit Non-Goals
- Real-time user metrics collection or application metrics monitoring (e.g. Prometheus/Grafana).
- Database migrations tooling setup (e.g. Alembic) in the initial scaffolding.
- Implementing UI layers (the database and MLflow files are read by the Streamlit dashboard in Stage 09).

## Acceptance Criteria
- Running evaluations automatically populates `data/runs.db` with query transactions.
- Running evaluations logs parameters (e.g. strategy, model) and aggregated metrics (e.g. faithfulness, BLEU) to MLflow.
- If LangSmith environment variables are set to `true`, traces are sent successfully during pipeline runs without blocking execution on API timeouts.
- Database tables (e.g. `runs`, `queries`) are validated against schema declarations.
- Unit tests verify database logging functions by checking query inserts and runs tables.
