# Specification - Stage 09: Streamlit Dashboard

## Goal / Responsibility
The goal of this stage is to build a visual, production-grade comparison dashboard using Streamlit. It is responsible for loading run history and evaluation metrics from SQLite and MLflow, presenting comparisons between the three retrieval strategies (V1, V2, V3), highlighting performance regressions, enabling deep-dive failure analysis (e.g. inspecting questions where LLM-as-judge scores were low), and exporting reports.

## Inputs and Outputs

### Inputs
- **SQLite Database (`data/runs.db`)**: Historic run information and query transaction metrics.
- **MLflow Artifacts**: Logged parameters, charts, and metrics runs.
- **Golden Dataset**: For comparing exact queries and original reference texts.

### Outputs
- **Interactive Web Interface**: Streamlit application hosted locally or containerized.
- **Export Files**: CSV tables of run results, or print-ready PDF summary reports showing RAG metrics comparisons.

## Interfaces/APIs
This stage is a visual interface. It exposes the application entrypoint:
- `streamlit run src/stage_09_streamlit_dashboard/app.py`
- Exposes no programmatic API to upstream stages. It consumes data written by Stages 07 and 08.

## Tech Choices and Why
- **Streamlit**: Python web application framework allowing rapid development of interactive data applications, native chart components, and simple layout scripting.
- **Plotly Express / Altair**: Interactive graphing libraries that integrate with Streamlit for rendering score distributions, latency boxplots, and metric-over-time trends.
- **Pandas**: To manipulate SQL execution tables and generate aggregate pivot views easily.
- **Alternatives**: Gradio (good for simple model inputs/outputs, but less customizable for tabular dashboards and multiple pages/tabs).

## Config/Env Vars
- `STREAMLIT_PORT` (Default: `8501`)
- `DATABASE_URL` (Default: `sqlite:///data/runs.db`)
- `MLFLOW_TRACKING_URI` (Default: `./mlruns`)

## Explicit Non-Goals
- Running pipeline evaluations directly from the UI (evaluations are triggered via CLI or CI workflows; the dashboard is for analysis).
- Modifying SQLite runs or deleting historic experiments from the database (read-only application).
- User authentication and access control (runs in secure private networks or local workspaces).

## Acceptance Criteria
- Dashboard runs and is accessible at `http://localhost:8501`.
- Contains a "Compare Runs" tab showing a side-by-side comparison table of retrieval strategies across Faithfulness, Recall, Precision, ROUGE-L, and Latency.
- Includes a "Failure Analysis" tab filtering queries with scores below a user-defined slider threshold.
- Renders interactive charts showing:
  - Metric distribution comparison (box plots).
  - Accuracy progression over historic runs (line charts).
- Export button downloads a formatted CSV containing filtered query details.
- UI elements (sliders, select boxes, inputs) have unique CSS IDs/keys for testing.
