# Deployment Strategy - Cancer Clinical AI Evaluation Platform

## Multi-Stage Dockerfile Blueprint

```dockerfile
# Stage 1: Build dependencies
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Final minimal runtime
FROM python:3.11-slim AS runner
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY src/ ./src
COPY datasets/ ./datasets
COPY experiments/ ./experiments
COPY run_pipeline.py .

ENV PATH=/root/.local/bin:$PATH
EXPOSE 8501
CMD ["streamlit", "run", "src/dashboard/app.py", "--server.port=8501"]
```

## CI/CD Gating (GitHub Actions)
- **Pull Request Trigger**: Code changes trigger the GitHub workflow runner.
- **Verification Runner**: Runs linting (ruff), unit tests (pytest), and executes the RAG regression gate.
- **Status Gating**:
  ```bash
  python3 run_pipeline.py --retriever hybrid --baseline-exp exp_baseline_production
  ```
  If this command exits with status code `1`, the PR merge is blocked.
