# Runbook - Stage 07: Evaluation Harness

## Local Development Setup
- Install dependencies: `pip install nltk rouge-score openai`.
- Verify the golden dataset exists: `ls ./data/evaluation/golden_dataset.json`. If it doesn't exist, create it or run:
  ```bash
  python -m src.stage_07_evaluation_harness.generate_sample_golden
  ```

## Running Evaluations
To execute evaluation on a specific strategy:
```bash
python -m src.stage_07_evaluation_harness.run --strategy hybrid
```

To run all three pipelines and produce comparison results:
```bash
python -m src.stage_07_evaluation_harness.run --strategy all
```

## Running Tests
Run metrics testing:
```bash
pytest tests/test_stage_07.py
```

## CI/CD Workflow
- Triggered inside GitHub Actions `eval.yml` workflow when changes affect retriever or generator packages.
- The evaluation runs against the golden dataset. If the aggregate Faithfulness or Context Recall falls below the regression limit, the CI run fails and block integration.

## Configuration and Environment
- Environment Variables:
  - `OPENAI_API_KEY`: Required for LLM-as-judge calls.
  - `EVAL_MIN_THRESHOLD`: e.g. `0.80`.

## Troubleshooting and Rollback
- High evaluator cost: Ensure the judge model runs with cached responses or mock environments when testing orchestration logic.
- Parser failure: If judge response fails to parse, review JSON-formatting constraints within the evaluation prompt.
