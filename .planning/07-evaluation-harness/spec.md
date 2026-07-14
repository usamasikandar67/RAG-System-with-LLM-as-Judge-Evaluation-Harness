# Specification - Stage 07: Evaluation Harness

## Goal / Responsibility
The goal of this stage is to implement the automated evaluation system. It is responsible for assessing the performance of our RAG pipelines by running queries from a golden dataset, using an LLM-as-judge framework to score answer quality, computing classical text overlap metrics, and aggregating the final results for quality tracking.

## Inputs and Outputs

### Inputs
- **Golden Dataset**: `data/evaluation/golden_dataset.json` containing a list of records:
  - `query_id` (string)
  - `query` (string)
  - `ground_truth_context` (string, target reference text)
  - `ground_truth_answer` (string, perfect target response)
- **Pipeline Executions**: Generated answers and retrieved contexts for each golden query from all three retrieval strategies (V1, V2, V3).

### Outputs
- **Evaluation Run Report**: A JSON file (`data/evaluation/runs/run_[timestamp].json`) containing per-query and aggregated metric scores:
  - **Retrieval Metrics**: Context Precision, Context Recall.
  - **Generation Metrics**: Faithfulness (groundedness), Answer Relevance.
  - **Classical Metrics**: BLEU score, ROUGE-L score, Exact Match (EM).
  - Metadata: Retrieval strategy used, execution timestamp, model versions.

## Interfaces/APIs
This stage exposes an evaluation runner:
- CLI tool to execute evaluation runs:
  ```bash
  python -m src.stage_07_evaluation_harness.run --strategy [dense|multi_query|hybrid] --output [run_path]
  ```
- Metric computing API interface:
  ```python
  def compute_faithfulness(query: str, contexts: List[str], response: str) -> float:
      pass

  def compute_context_recall(query: str, ground_truth: str, retrieved_contexts: List[str]) -> float:
      pass
  ```

## Tech Choices and Why
- **LLM-as-Judge**: GPT-4o with structured formatting to execute grading tasks (e.g. counting facts, verifying context contradictions).
- **Ragas-like custom prompts**: Write custom, lightweight evaluation prompts to compute Faithfulness (ratio of statements in the answer supported by context) and Context Recall (ratio of ground truth statements present in retrieved context), avoiding heavy imports.
- **NLTK / Rouge-Score library**: For fast calculation of classical lexical metrics (BLEU, ROUGE).
- **Alternatives**: Promptfoo, Ragas (good libraries, but custom prompts provide transparency, control over judges, and zero black-box dependencies).

## Config/Env Vars
- `GOLDEN_DATASET_PATH` (Default: `./data/evaluation/golden_dataset.json`)
- `EVAL_JUDGE_PROVIDER` (Default: `openai`)
- `EVAL_JUDGE_MODEL` (Default: `gpt-4o`)
- `EVAL_MIN_THRESHOLD` (Default: `0.80` - used to block regression PRs)

## Explicit Non-Goals
- Generating database embeddings (Stage 02) or scraping files (Stage 00).
- Building visual metrics charts (delegated to Streamlit Dashboard in Stage 09).
- Persistent raw storage of execution steps (handled in Stage 08).

## Acceptance Criteria
- Running the evaluation script produces a run report file JSON containing scores for all golden queries.
- LLM-as-judge returns float values between `0.0` and `1.0` using structured JSON format constraints.
- Evaluation harness can be run against all three retrieval strategies (V1, V2, V3) interchangeably.
- If a judge API call fails, the runner retry mechanism operates automatically, failing gracefully if retries are exhausted.
- Unit tests verify correctness of ROUGE/BLEU computation logic using deterministic sample text.
