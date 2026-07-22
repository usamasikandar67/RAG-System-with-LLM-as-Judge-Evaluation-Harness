# Milestones - Cancer Clinical AI Evaluation Platform

## Project Quality KPIs
The platform verifies system metrics against strict target values before deployment:

| Metric | Target Value | Hard CI Gate Limit |
|---|---|---|
| **Faithfulness** | $\ge 0.95$ | Fails if $< 0.90$ |
| **Correctness** | $\ge 0.95$ | Fails if $< 0.90$ |
| **Hits@3** | $\ge 0.85$ | Fails if $< 0.80$ |
| **MAP** | $\ge 0.85$ | Fails if $< 0.80$ |
| **Citation Accuracy** | $= 1.00$ (100% cited) | Fails if $< 1.00$ |
| **Latency** | $\le 2.0\text{s}$ | Warning if $> 3.5\text{s}$ |

## Phase Delivery Targets
- **Milestone 1**: Restructured codebase (`src/` configuration) and full design templates complete (Current Phase).
- **Milestone 2**: BM25 & Hybrid retriever with RRF scoring fully active.
- **Milestone 3**: SciSpaCy NER and ICD-10 coders integrated.
- **Milestone 4**: Langfuse tracking and tracing active.
- **Milestone 5**: Streamlit analytics interface complete.
- **Milestone 6**: Automated PR Gate verification blocks in CI branch merges.
