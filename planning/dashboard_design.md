# Dashboard Design - Cancer Clinical AI Evaluation Platform

## Streamlit Page Architecture (11 Pages)

1. **Executive Summary**: Overall Clinical Quality Score, deployment status flags (PASS/FAIL), average cost, latency, and faithfulness metrics.
2. **Experiment Comparison**: Multi-axis charts plotting metrics (correctness, MAP, cost) across different embedding models, chunk size, and retriever weights settings.
3. **Question Explorer**: Grid listing test questions, retrieved chunks side-by-side, answer completions, expected ground truth, and judge reasoning.
4. **Retrieval Analytics**: Context recall score distributions, top-retrieved document frequencies, and similarity score metrics.
5. **Cancer Analytics**: Histograms grouping correctness and MAP scores by Cancer Type, ICD Code, and Stage.
6. **NER Dashboard**: Tag clouds and frequency tables listing most common symptoms, procedures, and drugs extracted from queries.
7. **ICD Dashboard**: Distribution charts of ICD codes, mapping failure counts to help pinpoint code gaps.
8. **Demographic Dashboard**: Performance comparison charts slicing clinical utility by age groups, gender, and smoking status.
9. **Langfuse Dashboard**: Integration trace table displaying tokens, latency, cost, and spans breakdowns per request.
10. **Error Analysis**: Isolation dashboard highlighting hallucination runs, retrieval misses, and missing context flags.
11. **Deployment Center**: Gate checks summary, showing regression delta logs and status warnings, with one-click approval buttons.
