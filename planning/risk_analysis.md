# Risk Analysis - Cancer Clinical AI Evaluation Platform

## Identified Clinical and Operational Risks

### 1. Hallucinations & Wrong Dosages
- **Risk**: RAG assistant generates an incorrect drug dose.
- **Mitigation**: LLM-as-Judge strictly grades correctness and completeness. If correctness drops below $0.95$ vs baseline, the CI build fails and blocks the update.

### 2. Patient Data Leakage (HIPAA)
- **Risk**: Sending protected patient health data to public API models during trace evaluations.
- **Mitigation**: Clinical identifiers (names, IDs, addresses) are scrubbed at the query boundaries before retrieval or embeddings models execution. All SQLite logs are kept in localized, secure hospital database systems.

### 3. Model/Drift Silent Changes
- **Risk**: API providers upgrade model weights (e.g. GPT-4o update), silently breaking extraction rules.
- **Mitigation**: Configure nightly evaluation runs (crons) executing the full golden test dataset to check for sudden quality score drops.
