# Future Scope - Cancer Clinical AI Evaluation Platform

## Scale Path to 100,000+ Guidelines
To handle production volume representing millions of guideline chunks across thousands of oncology publications:
- **Distributed Ingestion**: Migrate standard python processing to Apache Spark or Databricks for ingestion.
- **PubMed/NCI Live Feeds**: Connect the document ingest pipelines directly to PubMed APIs and NCI RSS feeds for daily updates.

## Production Vector Databases
Transition from local numpy/SQLite storage to:
- **Qdrant**: Enables rapid metadata filtering combined with vector similarity searches.
- **Azure AI Search**: Integrates semantic ranking systems out-of-the-box.

## Active Learning and RLHF
- **Clinician Feedback Loop**: Implement thumbs-up/thumbs-down review mechanisms directly in clinical assistants to flag hallucinations.
- **RLHF Tuning**: Fine-tune domain models using collected feedback databases to minimize future citation omissions.
