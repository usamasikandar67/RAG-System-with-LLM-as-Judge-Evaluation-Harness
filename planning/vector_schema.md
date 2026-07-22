# Vector Schema - Cancer Clinical AI Evaluation Platform

## Index Payload Structure

```json
{
  "id": "chunk_cancer_doc_0_0_abcd1234efgh",
  "vector": [0.0125, -0.0456, "...", 0.0891],
  "metadata": {
    "source_file": "cancer_doc_0.txt",
    "clinical_category": "Oncology",
    "chunk_index": 0,
    "char_length": 780,
    "cancer_type": "Non-Small Cell Lung Cancer",
    "icd_10_code": "C34",
    "biomarkers": ["EGFR", "ALK"],
    "drugs": ["osimertinib", "crizotinib"],
    "demographics": {
      "min_age": 18,
      "gender_restriction": "None",
      "stage": "Stage IV"
    }
  }
}
```

## Database Settings
- **Vector Dimension**: 1536 (standard for OpenAI `text-embedding-3-small` and `text-embedding-ada-002`).
- **Distance Metric**: Cosine Similarity.
- **Index Type**: Pluggable support:
  - Prototype: HNSW / Flat Index in-memory NumPy.
  - Scale: HNSW index configured in Qdrant or Pinecone.
- **Metadata Filtering**: Enables exact boolean matches on `icd_10_code` or `cancer_type` before vector proximity calculations to filter noise.
