# Metadata Schema - Cancer Clinical AI Evaluation Platform

## Patient Demographics Registry
To evaluate performance variations across patient groups, demographic records are tracked separately from clinical chunk vectors:

```json
{
  "demographics": {
    "age": 62,
    "gender": "Male",
    "smoking_status": "Former Smoker",
    "alcohol_status": "Occasional",
    "family_history": "Yes",
    "bmi": 24.5,
    "country": "US",
    "ethnicity": "Non-Hispanic White",
    "genetic_mutation": "EGFR L858R",
    "cancer_stage": "Stage IV"
  }
}
```

## Guidelines Source Tracking Schema
Every ingested document lists strict source lineage to support audit checks:

```json
{
  "guidelines_metadata": {
    "source_file": "lung_cancer_nccn_2026.txt",
    "author_institution": "National Comprehensive Cancer Network (NCCN)",
    "publication_year": 2026,
    "therapeutic_area": "Solid Tumor Oncology",
    "last_reviewed_at": "2026-03-12"
  }
}
```
 Enables slicing metrics by patient characteristics (e.g. comparing accuracy for Stage IV cohorts vs. Stage I cohorts).
