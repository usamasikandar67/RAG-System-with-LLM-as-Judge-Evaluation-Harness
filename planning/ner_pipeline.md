# NER Pipeline - Cancer Clinical AI Evaluation Platform

## SciSpaCy Integration
The platform integrates the specialized biomedical parser model `en_core_sci_lg` (or `en_ner_bc5cdr_md`) to isolate medical entities from clinical notes.

## Entity Mapping Dictionary

| Entity Class | Description | Example Extraction |
|---|---|---|
| **Cancer Type** | Pathology name | Non-Small Cell Lung Cancer |
| **Primary Site** | Anatomical location | Lung, Breast, Colon |
| **Cancer Stage** | Severity level classification | Stage IV, Stage IIB |
| **Tumor Grade** | Differentiation grade | Grade 3 (Undifferentiated) |
| **Biomarkers** | Genomic mutations / indicators | EGFR, ALK, HER2, PD-L1 |
| **Chemotherapy Drug** | Chemotherapeutic agents | cisplatin, osimertinib, paclitaxel |
| **Procedure** | Medical surgeries or checks | Lobectomy, CT scan, Radiotherapy |
| **Dosage** | Medication strength | 80mg, 500mg/m2 |
| **Frequency** | Timing protocols | Daily, q3w (every 3 weeks) |

## Extraction Workflow
1. Raw document strings pass to the pipeline.
2. SciSpaCy resolves tokens and flags entities.
3. Regex rule matching cleans dose intervals andステージ labels.
4. Cleaned tokens are appended to chunk metadata dictionaries to populate filters.
