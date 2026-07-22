# src/icd_mapping/icd_resolver.py

from typing import Dict, Any

ICD_10_DICTIONARY = {
    "breast cancer": {"code": "C50", "category": "Breast Neoplasm", "description": "Malignant neoplasm of breast"},
    "lung cancer": {"code": "C34", "category": "Thoracic Oncology", "description": "Malignant neoplasm of bronchus and lung"},
    "non-small cell lung cancer": {"code": "C34", "category": "Thoracic Oncology", "description": "Malignant neoplasm of bronchus and lung"},
    "nsclc": {"code": "C34", "category": "Thoracic Oncology", "description": "Malignant neoplasm of bronchus and lung"},
    "colon cancer": {"code": "C18", "category": "Gastrointestinal", "description": "Malignant neoplasm of colon"},
    "prostate cancer": {"code": "C61", "category": "Genitourinary", "description": "Malignant neoplasm of prostate"},
    "leukemia": {"code": "C91", "category": "Hematologic", "description": "Lymphoid leukemia"},
    "lymphoma": {"code": "C81", "category": "Hematologic", "description": "Hodgkin lymphoma"},
    "plasmacytoma": {"code": "C90", "category": "Hematologic", "description": "Multiple myeloma and malignant plasma cell neoplasms"},
    "bone lesion": {"code": "C90", "category": "Hematologic", "description": "Multiple myeloma and malignant plasma cell neoplasms"},
    "melanoma": {"code": "C43", "category": "Dermatologic", "description": "Malignant melanoma of skin"},
    "pancreatic cancer": {"code": "C25", "category": "Gastrointestinal", "description": "Malignant neoplasm of pancreas"},
    "ovarian cancer": {"code": "C56", "category": "Gynecologic", "description": "Malignant neoplasm of ovary"},
    "diabetes": {"code": "E11", "category": "Endocrine", "description": "Type 2 diabetes mellitus"},
    "hypertension": {"code": "I10", "category": "Cardiovascular", "description": "Essential (primary) hypertension"},
    "asthma": {"code": "J45", "category": "Respiratory", "description": "Asthma"},
    "copd": {"code": "J44", "category": "Respiratory", "description": "Other chronic obstructive pulmonary disease"},
    "stroke": {"code": "I63", "category": "Neurological", "description": "Cerebral infarction"},
    "myocardial infarction": {"code": "I21", "category": "Cardiovascular", "description": "Acute myocardial infarction"},
    "heart failure": {"code": "I50", "category": "Cardiovascular", "description": "Heart failure"}
}

def resolve_icd_code(cancer_type: str) -> Dict[str, str]:
    cleaned = cancer_type.strip().lower()
    
    # Try exact match
    if cleaned in ICD_10_DICTIONARY:
        return ICD_10_DICTIONARY[cleaned]
        
    # Try substring match
    for key, data in ICD_10_DICTIONARY.items():
        if key in cleaned or cleaned in key:
            return data
            
    # Fallback default
    return {
        "code": "R69",
        "category": "Unknown Medical Condition",
        "description": "Illness, unspecified"
    }
