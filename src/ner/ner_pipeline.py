# src/ner/ner_pipeline.py

import re
from typing import Dict, Any, List

# Medical entity thesaurus
CANCER_PATTERNS = [
    r"non-small cell lung cancer", r"nsclc", r"breast cancer", r"colon cancer", 
    r"prostate cancer", r"leukemia", r"lymphoma", r"plasmacytoma", r"melanoma",
    r"bone lesion", r"lung cancer", r"pancreatic cancer", r"ovarian cancer"
]

DISEASE_PATTERNS = [
    r"diabetes", r"hypertension", r"asthma", r"copd", r"stroke", r"myocardial infarction", r"heart failure"
]

BIOMARKER_PATTERNS = [
    r"\begfr\b", r"\balk\b", r"\bher2\b", r"\bpd-l1\b", r"\bros1\b", r"\bbraf\b", r"\bbrca1\b", r"\bbrca2\b"
]

DRUG_PATTERNS = [
    r"osimertinib", r"cisplatin", r"paclitaxel", r"rituximab", r"carboplatin", 
    r"pembrolizumab", r"chemotherapy", r"tamoxifen", r"trastuzumab"
]

PROCEDURE_PATTERNS = [
    r"radiation therapy", r"radiotherapy", r"lobectomy", r"biopsy", r"surgery", r"ct scan", r"mri", r"mastectomy"
]

SYMPTOM_PATTERNS = [
    r"pain", r"fatigue", r"nausea", r"vomiting", r"fever", r"cough", r"shortness of breath", r"headache"
]

DOSAGE_PATTERN = r"\b\d+(?:\.\d+)?\s*(?:mg/m2|mg|g|mcg)\b"
STAGE_PATTERN = r"\b(?:stage)\s*([ivx]+|[0-4])\b"
AGE_PATTERN = r"\b(\d{1,3})\s*(?:year(?:s)?\s*old|yo|y\.o\.)\b"
GENDER_PATTERN = r"\b(male|female|man|woman|boy|girl)\b"
ICD_PATTERN = r"\b[A-Z]\d{2}(?:\.\d{1,4})?\b" # e.g. C50.911

def extract_clinical_entities(text: str) -> Dict[str, List[str]]:
    text_lower = text.lower()
    entities = {
        "diseases": [],
        "cancer_types": [],
        "stage": [],
        "biomarkers": [],
        "drugs": [],
        "procedures": [],
        "dosages": [],
        "symptoms": [],
        "age": [],
        "gender": [],
        "icd_codes": []
    }
    
    # Extract cancers
    for pattern in CANCER_PATTERNS:
        matches = re.findall(pattern, text_lower)
        if matches:
            name = pattern.replace(r"\b", "").strip().title()
            if name == "Nsclc": name = "Non-Small Cell Lung Cancer"
            if name not in entities["cancer_types"]: entities["cancer_types"].append(name)
            
    # Extract diseases
    for pattern in DISEASE_PATTERNS:
        matches = re.findall(pattern, text_lower)
        if matches:
            name = pattern.replace(r"\b", "").strip().title()
            if name not in entities["diseases"]: entities["diseases"].append(name)
                
    # Extract biomarkers/genes
    for pattern in BIOMARKER_PATTERNS:
        matches = re.findall(pattern, text_lower)
        if matches:
            name = pattern.replace(r"\b", "").upper()
            if name not in entities["biomarkers"]: entities["biomarkers"].append(name)
                
    # Extract drugs
    for pattern in DRUG_PATTERNS:
        matches = re.findall(pattern, text_lower)
        if matches:
            name = pattern.strip().title()
            if name not in entities["drugs"]: entities["drugs"].append(name)
                
    # Extract procedures
    for pattern in PROCEDURE_PATTERNS:
        matches = re.findall(pattern, text_lower)
        if matches:
            name = pattern.strip().title()
            if name not in entities["procedures"]: entities["procedures"].append(name)
            
    # Extract symptoms
    for pattern in SYMPTOM_PATTERNS:
        matches = re.findall(pattern, text_lower)
        if matches:
            name = pattern.strip().title()
            if name not in entities["symptoms"]: entities["symptoms"].append(name)
                
    # Regex Extractions
    for dose in re.findall(DOSAGE_PATTERN, text_lower):
        if dose.upper() not in entities["dosages"]: entities["dosages"].append(dose.upper())
        
    for stage in re.findall(STAGE_PATTERN, text_lower):
        val = f"Stage {stage.upper()}"
        if val not in entities["stage"]: entities["stage"].append(val)
        
    for age in re.findall(AGE_PATTERN, text_lower):
        if age not in entities["age"]: entities["age"].append(age)
        
    for gender in re.findall(GENDER_PATTERN, text_lower):
        val = gender.title()
        if val in ["Man", "Boy"]: val = "Male"
        if val in ["Woman", "Girl"]: val = "Female"
        if val not in entities["gender"]: entities["gender"].append(val)
        
    for icd in re.findall(ICD_PATTERN, text):
        if icd not in entities["icd_codes"]: entities["icd_codes"].append(icd)
            
    return entities
