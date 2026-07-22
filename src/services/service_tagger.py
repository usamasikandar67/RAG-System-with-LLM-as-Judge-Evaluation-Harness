# src/services/service_tagger.py

import re
from typing import Dict, Any, List


class ServiceTagger:
    """
    Medical services classifier mapping clinical text to healthcare
    department/service-line categories.
    """

    SERVICE_DEFINITIONS = {
        "Oncology": {
            "keywords": {"cancer", "tumor", "neoplasm", "chemotherapy", "radiation therapy",
                         "malignant", "metastasis", "biopsy", "carcinoma", "lymphoma",
                         "leukemia", "sarcoma", "oncology", "staging", "remission",
                         "palliative", "immunotherapy", "targeted therapy"},
            "weight": 1.0
        },
        "Radiology": {
            "keywords": {"x-ray", "ct scan", "mri", "ultrasound", "imaging",
                         "radiograph", "fluoroscopy", "mammogram", "pet scan",
                         "contrast", "radiology", "nuclear medicine", "angiography"},
            "weight": 1.0
        },
        "Surgery": {
            "keywords": {"surgery", "surgical", "operation", "resection", "excision",
                         "laparoscopic", "open surgery", "incision", "suture",
                         "anesthesia", "post-operative", "pre-operative",
                         "lobectomy", "mastectomy", "colectomy", "appendectomy"},
            "weight": 1.0
        },
        "Pathology": {
            "keywords": {"pathology", "histology", "cytology", "biopsy results",
                         "tissue sample", "microscopic", "staining", "cell morphology",
                         "frozen section", "immunohistochemistry", "histopathology"},
            "weight": 1.0
        },
        "Cardiology": {
            "keywords": {"cardiac", "heart", "coronary", "arrhythmia", "ecg", "ekg",
                         "echocardiogram", "stent", "myocardial", "angina",
                         "pacemaker", "valve", "aortic", "cardiovascular"},
            "weight": 1.0
        },
        "Neurology": {
            "keywords": {"brain", "neurological", "seizure", "stroke", "dementia",
                         "neuropathy", "epilepsy", "migraine", "cerebral",
                         "parkinson", "alzheimer", "spinal cord", "eeg"},
            "weight": 1.0
        },
        "Emergency Medicine": {
            "keywords": {"emergency", "trauma", "triage", "acute", "critical",
                         "resuscitation", "stabilize", "code blue", "er", "ed",
                         "crash cart", "life-threatening", "hemorrhage"},
            "weight": 1.0
        },
        "Obstetrics & Gynecology": {
            "keywords": {"pregnancy", "prenatal", "gestational", "obstetric", "delivery",
                         "trimester", "fetal", "maternal", "labor", "postpartum",
                         "gynecology", "cervical", "uterine", "ovarian", "pap smear"},
            "weight": 1.0
        },
        "Internal Medicine": {
            "keywords": {"diagnosis", "treatment", "chronic", "diabetes", "hypertension",
                         "infection", "antibiotic", "medication", "primary care",
                         "internal medicine", "outpatient", "inpatient"},
            "weight": 0.8  # slightly lower to avoid over-matching generic terms
        },
        "Pharmacy": {
            "keywords": {"drug", "medication", "prescription", "dosage", "dose",
                         "pharmaceutical", "formulary", "dispensing", "contraindication",
                         "drug interaction", "pharmacy", "mg", "mcg"},
            "weight": 0.9
        }
    }

    def tag(self, text: str) -> Dict[str, Any]:
        """Tag text with healthcare service-line categories."""
        text_lower = text.lower()
        tokens = set(re.findall(r"\w+", text_lower))

        service_scores: List[Dict[str, Any]] = []

        for service_name, definition in self.SERVICE_DEFINITIONS.items():
            keywords = definition["keywords"]
            weight = definition["weight"]

            # Single-word keyword matches
            single_word_hits = tokens.intersection({kw for kw in keywords if " " not in kw})

            # Multi-word phrase matches
            phrase_hits = 0
            for kw in keywords:
                if " " in kw and kw in text_lower:
                    phrase_hits += 1

            total_hits = len(single_word_hits) + (phrase_hits * 2)  # phrases weighted double
            score = (total_hits / max(len(keywords), 1)) * weight

            if score > 0:
                service_scores.append({
                    "service": service_name,
                    "score": round(float(score), 4),
                    "matched_terms": list(single_word_hits)[:5]  # cap for readability
                })

        # Sort by score descending
        service_scores.sort(key=lambda x: x["score"], reverse=True)

        if not service_scores:
            return {
                "primary_service": "General Medicine",
                "confidence": 0.2,
                "all_services": []
            }

        primary = service_scores[0]
        total_score = sum(s["score"] for s in service_scores)
        confidence = primary["score"] / total_score if total_score > 0 else 0.3

        return {
            "primary_service": primary["service"],
            "confidence": round(float(confidence), 3),
            "all_services": service_scores[:5]  # top 5
        }
