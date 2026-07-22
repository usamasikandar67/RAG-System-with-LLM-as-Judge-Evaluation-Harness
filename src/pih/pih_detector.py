# src/pih/pih_detector.py

import re
from typing import Dict, Any, List


class PIHRiskScorer:
    """
    Pregnancy-Induced Hypertension (PIH) risk scoring engine.
    
    Identifies clinical indicators of gestational hypertension,
    preeclampsia, and eclampsia from clinical notes and lab results.
    """

    # PIH condition terms
    CONDITION_TERMS = [
        r"preeclampsia", r"pre-eclampsia", r"eclampsia",
        r"gestational hypertension", r"pregnancy.induced hypertension",
        r"pih\b", r"hellp syndrome", r"toxemia of pregnancy",
        r"hypertensive disorder of pregnancy"
    ]

    # Symptom indicators
    SYMPTOM_PATTERNS = [
        (r"headache", "headache", 1),
        (r"visual disturbance|blurred vision|scotomata", "visual_disturbance", 2),
        (r"epigastric pain|right upper quadrant pain|ruq pain", "epigastric_pain", 2),
        (r"edema|swelling|oedema", "edema", 1),
        (r"nausea|vomiting", "nausea_vomiting", 1),
        (r"hyperreflexia|clonus", "hyperreflexia", 2),
        (r"oliguria|decreased urine", "oliguria", 2),
        (r"seizure|convulsion", "seizure", 3),
    ]

    # Lab value patterns
    LAB_PATTERNS = [
        (r"proteinuria|protein\s*(?:in\s*)?urine|urine protein", "proteinuria", 2),
        (r"platelets?\s*(?:<|less than|below)\s*(\d+)", "low_platelets", 2),
        (r"(?:ast|alt|liver enzymes?)\s*(?:elevated|raised|high|>)", "elevated_liver_enzymes", 2),
        (r"creatinine\s*(?:elevated|raised|high|>)", "elevated_creatinine", 1),
    ]

    # Blood pressure extraction
    BP_PATTERN = r"(?:bp|blood pressure)[:\s]*(\d{2,3})\s*/\s*(\d{2,3})"

    # Gestational age
    GESTATIONAL_PATTERN = r"(\d{1,2})\s*(?:weeks?|wks?)\s*(?:gestation|gestational|pregnant|ga)"

    def score(self, text: str) -> Dict[str, Any]:
        """Score clinical text for PIH risk indicators."""
        text_lower = text.lower()
        indicators: List[Dict[str, Any]] = []
        total_score = 0.0

        # 1. Check for direct condition mentions
        for pattern in self.CONDITION_TERMS:
            if re.search(pattern, text_lower):
                indicators.append({
                    "type": "condition",
                    "term": pattern.replace(r"\b", "").replace(".", " ").strip(),
                    "weight": 5
                })
                total_score += 5

        # 2. Check symptom patterns
        for pattern, name, weight in self.SYMPTOM_PATTERNS:
            if re.search(pattern, text_lower):
                indicators.append({
                    "type": "symptom",
                    "term": name,
                    "weight": weight
                })
                total_score += weight

        # 3. Check lab patterns
        for pattern, name, weight in self.LAB_PATTERNS:
            if re.search(pattern, text_lower):
                indicators.append({
                    "type": "lab_finding",
                    "term": name,
                    "weight": weight
                })
                total_score += weight

        # 4. Extract blood pressure readings
        bp_matches = re.findall(self.BP_PATTERN, text_lower)
        for systolic_str, diastolic_str in bp_matches:
            systolic = int(systolic_str)
            diastolic = int(diastolic_str)
            if systolic >= 160 or diastolic >= 110:
                indicators.append({
                    "type": "vital_sign",
                    "term": f"severe_hypertension_bp_{systolic}/{diastolic}",
                    "weight": 4
                })
                total_score += 4
            elif systolic >= 140 or diastolic >= 90:
                indicators.append({
                    "type": "vital_sign",
                    "term": f"hypertension_bp_{systolic}/{diastolic}",
                    "weight": 3
                })
                total_score += 3

        # 5. Extract gestational age
        ga_matches = re.findall(self.GESTATIONAL_PATTERN, text_lower)
        gestational_weeks = None
        for ga in ga_matches:
            weeks = int(ga)
            gestational_weeks = weeks
            if weeks >= 20:
                indicators.append({
                    "type": "gestational_age",
                    "term": f"{weeks}_weeks_gestation",
                    "weight": 1
                })
                total_score += 1

        # Determine risk level
        if total_score >= 8:
            risk_level = "high"
        elif total_score >= 4:
            risk_level = "moderate"
        elif total_score >= 1:
            risk_level = "low"
        else:
            risk_level = "none"

        return {
            "risk_level": risk_level,
            "score": round(float(total_score), 2),
            "indicators": indicators,
            "indicator_count": len(indicators),
            "gestational_weeks": gestational_weeks
        }
