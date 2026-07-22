# src/complaint/complaint_extractor.py

import re
from typing import Dict, Any, List


class ComplaintExtractor:
    """
    Patient chief complaint parser extracting presenting symptoms,
    body sites, pain severity scales, and duration descriptors from clinical text.
    """

    # Symptom vocabulary
    SYMPTOMS = [
        "pain", "ache", "discomfort", "tenderness", "swelling", "edema",
        "fever", "chills", "fatigue", "weakness", "dizziness", "nausea",
        "vomiting", "diarrhea", "constipation", "bleeding", "hemorrhage",
        "shortness of breath", "dyspnea", "cough", "wheezing", "palpitations",
        "headache", "migraine", "rash", "itching", "pruritus", "numbness",
        "tingling", "paresthesia", "stiffness", "cramping", "burning",
        "difficulty swallowing", "dysphagia", "weight loss", "weight gain",
        "insomnia", "anxiety", "depression", "confusion", "syncope",
        "chest pain", "abdominal pain", "back pain", "joint pain",
        "sore throat", "nasal congestion", "ear pain", "blurred vision",
        "urinary frequency", "dysuria", "hematuria", "loss of appetite"
    ]

    # Body sites
    BODY_SITES = [
        "head", "neck", "chest", "abdomen", "back", "spine", "lumbar",
        "thoracic", "cervical", "shoulder", "arm", "elbow", "wrist",
        "hand", "finger", "hip", "thigh", "knee", "leg", "ankle",
        "foot", "toe", "pelvis", "groin", "flank", "epigastric",
        "right upper quadrant", "left upper quadrant", "right lower quadrant",
        "left lower quadrant", "breast", "axilla", "throat", "ear", "eye"
    ]

    # Pain scale pattern (e.g., "7/10", "pain rated 8 out of 10")
    PAIN_SCALE_PATTERNS = [
        r"(\d{1,2})\s*/\s*10",
        r"pain\s*(?:rated?|scale|score|level)\s*(?:of\s*)?(\d{1,2})",
        r"(\d{1,2})\s*(?:out of|of)\s*10"
    ]

    # Duration patterns
    DURATION_PATTERNS = [
        r"(\d+)\s*(days?|weeks?|months?|hours?|hrs?|minutes?|mins?|years?|yrs?)\s*(?:of|ago|duration|history)",
        r"(?:for|since|x|times?)\s*(\d+)\s*(days?|weeks?|months?|hours?|hrs?|years?|yrs?)",
        r"(?:onset|started|began|presenting)\s*(\d+)\s*(days?|weeks?|months?|hours?|years?)\s*ago",
    ]

    # Severity qualifiers
    SEVERITY_TERMS = {
        "severe": ["severe", "intense", "excruciating", "unbearable", "worst", "acute", "sharp"],
        "moderate": ["moderate", "significant", "considerable", "persistent", "constant"],
        "mild": ["mild", "slight", "minor", "dull", "intermittent", "occasional", "subtle"]
    }

    def extract(self, text: str) -> Dict[str, Any]:
        """Extract chief complaints from clinical text."""
        text_lower = text.lower()
        complaints: List[Dict[str, Any]] = []

        # 1. Find symptoms
        found_symptoms = []
        for symptom in self.SYMPTOMS:
            if symptom in text_lower:
                found_symptoms.append(symptom)

        # 2. Find body sites
        found_sites = []
        for site in self.BODY_SITES:
            if site in text_lower:
                found_sites.append(site)

        # 3. Extract pain scale
        pain_score = None
        for pattern in self.PAIN_SCALE_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                score = int(match.group(1))
                if 0 <= score <= 10:
                    pain_score = score
                    break

        # 4. Extract duration
        duration_str = None
        for pattern in self.DURATION_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                groups = match.groups()
                if len(groups) >= 2:
                    duration_str = f"{groups[0]} {groups[1]}"
                break

        # 5. Determine severity
        severity = "unspecified"
        for level, terms in self.SEVERITY_TERMS.items():
            for term in terms:
                if term in text_lower:
                    severity = level
                    break
            if severity != "unspecified":
                break

        # If pain score available, override severity
        if pain_score is not None:
            if pain_score >= 7:
                severity = "severe"
            elif pain_score >= 4:
                severity = "moderate"
            elif pain_score >= 1:
                severity = "mild"

        # Build complaint objects by pairing symptoms with sites
        if found_symptoms:
            for symptom in found_symptoms:
                complaint = {
                    "symptom": symptom,
                    "body_site": self._match_site_to_symptom(symptom, found_sites, text_lower),
                    "severity": severity,
                    "duration": duration_str,
                    "pain_score": pain_score
                }
                complaints.append(complaint)
        elif found_sites:
            # Have sites but no explicit symptoms
            complaints.append({
                "symptom": "unspecified complaint",
                "body_site": found_sites[0] if found_sites else None,
                "severity": severity,
                "duration": duration_str,
                "pain_score": pain_score
            })

        return {
            "complaint_count": len(complaints),
            "complaints": complaints,
            "all_symptoms": found_symptoms,
            "all_body_sites": found_sites,
            "overall_severity": severity,
            "pain_score": pain_score,
            "duration": duration_str
        }

    def _match_site_to_symptom(self, symptom: str, sites: List[str], text_lower: str) -> str:
        """Try to find the body site closest to the symptom mention in text."""
        # Check if symptom itself contains a body site (e.g., "chest pain")
        for site in sites:
            if site in symptom:
                return site

        # Check for adjacent mentions (e.g., "pain in the chest")
        symptom_pos = text_lower.find(symptom)
        if symptom_pos >= 0:
            context_window = text_lower[max(0, symptom_pos - 40):symptom_pos + len(symptom) + 40]
            for site in sites:
                if site in context_window:
                    return site

        return sites[0] if sites else "unspecified"
