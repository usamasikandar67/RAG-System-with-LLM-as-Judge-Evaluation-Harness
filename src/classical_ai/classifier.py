# src/classical_ai/classifier.py

import re
import math
from typing import Dict, Any, List
from collections import Counter


class ClinicalTextClassifier:
    """
    Lightweight clinical text classifier using TF-IDF-style keyword frequency
    heuristics. No external ML dependencies required.
    """

    # Urgency keyword dictionaries with weights
    URGENCY_HIGH = {
        "emergency", "stat", "critical", "life-threatening", "hemorrhage",
        "cardiac arrest", "sepsis", "anaphylaxis", "stroke", "acute",
        "respiratory failure", "code blue", "unstable", "deteriorating",
        "shock", "intubation", "resuscitation", "unresponsive", "seizure"
    }
    URGENCY_MEDIUM = {
        "urgent", "worsening", "abnormal", "elevated", "concerning",
        "moderate", "progressive", "persistent", "recurring", "follow-up",
        "monitor", "observation", "escalation", "referral", "consult"
    }
    URGENCY_LOW = {
        "routine", "stable", "normal", "scheduled", "maintenance",
        "preventive", "screening", "wellness", "check-up", "elective",
        "mild", "benign", "resolved", "chronic stable", "asymptomatic"
    }

    # Topic keyword maps
    TOPIC_KEYWORDS = {
        "Oncology": {"cancer", "tumor", "neoplasm", "chemotherapy", "radiation", "malignant",
                     "metastasis", "biopsy", "carcinoma", "lymphoma", "leukemia", "sarcoma",
                     "oncology", "staging", "remission", "palliative"},
        "Cardiology": {"cardiac", "heart", "coronary", "arrhythmia", "ecg", "stent",
                       "myocardial", "hypertension", "atrial", "ventricular", "angina",
                       "pacemaker", "valve", "aortic"},
        "Neurology": {"brain", "neurological", "seizure", "stroke", "dementia", "neuropathy",
                      "epilepsy", "migraine", "cerebral", "mri brain", "parkinson"},
        "Pulmonology": {"lung", "pulmonary", "respiratory", "bronchial", "asthma", "copd",
                        "pneumonia", "ventilator", "oxygen", "thoracic"},
        "Gastroenterology": {"liver", "hepatic", "gastric", "colon", "intestinal",
                             "endoscopy", "gi", "pancreatic", "cirrhosis", "ulcer"},
        "Obstetrics": {"pregnancy", "prenatal", "gestational", "obstetric", "delivery",
                       "trimester", "fetal", "maternal", "labor", "postpartum"},
        "General Medicine": {"fever", "pain", "infection", "antibiotic", "diagnosis",
                             "treatment", "medication", "symptoms", "examination"}
    }

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def _keyword_score(self, tokens: set, keyword_set: set) -> float:
        hits = tokens.intersection(keyword_set)
        return len(hits) / max(len(keyword_set), 1)

    def classify_urgency(self, text: str) -> Dict[str, Any]:
        """Classify text urgency as high/medium/low with confidence score."""
        tokens = set(self._tokenize(text))
        text_lower = text.lower()

        # Check multi-word phrases too
        high_score = self._keyword_score(tokens, self.URGENCY_HIGH)
        for phrase in ["cardiac arrest", "code blue", "life-threatening", "respiratory failure"]:
            if phrase in text_lower:
                high_score += 0.3

        medium_score = self._keyword_score(tokens, self.URGENCY_MEDIUM)
        low_score = self._keyword_score(tokens, self.URGENCY_LOW)

        scores = {"high": high_score, "medium": medium_score, "low": low_score}
        best = max(scores, key=scores.get)

        # If no signal at all, default to medium
        if all(v == 0.0 for v in scores.values()):
            return {"urgency": "medium", "confidence": 0.3, "scores": scores}

        total = sum(scores.values())
        confidence = scores[best] / total if total > 0 else 0.3

        return {
            "urgency": best,
            "confidence": round(float(confidence), 3),
            "scores": {k: round(v, 4) for k, v in scores.items()}
        }

    def classify_topic(self, text: str) -> Dict[str, Any]:
        """Classify text into a clinical topic/specialty."""
        tokens = set(self._tokenize(text))
        text_lower = text.lower()

        topic_scores = {}
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            score = self._keyword_score(tokens, keywords)
            # Boost multi-word matches
            for kw in keywords:
                if " " in kw and kw in text_lower:
                    score += 0.15
            topic_scores[topic] = score

        best_topic = max(topic_scores, key=topic_scores.get)
        total = sum(topic_scores.values())
        confidence = topic_scores[best_topic] / total if total > 0 else 0.2

        if all(v == 0.0 for v in topic_scores.values()):
            return {"topic": "General Medicine", "confidence": 0.2, "all_topics": topic_scores}

        return {
            "topic": best_topic,
            "confidence": round(float(confidence), 3),
            "all_topics": {k: round(v, 4) for k, v in topic_scores.items()}
        }
