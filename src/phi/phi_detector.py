# src/phi/phi_detector.py

import re
from typing import Dict, Any, List


class PHIDetector:
    """
    Protected Health Information (PHI) detector using regex patterns
    to identify and optionally redact sensitive patient data per HIPAA guidelines.
    
    Detects: SSN, MRN, phone numbers, email addresses, dates of birth,
    and common patient name patterns.
    """

    PATTERNS = {
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        "MRN": r"\b(?:MRN|mrn|Medical Record Number)[:\s#]*(\d{6,10})\b",
        "HOSPITAL_ID": r"\b(?:HID|Hospital ID|Hosp ID)[:\s#]*([A-Z0-9]{5,12})\b",
        "PHONE": r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "DOB": r"\b(?:DOB|Date of Birth|D\.O\.B\.?)[:\s]*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\b",
        "DATE": r"\b(?:0?[1-9]|1[0-2])[/\-](?:0?[1-9]|[12]\d|3[01])[/\-](?:19|20)\d{2}\b",
        "AGE": r"\b(?:age|aged)[:\s]*(\d{1,3})\s*(?:years?|yrs?|y/?o)?\b",
        "CREDIT_CARD": r"\b(?:\d[ -]*?){13,16}\b",
        "PASSPORT": r"\b[A-Z]{1,2}[0-9]{7}\b",
        "ADDRESS": r"\b\d+\s+[A-Za-z0-9\s.,-]+\s+(?:Avenue|Lane|Road|Boulevard|Drive|Street|Ave|Dr|Rd|Blvd|Ln|St)\b(?:.*?\b\d{5}(?:-\d{4})?\b)?"
    }

    # Common name prefixes that often precede patient names
    NAME_PREFIXES = [
        r"\b(?:patient|pt|name)[:\s]+([A-Z][a-z]+\s+[A-Z][a-z]+)",
        r"\b(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
    ]

    def detect(self, text: str) -> Dict[str, Any]:
        """Scan text for PHI and return findings."""
        findings: List[Dict[str, Any]] = []

        for phi_type, pattern in self.PATTERNS.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                findings.append({
                    "type": phi_type,
                    "value": match.group(0),
                    "position": match.start(),
                    "end_position": match.end()
                })

        # Name detection
        for pattern in self.NAME_PREFIXES:
            for match in re.finditer(pattern, text):
                findings.append({
                    "type": "PATIENT_NAME",
                    "value": match.group(0),
                    "position": match.start(),
                    "end_position": match.end()
                })

        # Deduplicate by position
        seen_positions = set()
        unique_findings = []
        for f in findings:
            key = (f["position"], f["end_position"])
            if key not in seen_positions:
                seen_positions.add(key)
                unique_findings.append(f)

        return {
            "has_phi": len(unique_findings) > 0,
            "phi_count": len(unique_findings),
            "findings": sorted(unique_findings, key=lambda x: x["position"]),
            "risk_level": self._assess_risk(unique_findings)
        }

    def redact(self, text: str) -> str:
        """Replace all detected PHI with [REDACTED] tokens."""
        result = detect_result = self.detect(text)
        redacted = text

        # Replace in reverse order to preserve positions
        for finding in reversed(result["findings"]):
            start = finding["position"]
            end = finding["end_position"]
            redacted = redacted[:start] + "[REDACTED]" + redacted[end:]

        return redacted

    def _assess_risk(self, findings: List[Dict]) -> str:
        """Assess overall PHI risk level based on findings."""
        if not findings:
            return "none"

        types_found = {f["type"] for f in findings}
        critical_types = {"SSN", "MRN", "PATIENT_NAME"}

        if types_found.intersection(critical_types):
            return "high"
        elif len(findings) >= 3:
            return "high"
        elif len(findings) >= 1:
            return "moderate"
        return "low"
