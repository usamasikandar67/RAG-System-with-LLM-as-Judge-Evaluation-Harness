# tests/verify_improvements.py

import unittest
# pyrefly: ignore [missing-import]
from ner.ner_pipeline import extract_clinical_entities
# pyrefly: ignore [missing-import]
from icd_mapping.icd_resolver import resolve_icd_code
# pyrefly: ignore [missing-import]
from reranking.reranker import SimpleCrossEncoderReranker

class TestImprovements(unittest.TestCase):
    
    def test_clinical_ner_extraction(self):
        text = "Recommended treatment for NSCLC with EGFR mutation is osimertinib 80mg daily."
        entities = extract_clinical_entities(text)
        
        self.assertIn("Non-Small Cell Lung Cancer", entities["cancer_types"])
        self.assertIn("EGFR", entities["biomarkers"])
        self.assertIn("Osimertinib", entities["drugs"])
        self.assertIn("80MG", entities["dosages"])
        print("[Pass] Clinical NER extraction verified successfully.")

    def test_icd_code_resolution(self):
        # 1. Exact match
        icd_breast = resolve_icd_code("Breast Cancer")
        self.assertEqual(icd_breast["code"], "C50")
        
        # 2. Substring match
        icd_lung = resolve_icd_code("advanced NSCLC lesion")
        self.assertEqual(icd_lung["code"], "C34")
        
        # 3. Fallback
        icd_fallback = resolve_icd_code("some rare growth")
        self.assertEqual(icd_fallback["code"], "R69")
        print("[Pass] ICD-10 code resolution verified successfully.")

    def test_cross_encoder_reranker(self):
        query = "What is radiation therapy dose for isolated plasmacytoma?"
        docs = [
            {"text": "Chemotherapy cisplatin protocols for lung neoplasm.", "score": 1.0},
            {"text": "Treatment of isolated plasmacytoma of bone is usually radiation therapy.", "score": 0.5}
        ]
        
        reranker = SimpleCrossEncoderReranker()
        reranked = reranker.rerank(query, docs, top_n=2)
        
        self.assertEqual(len(reranked), 2)
        # The document referencing radiation and plasmacytoma must rank first due to higher token overlap score
        self.assertTrue("radiation therapy" in reranked[0]["text"])
        print("[Pass] Cross-Encoder reranker execution verified successfully.")

if __name__ == "__main__":
    unittest.main()
