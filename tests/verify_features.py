# tests/verify_features.py

import unittest


class TestChunkingToggle(unittest.TestCase):
    def test_chunking_enabled(self):
        import tempfile
        import os
        from ingestion.ingestion import ingest_documents
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(5):
                with open(os.path.join(tmpdir, f"doc_{i}.txt"), "w") as f:
                    f.write("Clinical protocol guidance text.\n\n" * 50)
            docs = ingest_documents(tmpdir, chunk_size=300, chunk_overlap=30, chunking_enabled=True)
            self.assertTrue(len(docs) > 5, "Chunking enabled should produce multiple chunks")
            self.assertTrue(docs[0]["metadata"]["chunking_enabled"])
            print(f"[Pass] Chunking enabled: {len(docs)} chunks produced.")

    def test_chunking_disabled(self):
        import tempfile
        import os
        from ingestion.ingestion import ingest_documents
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(5):
                with open(os.path.join(tmpdir, f"doc_{i}.txt"), "w") as f:
                    f.write("Clinical protocol guidance text.\n\n" * 50)
            docs = ingest_documents(tmpdir, chunk_size=300, chunk_overlap=30, chunking_enabled=False)
            self.assertEqual(len(docs), 5, f"Chunking disabled should produce exactly 1 chunk per file (5 total), got {len(docs)}")
            self.assertFalse(docs[0]["metadata"]["chunking_enabled"])
            print(f"[Pass] Chunking disabled: {len(docs)} whole-file documents produced.")


class TestClassicalAI(unittest.TestCase):
    def test_urgency_classification(self):
        from classical_ai.classifier import ClinicalTextClassifier
        clf = ClinicalTextClassifier()
        
        result = clf.classify_urgency("Patient in cardiac arrest, code blue, critical condition.")
        self.assertEqual(result["urgency"], "high")
        self.assertGreater(result["confidence"], 0.3)
        
        result_low = clf.classify_urgency("Routine screening appointment for wellness check-up.")
        self.assertEqual(result_low["urgency"], "low")
        print("[Pass] Classical AI urgency classification verified.")

    def test_topic_classification(self):
        from classical_ai.classifier import ClinicalTextClassifier
        clf = ClinicalTextClassifier()
        
        result = clf.classify_topic("Breast cancer chemotherapy treatment with radiation.")
        self.assertEqual(result["topic"], "Oncology")
        
        result2 = clf.classify_topic("Cardiac arrhythmia ECG stent myocardial infarction.")
        self.assertEqual(result2["topic"], "Cardiology")
        print("[Pass] Classical AI topic classification verified.")


class TestPHIDetector(unittest.TestCase):
    def test_phi_detection(self):
        from phi.phi_detector import PHIDetector
        detector = PHIDetector()
        
        text = "Patient name: John Smith, SSN: 123-45-6789, DOB: 01/15/1985, email: john@hospital.com"
        result = detector.detect(text)
        self.assertTrue(result["has_phi"])
        self.assertGreater(result["phi_count"], 0)
        self.assertEqual(result["risk_level"], "high")
        
        types_found = {f["type"] for f in result["findings"]}
        self.assertIn("SSN", types_found)
        self.assertIn("EMAIL", types_found)
        print(f"[Pass] PHI detection found {result['phi_count']} items, risk: {result['risk_level']}.")

    def test_pii_detection(self):
        from phi.phi_detector import PHIDetector
        detector = PHIDetector()
        
        text = "Credit card: 1234-5678-9012-3456 and Passport: AB1234567"
        result = detector.detect(text)
        self.assertTrue(result["has_phi"])
        
        types_found = {f["type"] for f in result["findings"]}
        self.assertIn("CREDIT_CARD", types_found)
        self.assertIn("PASSPORT", types_found)
        print("[Pass] Extended PII detection (Credit Card, Passport) verified.")

    def test_phi_redaction(self):
        from phi.phi_detector import PHIDetector
        detector = PHIDetector()
        
        text = "SSN: 123-45-6789 and email test@example.com"
        redacted = detector.redact(text)
        self.assertNotIn("123-45-6789", redacted)
        self.assertNotIn("test@example.com", redacted)
        self.assertIn("[REDACTED]", redacted)
        print("[Pass] PHI redaction verified.")

    def test_clean_text(self):
        from phi.phi_detector import PHIDetector
        detector = PHIDetector()
        
        result = detector.detect("Normal clinical notes about cancer treatment protocols.")
        self.assertFalse(result["has_phi"])
        self.assertEqual(result["risk_level"], "none")
        print("[Pass] Clean text PHI scan verified.")


class TestPIHDetector(unittest.TestCase):
    def test_high_risk_pih(self):
        from pih.pih_detector import PIHRiskScorer
        scorer = PIHRiskScorer()
        
        text = "32 weeks gestation, BP: 160/110, preeclampsia suspected. Proteinuria detected. Severe headache and visual disturbance."
        result = scorer.score(text)
        self.assertEqual(result["risk_level"], "high")
        self.assertGreater(result["score"], 5)
        self.assertIsNotNone(result["gestational_weeks"])
        print(f"[Pass] PIH high risk detected. Score: {result['score']}, indicators: {result['indicator_count']}.")

    def test_no_pih_risk(self):
        from pih.pih_detector import PIHRiskScorer
        scorer = PIHRiskScorer()
        
        result = scorer.score("Standard chemotherapy protocol for stage 3 breast cancer.")
        self.assertEqual(result["risk_level"], "none")
        self.assertEqual(result["score"], 0.0)
        print("[Pass] No PIH risk for non-obstetric text verified.")


class TestComplaintExtractor(unittest.TestCase):
    def test_complaint_extraction(self):
        from complaint.complaint_extractor import ComplaintExtractor
        extractor = ComplaintExtractor()
        
        text = "Patient presents with severe chest pain rated 8/10 for 3 days. Also reports shortness of breath and nausea."
        result = extractor.extract(text)
        self.assertGreater(result["complaint_count"], 0)
        self.assertEqual(result["pain_score"], 8)
        self.assertEqual(result["overall_severity"], "severe")
        self.assertIn("chest pain", result["all_symptoms"])
        print(f"[Pass] Extracted {result['complaint_count']} complaints. Severity: {result['overall_severity']}, Pain: {result['pain_score']}/10.")

    def test_no_complaints(self):
        from complaint.complaint_extractor import ComplaintExtractor
        extractor = ComplaintExtractor()
        
        result = extractor.extract("Normal laboratory results within expected ranges.")
        self.assertEqual(result["complaint_count"], 0)
        print("[Pass] No complaints for normal text verified.")


class TestServiceTagger(unittest.TestCase):
    def test_oncology_tagging(self):
        from services.service_tagger import ServiceTagger
        tagger = ServiceTagger()
        
        result = tagger.tag("Breast cancer chemotherapy with radiation therapy and biopsy staging.")
        self.assertEqual(result["primary_service"], "Oncology")
        self.assertGreater(result["confidence"], 0.2)
        print(f"[Pass] Service tagged as {result['primary_service']} (confidence: {result['confidence']}).")

    def test_cardiology_tagging(self):
        from services.service_tagger import ServiceTagger
        tagger = ServiceTagger()
        
        result = tagger.tag("Cardiac arrhythmia detected on ECG, stent placement recommended.")
        self.assertEqual(result["primary_service"], "Cardiology")
        print(f"[Pass] Service tagged as {result['primary_service']}.")


class TestBedrockClient(unittest.TestCase):
    def test_mock_generation(self):
        from bedrock.bedrock_client import BedrockGenerator
        gen = BedrockGenerator(model_name="claude-3-haiku")
        
        result = gen.generate("You are a clinical assistant.", "What is the treatment?")
        self.assertIn("response_text", result)
        self.assertIn("cost", result)
        self.assertIn("provider", result)
        self.assertIn("mock", result["provider"])
        print(f"[Pass] Bedrock mock generation verified. Provider: {result['provider']}.")

    def test_mock_embeddings(self):
        from bedrock.bedrock_client import BedrockEmbeddings
        embedder = BedrockEmbeddings()
        
        vec = embedder.embed_query("cancer treatment protocol")
        self.assertEqual(len(vec), 1536)
        self.assertIsInstance(vec[0], float)
        
        batch = embedder.embed_documents(["text one", "text two"])
        self.assertEqual(len(batch), 2)
        print("[Pass] Bedrock mock embeddings verified.")


if __name__ == "__main__":
    unittest.main()
