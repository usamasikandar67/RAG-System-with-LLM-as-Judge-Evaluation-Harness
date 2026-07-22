import os
import json
import hashlib
from typing import List, Dict, Any

from ner.ner_pipeline import extract_clinical_entities
from icd_mapping.icd_resolver import resolve_icd_code
from phi.phi_detector import PHIDetector
from complaint.complaint_extractor import ComplaintExtractor
from services.service_tagger import ServiceTagger
from experiments.db import log_document, log_chunk



class BaseChunker:
    def split_text(self, text: str) -> List[str]:
        raise NotImplementedError

class RecursiveCharacterTextSplitter(BaseChunker):
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 30):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> List[str]:
        return self._split(text, self.separators)

    def _split(self, text: str, separators: List[str]) -> List[str]:
        if not separators:
            return [text]
        
        separator = separators[0]
        next_separators = separators[1:]
        
        if separator == "":
            splits = list(text)
        else:
            splits = text.split(separator)
            
        chunks = []
        current_chunk = []
        current_len = 0
        
        for part in splits:
            part_len = len(part)
            addition = len(separator) if len(current_chunk) > 0 else 0
            
            if current_len + part_len + addition <= self.chunk_size:
                current_chunk.append(part)
                current_len += part_len + addition
            else:
                if current_chunk:
                    chunks.append(separator.join(current_chunk))
                    
                if part_len > self.chunk_size:
                    chunks.extend(self._split(part, next_separators))
                    current_chunk = []
                    current_len = 0
                else:
                    overlap_chunk = []
                    overlap_len = 0
                    for item in reversed(current_chunk):
                        item_len = len(item)
                        item_addition = len(separator) if len(overlap_chunk) > 0 else 0
                        if overlap_len + item_len + item_addition <= self.chunk_overlap:
                            overlap_chunk.insert(0, item)
                            overlap_len += item_len + item_addition
                        else:
                            break
                    current_chunk = overlap_chunk + [part]
                    current_len = sum(len(x) for x in current_chunk) + (len(separator) * (len(current_chunk) - 1))
                    
        if current_chunk:
            chunks.append(separator.join(current_chunk))
            
        return [c.strip() for c in chunks if c.strip()]

class MarkdownTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 30):
        super().__init__(chunk_size, chunk_overlap)
        # Markdown specific hierarchy
        self.separators = [
            "\n# ", "\n## ", "\n### ", "\n#### ", "\n##### ", "\n###### ",
            "```\n", "\n\n", "\n", " ", ""
        ]

class SentenceTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 30):
        super().__init__(chunk_size, chunk_overlap)
        # Split primarily by sentence endings
        self.separators = [". ", "? ", "! ", "\n\n", "\n", " ", ""]

class SemanticChunker(BaseChunker):
    """Placeholder for true semantic chunking (e.g. embedding-based). Falls back to sentence chunking."""
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 30):
        self.fallback = SentenceTextSplitter(chunk_size, chunk_overlap)
        
    def split_text(self, text: str) -> List[str]:
        # In a real implementation, this would use an embedding model to group sentences by cosine similarity
        return self.fallback.split_text(text)

def get_chunker(strategy: str = "recursive", chunk_size: int = 300, chunk_overlap: int = 30) -> BaseChunker:
    if strategy == "markdown": return MarkdownTextSplitter(chunk_size, chunk_overlap)
    if strategy == "sentence": return SentenceTextSplitter(chunk_size, chunk_overlap)
    if strategy == "semantic": return SemanticChunker(chunk_size, chunk_overlap)
    return RecursiveCharacterTextSplitter(chunk_size, chunk_overlap)

def ingest_documents(knowledge_base_dir: str, chunk_strategy: str = "recursive", chunk_size: int = 300, chunk_overlap: int = 30, chunking_enabled: bool = True) -> List[Dict[str, Any]]:
    if not os.path.exists(knowledge_base_dir):
        print(f"[Ingestion] Warning: Knowledge base directory {knowledge_base_dir} does not exist.")
        return []

    splitter = get_chunker(strategy=chunk_strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    phi_detector = PHIDetector()
    complaint_extractor = ComplaintExtractor()
    service_tagger = ServiceTagger()
    documents = []
    
    # Read manifest.json if exists to get category/topic mapping
    manifest_path = os.path.join(os.path.dirname(knowledge_base_dir), "manifest.json")
    category_map = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as fm:
                manifest_data = json.load(fm)
                for filename, meta in manifest_data.items():
                    category_map[filename] = meta.get("topic", "Oncology")
        except:
            pass

    for filename in sorted(os.listdir(knowledge_base_dir)):
        if not filename.endswith(".txt"):
            continue
        
        filepath = os.path.join(knowledge_base_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        category = category_map.get(filename, "Oncology")

        # Generate a unique document ID
        doc_hash = hashlib.md5(content.encode("utf-8")).hexdigest()[:8]
        actual_doc_id = f"doc_{filename.split('.')[0]}_{doc_hash}"
        
        # Log Document to DB
        log_document("src/experiments/evaluation_results.db", actual_doc_id, filename, filename, "clinical_guideline", {"category": category})

        # Chunking toggle: split into chunks or treat whole file as one chunk
        if chunking_enabled:
            chunks = splitter.split_text(content)
        else:
            chunks = [content.strip()] if content.strip() else []
        
        for idx, chunk_text in enumerate(chunks):
            # Generate stable chunk ID based on source metadata and content hash
            chunk_hash = hashlib.md5(f"{filename}_{idx}_{chunk_text}".encode("utf-8")).hexdigest()[:12]
            chunk_id = f"chunk_{filename.split('.')[0]}_{idx}_{chunk_hash}"
            
            # Extract Clinical Entities
            entities = extract_clinical_entities(chunk_text)
            
            # Resolve ICD-10 Code
            icd_data = {"code": "C80", "category": "Unknown Neoplasm", "description": "Malignant neoplasm without specification"}
            if entities["cancer_types"]:
                icd_data = resolve_icd_code(entities["cancer_types"][0])

            # PHI Detection
            phi_result = phi_detector.detect(chunk_text)

            # Complaint Extraction
            complaint_result = complaint_extractor.extract(chunk_text)

            # Services Tagging
            service_result = service_tagger.tag(chunk_text)
                
            metadata = {
                "source_file": filename,
                "clinical_category": category,
                "chunk_index": idx,
                "chunking_enabled": chunking_enabled,
                "cancer_types": entities["cancer_types"],
                "biomarkers": entities["biomarkers"],
                "drugs": entities["drugs"],
                "procedures": entities["procedures"],
                "dosages": entities["dosages"],
                "icd_10_code": icd_data["code"],
                "icd_category": icd_data["category"],
                "icd_description": icd_data["description"],
                "phi_detected": phi_result["has_phi"],
                "phi_risk_level": phi_result["risk_level"],
                "phi_count": phi_result["phi_count"],
                "complaint_count": complaint_result["complaint_count"],
                "overall_severity": complaint_result["overall_severity"],
                "primary_service": service_result["primary_service"],
                "service_confidence": service_result["confidence"]
            }
            
            # Log Chunk to DB
            log_chunk("src/experiments/evaluation_results.db", chunk_id, actual_doc_id, idx, chunk_text, len(chunk_text.split()), metadata)
            
            documents.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": metadata
            })

            
    print(f"[Ingestion] Processed {len(documents)} chunks from {knowledge_base_dir}.")
    return documents
