# ICD Mapping - Cancer Clinical AI Evaluation Platform

## ICD-10 Coding Reference Mappings
Clinical text cancers must resolve to standard codes to enable structured searches and performance cohorts mapping.

| Cancer Type | ICD-10 Code | Description | Category |
|---|---|---|---|
| **Breast Cancer** | C50 | Malignant neoplasm of breast | Breast Neoplasm |
| **Lung Cancer** | C34 | Malignant neoplasm of bronchus and lung | Thoracic Oncology |
| **Colon Cancer** | C18 | Malignant neoplasm of colon | Gastrointestinal |
| **Prostate Cancer** | C61 | Malignant neoplasm of prostate | Genitourinary |
| **Liver Cancer** | C22 | Malignant neoplasm of liver & intrahepatic ducts | Hepatobiliary |
| **Melanoma** | C43 | Malignant melanoma of skin | Dermatologic |
| **Leukemia** | C91 | Lymphoid leukemia | Hematologic |
| **Lymphoma** | C81 | Hodgkin lymphoma | Hematologic |

## Modular Resolver Architecture
The architecture isolates the mapper behind an interface (`BaseMedicalCoder`). This guarantees future extensions for RxNorm, SNOMED CT, or UMLS lookup dictionaries are pluggable without changing query pipeline routing:

```python
class BaseMedicalCoder(ABC):
    @abstractmethod
    def resolve_code(self, entity_text: str) -> Dict[str, str]:
        # Returns {"code": "C34", "system": "ICD-10", "description": "..."}
        pass
```
