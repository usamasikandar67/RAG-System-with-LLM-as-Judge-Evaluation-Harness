# Evaluation Framework - Cancer Clinical AI Evaluation Platform

## 1. Information Retrieval (IR) Metrics Formulas

### Hits@K
$$\text{Hits}@K = \begin{cases} 
      1.0 & \text{if } E \cap R_{:K} \neq \emptyset \\
      0.0 & \text{otherwise}
   \end{cases}$$
Where $E$ represents expected documents and $R_{:K}$ represents top $K$ retrieved document chunks.

### Reciprocal Rank (RR)
$$\text{RR} = \frac{1}{\text{rank}_{\text{first}}}$$
Where $\text{rank}_{\text{first}}$ is the position index of the first expected document in the retrieval list.

### Average Precision (AP)
$$\text{AP} = \frac{\sum_{i=1}^{M} \text{Precision}(i) \times \text{relevance}(i)}{\text{total relevant documents}}$$

---

## 2. LLM-as-Judge Grading Prompts

The generator output is graded against ground truth and retrieved contexts:
- **Correctness**: Semantic accuracy of clinical assertions.
- **Completeness**: Ratio of matching clinical points mentioned.
- **Faithfulness**: Detection of ungrounded sentences (hallucinations).
- **Citation Accuracy**: Validates that sources are labeled in square brackets `[source_file.txt]`.
- **Medical Safety**: Flags drug dosage overrides or incorrect stage combinations.
