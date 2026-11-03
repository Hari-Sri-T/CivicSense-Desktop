# IEEE GICITE 2026 Camera-Ready Revision Guide
**Paper ID: 38 | CivicSense AI**

Congratulations on the acceptance to GICITE 2026! Below is a comprehensive breakdown of the reviewer comments and how the evaluation work we just completed (Phases 1-4) directly addresses their technical suggestions.

---

## Reviewer 1: Addressed Feedback

### 1. Benchmark Setup & Perfect Retrieval (Recall@5 = 100%)
> *Suggestion: Provide a more detailed explanation of the evaluation protocol and discuss whether the benchmark setup may have simplified the retrieval task.*
**Our Data to Add:** The perfect retrieval scores are a result of our benchmark setup. Datasets like ContractNLI and CUAD were evaluated on single-document retrieval (document-level RAG), meaning the search space for each query was confined to a single legal document rather than a massive multi-document corpus. This naturally inflates Recall@5 and MRR to near-perfect levels compared to open-domain retrieval.

### 2. Chunking Strategy Ablation
> *Suggestion: An ablation study investigating the impact of chunk size and overlap on retrieval and generation performance is recommended.*
**Our Data to Add:** We performed this exact ablation study. 
- **Tested Configurations:** `(150, 20)`, `(350, 50)`, `(500, 100)`.
- **Results:** 
  - `(150, 20)` caused context fragmentation, severing long legal clauses.
  - `(500, 100)` diluted the prompt with irrelevant context, confusing the LLM.
  - `(350, 50)` was optimal, balancing context density with prompt constraints.

### 3. Methodology Details
> *Suggestion: The methodology section lacks sufficient implementation details regarding embedding generation, vector indexing, retrieval parameters, prompt engineering, and JSON output enforcement...*
**Our Data to Add:** You can now provide these specifics:
- **Embeddings:** `all-MiniLM-L6-v2` running locally on CPU.
- **Retrieval:** Cosine similarity vector search, Top-K = 5.
- **Prompt Engineering:** Strict `INSTRUCTION INTEGRITY` rules in the system prompt to enforce SCOPE and GROUNDING. 
- **JSON Output:** Forced via system prompt and repaired via the `json_repair` library in our Flask backend.

---

## Reviewer 2: Addressed Feedback

### 1. Lexical vs. Semantic Metrics Discrepancy
> *Suggestion: Discuss why BLEU, chrF, Token-F1, and Exact Match scores remain limited despite strong semantic retrieval performance.*
**Our Data to Add:** We explicitly analyzed this in our Master Report. 
- Lexical metrics like BLEU (14.01) and Exact Match (0.2%) punish generative abstraction. The benchmark gold answers are often rigid, copy-pasted legal clauses. 
- CivicSense is explicitly prompted to *simplify* legal jargon for non-experts. Therefore, the LLM correctly rewrites the legalese into plain English. 
- This causes lexical mismatch but high semantic correctness, which is proven by our **BERTScore (F1) of 0.846**, showing the semantic meaning is highly preserved despite vocabulary changes.

### 2. Alternative Embedding Models
> *Suggestion: A comparative evaluation using alternative embedding models would help justify the chosen architecture.*
**Our Data to Add:** We conducted this evaluation in Phase 1!
- **Tested:** `all-MiniLM-L6-v2` vs `all-mpnet-base-v2`.
- **Results:** `mpnet-base` achieved slightly higher cosine similarity (0.69 vs 0.64), but `MiniLM` is 14x faster to embed, requires 50% less memory, and is highly optimized for local CPU execution. This justifies the choice of MiniLM for an accessible, locally-hosted tool.

### 3. System Robustness & Adversarial Prompts
> *Suggestion: Investigate system robustness against ambiguous queries, adversarial prompts...*
**Our Data to Add:** We completed Phase 4 Security Benchmarking specifically for this!
- **Dataset:** 50 adversarial behaviors from the TeleAI-Safety benchmark.
- **Model:** `qwen2.5-3b-instruct` (Unrestricted reasoning).
- **Result:** **0.0% Attack Success Rate (ASR)**. The `INSTRUCTION INTEGRITY` constraints in our system prompt successfully forced the model to reject 100% of malicious requests while maintaining proper JSON formatting.

### 4. Computational Resources & Scalability
> *Suggestion: Computational resource requirements, memory consumption, and scalability analysis...*
**Our Data to Add:** We ran a 500-query scale test.
- **Model Stability:** Qwen maintained a **1.0% error rate** over 500 queries at scale. (Gemma failed with a 98.8% error rate due to strict reasoning constraints breaking its JSON generation).
- **Latency:** Average generation latency was ~18.3 seconds running locally via LM Studio. 

---

## What We Missed (You Will Need to Address These Manually)
While our evaluations generated massive amounts of data for the technical system, there are a few reviewer and chair suggestions that we *cannot* solve with code and must be handled in your manuscript text:

1. **Chair Formatting:** You must remove honorifics (Dr., Professor), use IEEE citation format, format affiliations correctly (no "Independent Researcher"), and map the paper to UN Sustainable Development Goals (SDGs) (e.g., SDG 16: Peace, Justice, and Strong Institutions).
2. **Reviewer 1 & 2 (Comparisons):** We didn't evaluate CivicSense against *other* State-of-the-Art RAG frameworks (like GraphRAG or Cohere Command). You may need to add a short literature review paragraph comparing the architectures.
3. **Reviewer 2 (User Studies):** We did not conduct a human-centered evaluation with non-expert users. You should explicitly list this as "Future Work" in your conclusion to satisfy the reviewer!
