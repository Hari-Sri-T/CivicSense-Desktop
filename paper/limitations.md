# 09_limitations.md

# CivicSense AI
## Limitations

---

# IMPORTANT

Every strong research paper acknowledges its limitations.

This section should not weaken the contribution. Instead, it should demonstrate scientific maturity by identifying opportunities for future improvement.

---

# 1. Limited Model Scope

The experimental evaluation focuses on lightweight open-weight language models that can be deployed on commodity consumer hardware.

The objective of this work was not to identify the strongest language model overall, but rather to determine which lightweight models provide the best balance between response quality, computational efficiency, explainability, and deployment cost.

Consequently, larger proprietary language models such as GPT-4, Claude, Gemini, or other cloud-hosted systems were intentionally excluded from the evaluation.

Future work may compare CivicSense AI against larger commercial systems while considering privacy, latency, and operational cost.

---

# 2. Domain Specificity

The proposed framework was designed and evaluated primarily for legal and civic document understanding.

Although the underlying Retrieval-Augmented Generation architecture is domain independent, experimental validation was performed exclusively on legal datasets.

Additional evaluation across other knowledge-intensive domains—including healthcare, finance, education, scientific literature, and government records—would provide stronger evidence regarding the generalizability of the proposed framework.

---

# 3. Lightweight Deployment Constraints

One of the primary design goals of CivicSense AI was enabling fully local deployment on modest consumer hardware.

This design decision influenced several architectural choices, including

- lightweight embedding models,
- compact language models,
- simple cosine similarity retrieval,
- in-memory indexing.

While these choices improve accessibility and deployment efficiency, they may not fully exploit the capabilities of larger hardware configurations.

Future versions of the framework may investigate larger language models, GPU-accelerated retrieval, and distributed indexing while preserving explainability.

---

# 4. Retrieval Architecture

The proposed system employs dense semantic retrieval using cosine similarity over MiniLM embeddings.

Although this approach demonstrated excellent retrieval performance throughout the evaluation, more sophisticated retrieval architectures remain unexplored.

Potential future improvements include

- hybrid dense-lexical retrieval,
- reranking models,
- late interaction retrieval,
- knowledge graph augmentation,
- adaptive retrieval,
- query rewriting.

These techniques may further improve retrieval precision for complex legal queries.

---

# 5. Prompt Engineering

The reasoning-ablation experiments demonstrate that prompting strategy significantly influences Retrieval-Augmented Generation performance.

However, only two prompting configurations were evaluated:

- unrestricted reasoning,
- reasoning-suppressed prompting.

Future work may investigate

- chain-of-thought prompting,
- retrieval-aware prompting,
- adaptive prompting,
- self-reflection,
- multi-agent reasoning,
- confidence-guided prompting.

Understanding how prompting interacts with retrieval remains an important research direction.

---

# 6. Benchmark Coverage

Evaluation employed four datasets from LegalBench-RAG together with adversarial prompts from TeleAI-Safety.

Although these benchmarks provide comprehensive evaluation across multiple legal tasks, they cannot represent the complete diversity of legal documentation encountered in practice.

Additional evaluation should include

- international legal systems,
- multilingual legal corpora,
- government regulations,
- judicial opinions,
- legislative documents,
- legal correspondence.

Such evaluation would further validate the robustness of the proposed framework.

---

# 7. Security Evaluation

The Instruction Integrity mechanism demonstrated excellent resistance against prompt injection within the TeleAI-Safety benchmark.

Nevertheless, Retrieval-Augmented Generation security remains an evolving research area.

Future investigations should evaluate resilience against

- retrieval poisoning,
- adversarial indexing,
- indirect prompt injection,
- multi-document attacks,
- retrieval manipulation,
- context contamination,
- malicious metadata.

These attack vectors extend beyond the scope of the present work.

---

# 8. Human Evaluation

The current evaluation relies primarily on benchmark metrics together with LLM-as-a-Judge assessment.

Although semantic evaluation provides valuable insight into explanation quality, the study does not include formal human evaluation by legal professionals or domain experts.

Future work should incorporate

- lawyer evaluation,
- usability studies,
- readability assessment,
- trust calibration,
- user satisfaction,
- decision support effectiveness.

Human-centered evaluation would provide a stronger understanding of the practical utility of explainable legal Retrieval-Augmented Generation systems.

---

# Summary

The limitations identified above primarily arise from deliberate engineering decisions intended to prioritize reproducibility, local deployment, computational efficiency, and systematic evaluation.

Rather than diminishing the contribution, these limitations define a clear roadmap for future research while reinforcing the practical nature of CivicSense AI as an explainable, privacy-preserving Retrieval-Augmented Generation framework suitable for deployment on commodity consumer hardware.