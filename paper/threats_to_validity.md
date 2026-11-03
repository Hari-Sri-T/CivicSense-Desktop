# 08_threats_to_validity.md

# CivicSense AI
## Threats to Validity

---

# IMPORTANT

This section should appear near the end of the paper, preferably after the Discussion and before the Conclusion.

Including a Threats to Validity section demonstrates awareness of experimental limitations and generally improves reviewer confidence.

---

# 1. Internal Validity

The experiments presented in this work rely on a fixed Retrieval-Augmented Generation pipeline in which only one component was modified during each experimental phase. This design minimizes confounding variables; however, interactions between multiple components were not exhaustively explored.

For example, embedding models were compared while keeping chunk size, retrieval strategy, and prompting configuration constant. Similarly, reasoning-ablation experiments modified only the prompting strategy while maintaining identical retrieval settings.

Although this controlled methodology improves interpretability, alternative combinations of embeddings, chunking strategies, and prompting configurations may produce different results.

---

# 2. Construct Validity

The evaluation employs multiple complementary metrics, including lexical similarity, semantic similarity, retrieval effectiveness, and robustness.

Nevertheless, benchmark metrics such as BLEU, Exact Match, and Token F1 primarily reward textual overlap rather than semantic equivalence.

Because CivicSense AI intentionally generates simplified natural-language explanations instead of reproducing legal clauses verbatim, lexical metrics may underestimate practical explanation quality.

To mitigate this limitation, semantic evaluation using BERTScore together with LLM-as-a-Judge faithfulness assessment was incorporated into the evaluation framework.

---

# 3. External Validity

The proposed framework was evaluated using four datasets from the LegalBench-RAG benchmark.

Although these datasets represent diverse legal tasks—including contracts, privacy policies, merger agreements, and legal entailment—they do not encompass every type of legal document encountered in real-world practice.

Consequently, performance on additional legal domains such as taxation, criminal law, constitutional law, intellectual property, or jurisdiction-specific legislation remains an open research question.

Future work should evaluate CivicSense AI across broader collections of legal and governmental documents.

---

# 4. Hardware Validity

All experiments were conducted using a single commodity laptop equipped with

- AMD Ryzen 5 5600H
- 8 GB RAM
- NVIDIA GTX 1650 Laptop GPU (4 GB VRAM)

The reported latency and deployment characteristics therefore reflect this hardware configuration.

Although the framework is expected to scale to more capable consumer hardware, absolute inference latency may differ across systems.

However, retrieval quality and benchmark metrics are expected to remain largely unaffected because they depend primarily on model outputs rather than computational performance.

---

# 5. Model Validity

This study focuses exclusively on lightweight open-weight language models suitable for local deployment.

The evaluated models represent practical deployment candidates rather than the strongest language models currently available.

Consequently, the reported conclusions should not be interpreted as comparisons against large proprietary cloud-hosted systems such as GPT-4, Claude, or Gemini.

Instead, the objective is to identify engineering trade-offs among lightweight locally deployable models operating under identical Retrieval-Augmented Generation conditions.

---

# 6. Security Validity

Security evaluation employed the TeleAI-Safety benchmark, which provides a standardized collection of prompt injection attacks.

Although this benchmark covers diverse document-based attacks, it cannot represent every possible adversarial strategy.

Future work should investigate additional attack vectors including

- indirect prompt injection,
- retrieval poisoning,
- adversarial document modification,
- jailbreak attacks,
- multi-document prompt injection.

---

# 7. Sampling Validity

Large-scale evaluation employed a deterministic subset consisting of the first 125 benchmark queries from each LegalBench-RAG dataset.

This deterministic sampling improves reproducibility and enables direct comparison across experiments.

Nevertheless, alternative subsets or complete benchmark evaluation may produce slightly different aggregate scores.

Future work should validate the proposed framework using the complete LegalBench-RAG benchmark together with additional public legal datasets.

---

# Summary

Despite these limitations, the proposed evaluation combines controlled component studies, large-scale benchmarking, semantic evaluation, and adversarial robustness assessment within a unified experimental framework.

To the best of our knowledge, few existing studies simultaneously evaluate embedding selection, chunk optimization, lightweight language model comparison, reasoning behaviour, faithfulness, hallucination, large-scale legal benchmarking, and prompt injection robustness for fully local explainable legal Retrieval-Augmented Generation systems.