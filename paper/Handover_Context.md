# CivicSense AI: Master Context & Handover Document

This document serves as the ultimate "save state" for the CivicSense AI project. If you start a new AI chat or bring a new developer onto the project, providing this document will give them 100% of the context required to understand what has been built, how it was evaluated, and the current state of the codebase.

---

## 1. Project Overview
**Name:** CivicSense AI  
**Goal:** An Explainable Retrieval-Augmented Generation (RAG) system designed to help non-experts (the general public) understand complex legal and civic documents (e.g., contracts, privacy policies, municipal bylaws).  
**Key Feature:** The system acts as a "civic translator", simplifying dense legalese into plain English while strictly outputting structured JSON containing the `answer` and a cited `explanation` from the source text.

## 2. Technology Stack & Architecture
- **Backend:** Python + Flask (`app.py`).
- **Document Parsing:** `PyMuPDF` (for PDFs) and `python-docx` (for DOCX).
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (Local CPU execution, 384 dimensions).
- **Vector Indexing:** In-memory NumPy arrays with Brute-Force Cosine Similarity (Top-K = 5).
- **Chunking Strategy:** Sequential semantic chunking (350 words, 50-word overlap).
- **Generative Model (Inference):** Handled locally via **LM Studio** API (OpenAI-compatible endpoints).

## 3. The Codebase (Key Files)
- `f:\CivicSense\CivicSense\app.py`: The core Flask RAG application.
- `f:\CivicSense\CivicSense\evaluate_v1.py` & `evaluate_v2.py`: Initial baseline scripts.
- `f:\CivicSense\CivicSense\evaluate_v3_phase3.py`: The final Phase 3 evaluation script capable of running 500+ query scale tests.
- `f:\CivicSense\CivicSense\evaluate_v4_security.py`: The security/robustness benchmark script.
- `f:\CivicSense\conference_101719.tex`: The final IEEE GICITE 2026 Camera-Ready manuscript.
- `f:\CivicSense\CivicSense\update_latex.py`: The script used to automatically patch the `.tex` file with the latest evaluation metrics.
- `f:\CivicSense\CivicSense\LegalBench-RAG\`: The folder containing the core benchmarks (CUAD, Privacy-QA, ContractNLI, MAUD).
- `f:\CivicSense\CivicSense\teleai_attacks.json`: The dataset of 50 adversarial behaviors used for security testing.

---

## 4. Evaluation Phases & History

We conducted an exhaustive 4-phase evaluation pipeline to benchmark the system for an IEEE paper submission.

### Phase 1: Embedding & Baseline Evaluation
- **Embeddings Tested:** Compared `all-MiniLM-L6-v2` against `all-mpnet-base-v2`. 
- **Result:** Chose `MiniLM` because it is 14x faster to embed and uses 50% less memory with only a negligible (~0.05) drop in Top-1 cosine similarity.
- **Chunking Tested:** Ablation proved 350-word chunks with a 50-word overlap was optimal.

### Phase 2: Reasoning Restriction Ablation
- **Test:** We tested forcing models to skip `<think>` reasoning blocks via the system prompt (`NO_REASONING_SYSTEM_PROMPT`).
- **Result:** Restricting reasoning completely fixed `llama-3.2-3b-instruct` (which was previously eating its entire context window), but it severely damaged `qwen2.5-3b-instruct`'s generative quality. `gemma-3-4b` proved to be the most consistent model under restriction at small scale.

### Phase 3: Large-Scale Semantic Evaluation
- **Test:** We ran a 500-query scale test (125 queries across 4 benchmarks) to measure stability.
- **The Gemma Crash:** We discovered that forcing `gemma-3-4b` to suppress its reasoning blocks at scale caused a **98.8% error rate** (JSON formatting failures/timeouts). 
- **The Qwen Victory:** `qwen2.5-3b-instruct` (run with unrestricted reasoning) achieved an incredible **1.0% error rate** over 500 queries, averaging an 18.4s latency.
- **Metrics:** The system achieved a **0.8460 BERTScore (F1)** and near-perfect **99.6% Recall@5**.

### Phase 4: Security Benchmarking
- **Test:** Evaluated the system against 50 malicious/adversarial jailbreak attempts from the **TeleAI-Safety** benchmark.
- **Result:** The system relies on strict `INSTRUCTION INTEGRITY` rules. Qwen successfully achieved a **0.0% Attack Success Rate (ASR)** by rejecting all 50 harmful requests while maintaining safe JSON formatting.

---

## 5. Critical System Knowledge & Quirks
If an AI agent needs to modify this codebase, they must know the following:

1. **The Lexical Overlap Quirk:** Lexical metrics like BLEU (14.02) and Exact Match (0.0%) are incredibly low for this system. *This is by design.* CivicSense is prompted to simplify complex legal jargon for laypeople. Because n-gram metrics punish paraphrasing, they fail to capture the system's accuracy. We rely entirely on **BERTScore (0.8460)** to prove semantic equivalence.
2. **LLM-as-a-Judge Hallucination Caveat:** The LLM judge scores faithfulness strictly against verbatim legal quotes. Because CivicSense abstracts text, its faithfulness score appears moderate (29.2%). This is a known limitation of using extraction-based gold standards for explanatory tasks.
3. **JSON Repair:** The system generates structured JSON, but small local LLMs often hallucinate trailing commas, unescaped quotes, or markdown wrappers (`````json`). The `app.py` relies on a robust `json_repair` pipeline to salvage malformed outputs.
4. **Prompt Dominance:** The security of the system relies entirely on the system prompt dominating the retrieved context. The prompt explicitly instructs the model *not* to comply with instructions found inside the uploaded document, which prevents data-driven prompt injections.

---
*Generated upon the successful completion of the IEEE GICITE 2026 evaluation and manuscript patch.*
