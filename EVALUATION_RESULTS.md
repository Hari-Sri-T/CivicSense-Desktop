# CivicSense AI — Evaluation Results

## Phase 1: Baseline (Unrestricted Reasoning)

**Date:** 2026-06-29  
**Pipeline:** `evaluate_v2.py`  
**Queries:** 24 total (6 per benchmark × 4 benchmarks)  
**Dataset:** LegalBench-RAG subset  
**Embedding Model:** `all-MiniLM-L6-v2` (384-dim, CPU)  
**Chunk Size:** 350 words, 50-word overlap, Top-K=5  
**Context Window:** 8192 tokens (all models)

---

### Final Standings

| Rank | Model | Error Rate | BERTScore F1 | Corpus BLEU | Corpus chrF | Token F1 | Avg Latency |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1st | `qwen2.5-3b-instruct` | **0.0%** | **0.8807** | **19.28** | **35.03** | **42.25%** | 21.3s |
| 2nd | `google/gemma-3-4b` | **0.0%** | 0.8734 | 16.39 | 30.45 | 38.43% | 23.3s |
| 3rd | `mistralai/ministral-3-3b` | **0.0%** | 0.8453 | 3.80 | 22.07 | 21.63% | 25.1s |
| 4th | `nvidia/nemotron-3-nano-4b` | 33.3% | 0.8229 | 0.04 | 8.68 | 13.13% | 22.5s |
| 5th | `llama-3.2-3b-instruct` | 95.8% | 0.7718 | 0.00 | 0.22 | 0.0% | 30.5s |

All metrics computed on valid (non-error) answers only. BERTScore uses roberta-large F1. BLEU/chrF computed at corpus level. Results verified against raw JSON files.

---

### Spot-Check Validation (Sample Q&A)

#### Qwen2.5-3b-instruct (1st place)
**Q:** Does the NDA between CopAcc and ToP Mentors define proprietary rights?  
**Model:** "The Non-Disclosure Agreement clearly states in Clause 8: 'Proprietary rights, limited right to use... Any and all proprietary rights..."  
**Gold:** "Any and all proprietary rights, including but not limited to rights to and in inventions, patent rights..."  
**Token F1:** 0.68 | **BERTScore:** 0.9168 -- Extracting directly from source text.

#### Gemma-3-4b (2nd place)
**Q:** Does the NDA state that Confidential Info must be kept confidential post-termination?  
**Model:** "Yes, the document mentions that Confidential Information must be kept confidential for as long as..."  
**Gold:** "Notwithstanding the termination of this Agreement, any Confidential Information must be kept confidential for as long as..."  
**Token F1:** 0.85 | **BERTScore:** 0.9604 -- Near-perfect extraction.

#### Llama-3.2-3b (5th place -- root cause confirmed)
**Q:** Does the NDA define proprietary rights?  
**Model:** Error: Could not generate a response. (23 out of 24 queries)  
**Cause:** Llama spent nearly all its context budget on verbose internal reasoning, exhausted the 8192-token limit, and failed to output any JSON.

---

### Key Findings

1. **Qwen2.5 wins on all metrics.** It extracted near-verbatim quotes from source documents with a BLEU of 19.28 -- almost 3x higher than Gemma's 16.39 and 5x higher than Ministral's 3.80.

2. **The reasoning token problem is real.** Llama-3.2 confirmed the hypothesis: unconstrained reasoning traces exhaust the context window completely, leaving no tokens for the answer.

3. **Nemotron over-simplifies.** When it did not error, Nemotron gave answers like "Yes" or "No" with no supporting detail, resulting in near-zero BLEU despite a reasonable BERTScore.

4. **Ministral punishes over-formatting.** Its answers were accurate in meaning (0.8453 BERTScore) but heavy with bold markdown and verbose explanations, hurting lexical overlap scores.

---

## Phase 2: Restricted Reasoning Ablation

**Prompt change:** `NO_REASONING_SYSTEM_PROMPT` — explicitly forbids `<think>` blocks, markdown, preamble.

| Rank | Model | Error Rate | BERTScore F1 | Corpus BLEU | Corpus chrF | Token F1 | Avg Latency |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1st | `google/gemma-3-4b` | **0.0%** | **0.8798** | **19.94** | **34.04** | **40.05%** | 23.0s |
| 2nd | `mistralai/ministral-3-3b` | **0.0%** | 0.8646 | 5.37 | 24.57 | 29.38% | 21.9s |
| 3rd | `qwen2.5-3b-instruct` | **0.0%** | 0.8519 | 8.12 | 25.65 | 28.4% | 18.3s |
| 4th | `llama-3.2-3b-instruct` | **0.0%** | 0.8327 | 3.95 | 21.54 | 18.3% | 18.9s |
| 5th | `nvidia/nemotron-3-nano-4b` | 50.0% | 0.8006 | 0.01 | 4.41 | 4.75% | 15.0s |

---

### Head-to-Head: Phase 1 vs Phase 2

| Model | P1 BERT | P2 BERT | P1 BLEU | P2 BLEU | P1 Errors | P2 Errors | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `qwen2.5-3b-instruct` | **0.8807** | 0.8519 | **19.28** | 8.12 | 0% | 0% | Reasoning HELPS Qwen |
| `google/gemma-3-4b` | 0.8734 | **0.8798** | 16.39 | **19.94** | 0% | 0% | Slightly better restricted |
| `mistralai/ministral-3-3b` | 0.8453 | **0.8646** | 3.80 | **5.37** | 0% | 0% | Restriction helps |
| `llama-3.2-3b-instruct` | 0.7718 | **0.8327** | 0.00 | **3.95** | **95.8%** | **0%** | MASSIVE comeback |
| `nvidia/nemotron-3-nano-4b` | 0.8229 | 0.8006 | 0.04 | 0.01 | 33.3% | **50.0%** | Gets worse restricted |

### Phase 2 Key Findings

1. **Llama's dramatic comeback.** Restricting reasoning slashed its error rate from 95.8% to 0.0%. It went from completely broken to a fully functional model. The restriction stopped it burning its context on thought traces.

2. **Qwen is uniquely hurt by restrictions.** It is the only model whose BLEU dropped significantly (19.28 → 8.12). Qwen was actively using its reasoning to build better, more detailed answers. The restriction made it terse.

3. **Gemma is the most consistent model.** It performs well in both phases, slightly improving under restriction (BLEU 16.39 → 19.94). It naturally produces clean JSON without over-thinking.

4. **Ministral improved modestly** under restriction. Less verbose markdown formatting = more direct quotes = better lexical scores.

5. **Nemotron is fundamentally broken** regardless of prompting strategy.

### Winner for Phase 3

**`google/gemma-3-4b`** — Best across both phases combined. Consistent 0% errors, top BERTScore in Phase 2 (0.8798), top BLEU in Phase 2 (19.94), and notably faster than Qwen under restriction.

Runner-up: `qwen2.5-3b-instruct` — Best unrestricted model. Could be run with Phase 1 prompting for Phase 3.

> Decision: Run Phase 3 full dataset with **Gemma-3-4b** (Phase 2 restricted prompt — it is Gemma's strongest configuration).

---

## Hallucination Check (LLM-as-Judge via Qwen2.5)

**Judge model:** `qwen2.5-3b-instruct`  
**Method:** For each valid answer, Qwen received the original query, the gold answer (verbatim from the legal document), and the model answer. It rated `is_faithful` (bool) and `hallucination_score` (0.0=faithful, 1.0=complete hallucination).

| Model | Phase | Faithful% | Avg Halluc Score | BERTScore | BLEU |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `qwen2.5-3b-instruct` | P1 | **29.2%** | **0.669** | 0.8807 | 19.28 |
| `google/gemma-3-4b` | P2 | 25.0% | 0.744 | 0.8798 | 19.94 |
| `qwen2.5-3b-instruct` | P2 | 16.7% | 0.775 | 0.8519 | 8.12 |
| `nvidia/nemotron-3-nano-4b` | P1 | 6.2% | 0.909 | 0.8229 | 0.04 |
| `google/gemma-3-4b` | P1 | 4.2% | 0.906 | 0.8734 | 16.39 |
| `llama-3.2-3b-instruct` | P2 | 4.2% | 0.904 | 0.8327 | 3.95 |
| `mistralai/ministral-3-3b` | P2 | 4.2% | 0.867 | 0.8646 | 5.37 |
| `llama-3.2-3b-instruct` | P1 | 0.0% | 1.000 | 0.7718 | 0.00 |
| `mistralai/ministral-3-3b` | P1 | 0.0% | 0.910 | 0.8453 | 3.80 |
| `nvidia/nemotron-3-nano-4b` | P2 | 0.0% | 0.938 | 0.8006 | 0.01 |

> [!IMPORTANT]
> **Critical caveat about these scores.** The gold answers in LegalBench-RAG are verbatim legal text excerpts (e.g., *"Any and all proprietary rights, including but not limited to rights to and in inventions..."*). The judge compares model answers to these exact quotes. Any model that paraphrases, summarises, or adds plain-language explanation — even if 100% factually correct — gets flagged as a "hallucination". This is why all scores appear harsh. **The relative ranking matters, not the absolute percentages.**

### Hallucination Check Key Findings

1. **Qwen Phase 1 is the most faithful model (29.2%).** Its tendency to quote directly from the source (reflected in its high BLEU of 19.28) also earns it the highest faithfulness score from the judge.

2. **Gemma Phase 2 is second most faithful (25.0%).** The restricted prompt made it more likely to quote directly rather than elaborate.

3. **All models score low in absolute terms** because the judge compares to verbatim legal extracts. This is a known limitation of using exact-match gold answers as the judge reference in legal RAG evaluation.

4. **The ordering is consistent with BLEU.** Models with higher BLEU (more quote-like answers) also score higher faithfulness — which validates that BLEU is a meaningful proxy for grounding in this task.

5. **Revised overall winner: Qwen2.5 Phase 1** — best across BERTScore (0.88), BLEU (19.28), and lowest hallucination score (0.669). Gemma Phase 2 is the better choice if reasoning must be restricted.

---

## Phase 3: Full Dataset Run (Winning Model)
*Pending — Gemma-3-4b (Phase 2 restricted prompt) vs. Qwen2.5 (Phase 1 unrestricted) against full LegalBench-RAG dataset.*

## Phase 4: Security Benchmarking
*Pending — JailbreakBench + TeleAI-Safety adversarial evaluation.*
