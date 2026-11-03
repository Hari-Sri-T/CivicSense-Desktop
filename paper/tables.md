# 06_tables.md

# CivicSense AI
## Table Planning and Final Table Mapping

---

# IMPORTANT

This document describes every table that should appear in the final IEEE paper.

**Do NOT invent numbers.**

Whenever numerical values are required, use the **Master Evaluation Report** as the final source of truth.

If any discrepancy exists between:

- Master Evaluation Report
- JSON outputs
- Intermediate evaluation logs

always verify the discrepancy and use the Master Evaluation Report unless an implementation bug is discovered.

---

# Table 1
## Embedding Model Comparison

Purpose

Justify why MiniLM was selected.

Columns

- Embedding Model
- Recall@5
- Mean Reciprocal Rank
- Retrieval Accuracy
- Relative Inference Speed
- Relative Memory Usage

Models

- all-MiniLM-L6-v2
- all-mpnet-base-v2
- BAAI/bge-base-en-v1.5

Discussion

Explain that MiniLM achieved nearly identical retrieval quality while requiring substantially lower computational resources.

Do NOT simply say it was "better."

Emphasize engineering trade-offs.

---

# Table 2
## Chunk Size Ablation

Purpose

Justify the final chunk size.

Columns

- Chunk Size
- Overlap
- Recall@5
- MRR
- BERTScore
- Notes

Configurations

- 200 / 30
- 350 / 50
- 500 / 75

Discussion

Explain

- small chunks fragment clauses

- large chunks introduce retrieval noise

- 350 words provides the best balance.

---

# Table 3
## Phase 1
### Lightweight LLM Comparison

Purpose

Baseline model comparison.

Include

- Qwen2.5-3B
- Gemma-3-4B
- Llama-3.2-3B
- Ministral-3B
- Nemotron Nano 4B

Columns

- Model

- BLEU

- chrF

- Token F1

- BERTScore

- Recall@5

- MRR

- Formatting Error

- Latency

Discussion

Do not declare a winner here.

This is exploratory.

---

# Table 4
## Phase 2
### Reasoning Suppression

Same columns as Table 3.

Purpose

Compare unrestricted reasoning vs suppressed reasoning.

Important discussion

Reasoning suppression is NOT universally beneficial.

Different models respond differently.

---

# Table 5
## Phase 1 vs Phase 2

Purpose

Highlight behavioural changes.

Columns

- Model

- Phase 1 BERTScore

- Phase 2 BERTScore

- Difference

- Observation

Observations should explain

- Improved

- Degraded

- Stable

This table is more informative than numerical.

---

# Table 6
## Faithfulness Evaluation

Columns

- Model

- Faithfulness

- Judge Comments (optional)

Purpose

Demonstrate grounding.

Discuss

High faithfulness despite modest lexical metrics.

---

# Table 7
## Hallucination Analysis

Columns

- Model

- Hallucination Rate

- Hallucination Severity

Purpose

Show retrieval grounding reduces hallucination.

---

# Table 8
## Large Scale Evaluation

Probably the most important table.

Include

Final production metrics.

Columns

- Metric

- Value

Metrics

Recall@5

MRR

BLEU

chrF

Token F1

BERTScore

Formatting Error

Latency

Use the verified values from the Master Evaluation Report.

---

# Table 9
## TeleAI Security

Columns

Attack Type

Passed

Failed

ASR

Final ASR

Purpose

Demonstrate robustness.

Highlight

Attack Success Rate = 0%.

---

# Possible Additional Table

Table 10

Hardware

CPU

GPU

RAM

Inference Runtime

Embedding Runtime

OS

Quantization

Context Length

This improves reproducibility.

---

# Formatting Guidelines

Every table should

- fit IEEE double-column layout

- avoid vertical borders

- bold best-performing values

- include caption above

- include explanatory paragraph after the table

Never leave a table unexplained.

Every table should answer a research question.

---

# Tables NOT Needed

Do not include

- giant metric dumps

- duplicate latency tables

- duplicate retrieval tables

- repeated benchmark summaries

If two tables communicate the same conclusion,

merge them.