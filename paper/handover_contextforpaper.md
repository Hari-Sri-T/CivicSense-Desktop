# PROJECT_HANDOFF.md

# CivicSense AI
## Complete Research Handoff Document

Version: 1.0

Author:
OpenAI Research Collaboration Session

Last Updated:
July 2026

---

# READ THIS FIRST

This document contains the complete context required to continue development of the CivicSense AI research paper.

The next AI assistant should read this file before editing any part of the paper.

The objective is NOT to rewrite everything.

The objective is to transform Draft V2 into a submission-ready IEEE conference paper.

---

# 1. Project Overview

Project Name

CivicSense AI

Paper Title

CivicSense AI: An Explainable Retrieval-Augmented Generation System for Public Document Understanding

Project Type

Research Paper

Target

IEEE Conference

Current Stage

Complete Draft

Needs polishing and IEEE formatting.

---

# 2. Main Research Goal

This paper is NOT simply about building another RAG system.

The central research question is

> Can a fully local Explainable Retrieval-Augmented Generation system operating on commodity consumer hardware provide reliable, explainable, secure, and privacy-preserving legal document understanding?

Every section should contribute toward answering this question.

---

# 3. Paper Story

The original paper presented the work as

Architecture

↓

Results

This was intentionally changed.

The new story is

Engineering Problem

↓

System Design

↓

Component Selection

↓

Controlled Experiments

↓

Deployment Decisions

↓

Final System

↓

Discussion

The paper should read like an engineering research study rather than a software project report.

---

# 4. Current Paper Status

Completed Markdown Files

01_Title_Abstract.md

02_Introduction.md

03_Related_Work.md

04_Methodology.md

05_Experimental_Evaluation.md

06_tables.md

07_figures.md

08_threats_to_validity.md

09_limitations.md

10_future_work.md

Remaining Work

IEEE LaTeX conversion

Professional table formatting

Professional figures

Language polishing

Reviewer pass

Reference verification

Submission formatting

---

# 5. Verified Technical Facts

These values have been verified from the implementation.

Never change them unless the implementation changes.

Operating System

Windows

CPU

AMD Ryzen 5 5600H

RAM

8 GB

GPU

NVIDIA GTX 1650 Laptop GPU

VRAM

4 GB

Inference Runtime

LM Studio

Inference API

OpenAI Compatible

Final Model

qwen2.5-3b-instruct

Quantization

Q4_K_M

Context Window

8192

Temperature

0.1

Maximum Tokens

4096

Embedding Model

all-MiniLM-L6-v2

Embedding Device

CPU

Retrieval

Cosine Similarity

Top K

5

Chunk Size

350 words

Overlap

50 words

JSON Repair

Enabled

Instruction Integrity

Enabled

---

# 6. Experimental Pipeline

The experiments MUST appear in this order.

Embedding Model Selection

↓

Chunk Size Optimization

↓

Phase 1
Baseline Model Comparison

↓

Phase 2
Reasoning Ablation

↓

Faithfulness

↓

Hallucination

↓

Phase 3
500 Query Benchmark

↓

Security Evaluation

↓

Discussion

Changing this order weakens the story.

---

# 7. Source of Truth Priority

Whenever metrics differ

Priority

1

Master Evaluation Report

↓

2

Verified JSON Outputs

↓

3

Intermediate Logs

↓

4

Old Paper

Never trust the original paper over the Master Evaluation Report.

---

# 8. Verified Metrics

The Master Evaluation Report is the authoritative source.

Known verified values include

Recall@5

100%

MRR

1.000

BERTScore

0.8460

BLEU

14.02

chrF

31.54

Token F1

29%

Formatting Error

1%

Attack Success Rate

0%

Verify any remaining values directly from the Master Evaluation Report before publication.

---

# 9. Writing Philosophy

Do NOT write like a project report.

Write like an engineering research paper.

Every section should answer

Why?

How?

What did we learn?

Avoid simply describing implementation.

Always explain why engineering decisions were made.

---

# 10. What Makes This Paper Novel

The novelty is NOT

"We built a legal chatbot."

The novelty is

A systematic engineering study of lightweight local Retrieval-Augmented Generation.

The reasoning-ablation study.

Embedding selection.

Chunk optimization.

Prompt engineering.

Faithfulness evaluation.

Security evaluation.

Deployment on commodity hardware.

Explainability.

These contributions should be emphasized.

---

# 11. Tables

Planned tables

Embedding Comparison

Chunk Study

Phase 1

Phase 2

Phase Comparison

Faithfulness

Hallucination

Large Scale Benchmark

Security

Hardware Configuration

Refer to

06_tables.md

---

# 12. Figures

Planned figures

Architecture

Evaluation Workflow

RAG Pipeline

Chunk Strategy

Model Selection Funnel

Security Workflow

Metric Relationships

Refer to

07_figures.md

---

# 13. Reviewer Notes

Current draft is approximately

70%

complete.

Major improvements still needed.

Rewrite discussion.

Reduce repetition.

Improve transitions.

Strengthen novelty.

Add stronger interpretation.

Improve captions.

Professional IEEE tables.

Professional figures.

Humanize language.

Reduce AI writing style.

Shorten abstract slightly.

Threats to Validity should become part of the paper.

Limitations should become part of the paper.

---

# 14. Things NOT To Change

Do not claim CPU-only inference.

GPU was used.

Do not claim 24 GB RAM.

System has 8 GB RAM.

Do not replace LM Studio with Transformers.

Inference used LM Studio.

Do not invent benchmark numbers.

Do not invent latency values.

Do not invent security metrics.

Do not remove reasoning-ablation experiments.

Do not remove engineering story.

---

# 15. Known Weaknesses

Discussion is currently descriptive.

Needs stronger scientific interpretation.

Need comparison with related work.

Need stronger conclusion.

Need cleaner table design.

Need figure references.

Need better transitions.

Need IEEE wording pass.

---

# 16. Paper Quality Estimate

Current Draft

7.5–8/10

Expected after polishing

9–9.5/10

Main improvements remaining

Writing

Figures

Discussion

Flow

Professional formatting

---

# 17. Future Work

Already documented in

10_future_work.md

Do not repeat unnecessarily.

---

# 18. Submission Checklist

✓ Verify every metric

✓ Replace placeholder tables

✓ Create vector figures

✓ Convert Markdown to IEEE LaTeX

✓ Verify references

✓ Grammar pass

✓ Remove repetition

✓ Add Threats to Validity

✓ Add Limitations

✓ IEEE formatting

✓ Final PDF review

---

# 19. Prompt For The Next Chat

Paste the following into a new ChatGPT conversation.

------------------------------------------------------------

You are my research co-author.

We are preparing an IEEE conference paper titled

"CivicSense AI: An Explainable Retrieval-Augmented Generation System for Public Document Understanding."

Attached are

• PROJECT_HANDOFF.md
• All markdown sections
• Original LaTeX
• Master Evaluation Report
• Evaluation JSONs (only for verification)

Your responsibilities are

1. Never invent metrics.

2. Treat the Master Evaluation Report as the primary source of truth.

3. Preserve all verified implementation details.

4. Improve writing without changing experimental results.

5. Make the paper sound like it was written by experienced researchers.

6. Keep the engineering-study narrative.

7. Strengthen discussions and novelty.

8. Produce IEEE-ready LaTeX.

9. Suggest reviewer-level improvements whenever appropriate.

Assume the current draft is approximately 70% complete and focus on producing a submission-ready paper.

------------------------------------------------------------

# Final Note

The paper deliberately evolved from describing a software project into presenting a systematic engineering investigation of trustworthy local legal Retrieval-Augmented Generation.

That narrative should be preserved throughout every subsequent revision.

The objective is not merely to report benchmark scores, but to justify every engineering decision through controlled experimentation and produce a paper suitable for publication.