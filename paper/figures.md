# 07_figures.md

# CivicSense AI
## Figure Planning Document

---

# IMPORTANT

Figures should communicate ideas, not decorate the paper.

Every figure must answer one engineering question.

Avoid unnecessary complexity.

Use vector graphics wherever possible.

Prefer black-and-white IEEE-friendly diagrams.

---

# Figure 1
## Overall System Architecture

Status
Required

Purpose

Provide a complete overview of CivicSense AI.

Layout

                Upload Document
                        │
                        ▼
              Document Processing
                        │
                        ▼
              Semantic Chunking
                        │
                        ▼
          MiniLM Embedding Generation
                        │
                        ▼
                Local Vector Index

────────────────────────────────────────────

                  User Query
                        │
                        ▼
              Query Embedding
                        │
                        ▼
               Cosine Retrieval
                        │
                Top-5 Chunks
                        │
                        ▼
          Prompt Construction
                        │
                        ▼
          Qwen2.5-3B (LM Studio)
                        │
                        ▼
      JSON Validation & Repair
                        │
                        ▼
 Answer + Explanation + Sources

Highlight

Offline Pipeline

vs

Online Pipeline

This should be the first figure.

---

# Figure 2
## Evaluation Workflow

Status

Required

Purpose

Explain the engineering methodology.

Layout

Embedding Models
        │
        ▼
Embedding Study
        │
        ▼
Chunk Size Study
        │
        ▼
Phase 1
(Model Comparison)
        │
        ▼
Phase 2
(Reasoning Ablation)
        │
        ▼
LLM Judge
        │
        ▼
Phase 3
(500 Query Benchmark)
        │
        ▼
Security Benchmark
        │
        ▼
Final Deployment

This is one of the most important figures.

It immediately tells reviewers

"This system was engineered systematically."

---

# Figure 3
## Retrieval-Augmented Generation Pipeline

Purpose

Explain one inference cycle.

Layout

Question

↓

Embed

↓

Retrieve

↓

Top-5 Chunks

↓

Prompt

↓

LLM

↓

Structured JSON

↓

Answer

↓

Evidence

Emphasize

Grounding

instead of

Hallucination.

---

# Figure 4
## Chunking Strategy

Purpose

Explain overlap.

Show

Chunk 1

------------------------

Chunk 2

          ------------------------

Chunk 3

                    ------------------------

Highlight

350 words

50 overlap

Explain

Why overlap matters.

---

# Figure 5
## Model Selection Funnel

Purpose

Show how the final model was selected.

Layout

5 Models

↓

Phase 1

↓

Top 3

↓

Phase 2

↓

Top 2

↓

Large Scale

↓

Winner

↓

Production

Reviewers love this.

---

# Figure 6
## Security Workflow

Purpose

Explain Instruction Integrity.

Layout

Retrieved Document

↓

Contains Malicious Prompt

↓

Instruction Integrity Layer

↓

Prompt Construction

↓

LLM

↓

Safe Response

Instead of

Malicious Response

---

# Figure 7
## Metric Relationships

Optional

Purpose

Explain

Retrieval

↓

Generation

↓

Faithfulness

↓

Security

This helps readers understand why multiple evaluation metrics are necessary.

---

# Figure Placement

Figure 1

Methodology

Immediately after System Overview.

---

Figure 2

Experimental Setup

Before Phase 1.

---

Figure 3

Methodology

Near Prompt Construction.

---

Figure 4

Chunk Study.

---

Figure 5

Immediately before Phase 3.

---

Figure 6

Security Evaluation.

---

Figure 7

Discussion.

---

# Figures NOT Needed

Do NOT include

- screenshots

- UI images

- code snippets

- terminal output

- confusion matrices

- unnecessary flowcharts

Every figure should improve understanding.

---

# Style Guide

Use

rounded rectangles

minimal colors

consistent fonts

uniform spacing

vector graphics

Avoid gradients.

Avoid shadows.

Avoid decorative icons.

Think IEEE.

---

# Future Improvements

If submitting to a journal instead of a conference,

consider adding

- latency comparison graph

- memory usage graph

- qualitative response comparison

These are optional and should not replace the primary engineering figures.