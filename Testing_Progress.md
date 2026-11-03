# CivicSense AI — Testing & Evaluation Progress

**Paper:** CivicSense AI: An Explainable RAG System for Public Document Understanding
**Conference:** IEEE GICITE 2026 — TU Berlin, November 6–7, 2026 (Paper ID: #38, **ACCEPTED**)
**Last Updated:** 2026-06-20

---

## Current Session Goal

Prepare camera-ready submission by:
1. Migrating the system from Gemini API → **LM Studio local models** (no cloud API needed)
2. Running all experiments requested by reviewers
3. Revising the LaTeX paper (`conference_101719.tex`) with results + reviewer responses

---

## Environment Setup — ✅ DONE

| Item | Status | Notes |
|---|---|---|
| Virtual env | ✅ Done | `~/.venv-civicsense/` (Python 3.14) |
| All Python deps | ✅ Done | `pip install -r requirements.txt` succeeded |
| Gemini → LM Studio | ✅ Done | `app.py` now uses `openai` client → `localhost:1234` |
| `evaluate3.py` | ✅ Done | Replaced `ollama/phi3` → LM Studio `JUDGE_MODEL` |
| `.env` updated | ✅ Done | Model IDs use exact LM Studio format (e.g. `google/gemma-4-12b-qat`) |
| CUDA OOM fix | ✅ Done | MiniLM + compare_embeddings forced to `device="cpu"` (LM Studio uses full GPU) |
| JSON parse fix | ✅ Done | `llm_call_fn()` uses text mode + robust `{}` extraction (cross-model compatible) |
| max_tokens fix | ✅ Done | Increased to `2048` — was causing truncated JSON |
| Flask smoke test | ✅ Done | Load → Query → LLM answer confirmed working |

### LM Studio Models Downloaded
| Model ID | Size | Role |
|---|---|---|
| `google/gemma-4-12b-qat` | ~8 GB | ✅ Active — Primary RAG answering |
| `microsoft/phi-4` | 9.05 GB | Evaluation queue |
| `mistralai/ministral-3-14b-reasoning` | 9.12 GB | Evaluation queue |
| `deepseek/deepseek-r1-0528-qwen3-8b` | 5.03 GB | Evaluation queue |
| `qwen/qwen3.5-9b` | 6.55 GB | Evaluation queue |
| `mistralai/ministral-3-3b` | 2.99 GB | Evaluation queue (lightweight) |
| `nvidia/nemotron-3-nano-4b` | 2.84 GB | Evaluation queue (lightweight) |

---

## New Scripts Created This Session

| Script | Purpose | LLM Needed? |
|---|---|---|
| `evaluate_model_comparison.py` | Run 100-query eval for one model at a time | ✅ LM Studio |
| `compile_model_results.py` | Aggregate all model results → paper comparison table | ❌ |
| `ablation_chunk_size_fast.py` | Fast retrieval-only ablation (7 sizes × 5 overlaps = 34 configs) | ❌ Flask only |
| `run_all.sh` | Master orchestrator for all experiments | Both |

---

## Experiments — Status

### ✅ DONE
| Experiment | Output File | Addresses |
|---|---|---|
| Embedding model comparison | `embedding_comparison_summary.txt` | Reviewer 2 #4 |

**Embedding Results Summary:**
```
Model                    | Dim | Top-1 Sim | Query(ms) | Embed(ms) | Load(s)
all-MiniLM-L6-v2         | 384 |    0.6456 |      3.80 |     10.97 |    0.08  ← CHOSEN
all-mpnet-base-v2        | 768 |    0.6918 |     17.98 |    137.79 |   49.53
paraphrase-MiniLM-L6-v2  | 384 |    0.6502 |      3.71 |      4.95 |   24.10
all-distilroberta-v1     | 768 |    0.5829 |     15.13 |     96.92 |   60.47
```
→ Justifies MiniLM choice: competitive quality, 14x faster, 50% less memory.

---

### 🔄 CURRENTLY RUNNING (background)

| Process | PID | Log | ETA | Addresses |
|---|---|---|---|---|
| `ablation_chunk_size_fast.py` | 167097 | `/tmp/ablation_output.log` | ~57 min | Reviewer 1 #4 |
| `evaluate_model_comparison.py --model google/gemma-4-12b-qat` | 168568 | `/tmp/eval_gemma.log` | ~60-90 min | New model comparison table |
| `robustness_test.py` | 168569 | `/tmp/robustness.log` | ~5-10 min | Reviewer 2 #5 |

**Monitor all at once:**
```bash
echo "=Ablation=" && tail -2 /tmp/ablation_output.log
echo "=ModelEval=" && tail -2 /tmp/eval_gemma.log
echo "=Robustness=" && tail -2 /tmp/robustness.log
```

---

### ⏳ PENDING — Needs LM Studio model switching

Run `evaluate_model_comparison.py` for each model below.
**Workflow per model:**
1. In LM Studio → Developer tab → load the model
2. Run: `~/.venv-civicsense/bin/python evaluate_model_comparison.py --model <MODEL_ID>`
3. Result saved to `model_results/result_<name>.json`
4. Switch to next model

| Model | Command | Status |
|---|---|---|
| `google/gemma-4-12b-qat` | `--model google/gemma-4-12b-qat` | 🔄 Running |
| `microsoft/phi-4` | `--model microsoft/phi-4` | ⏳ Pending |
| `mistralai/ministral-3-14b-reasoning` | `--model mistralai/ministral-3-14b-reasoning` | ⏳ Pending |
| `deepseek/deepseek-r1-0528-qwen3-8b` | `--model deepseek/deepseek-r1-0528-qwen3-8b` | ⏳ Pending |
| `qwen/qwen3.5-9b` | `--model qwen/qwen3.5-9b` | ⏳ Pending |
| `mistralai/ministral-3-3b` | `--model mistralai/ministral-3-3b` | ⏳ Pending |
| `nvidia/nemotron-3-nano-4b` | `--model nvidia/nemotron-3-nano-4b` | ⏳ Pending |

**After ALL models done:**
```bash
~/.venv-civicsense/bin/python compile_model_results.py
# → produces model_comparison_summary.txt + model_comparison_results.json
```

---

### ⏳ PENDING — Full Evaluation Pipeline (6,881 queries)

> Requires LM Studio loaded. Can run overnight.

```bash
# Stage 1: Generate all answers (2-3 hours)
~/.venv-civicsense/bin/python evaluate1.py

# Stage 2: Compute BERTScore, BLEU, chrF (30-60 min, no LM Studio)
~/.venv-civicsense/bin/python evaluate2.py

# Stage 3: LLM-as-Judge faithfulness scoring
# → Change JUDGE_MODEL in .env to best model, then:
~/.venv-civicsense/bin/python evaluate3.py
```

Output: `evaluation_summary.txt` + `evaluation_summary_final.txt`

---

### ⏳ PENDING — Scalability Benchmark

```bash
# NOTE: Run WITHOUT LM Studio loaded (imports app.py directly, conflicts with GPU)
~/.venv-civicsense/bin/python scalability_bench.py
```
Output: `scalability_summary.txt` → addresses Reviewer 2 #3

---

## LaTeX Paper Revisions — Status

### ⏳ ALL PENDING (waiting for experiment results first)

#### Conference Chair Comments
| # | Comment | Status |
|---|---|---|
| 1 | Remove honorifics | ⏳ Verify (likely already clean) |
| 2 & 11 | IEEE affiliations + emails | ⏳ **Need author emails from team** |
| 3 | Cite all figures/tables/equations in text | ⏳ |
| 4 | Improve figure captions + add results chart | ⏳ |
| 5 | Fix IEEE reference formatting | ⏳ |
| 6 | Revise abstract with key results | ⏳ |
| 7 | Add 6–8 recent 2024–2025 references | ⏳ |
| 8 | Add `hyperref`, `booktabs`, `url` packages | ⏳ |
| 9 | Add numbered math equations (6 formulas) | ⏳ |
| 10 | Fix formatting consistency | ⏳ |
| 12 | Add SDG 16 + SDG 10 mapping subsection | ⏳ |

#### Reviewer 1 Comments
| # | Comment | Status |
|---|---|---|
| 1 | Explain perfect Recall@5 / evaluation protocol | ⏳ |
| 2 | Real-world doc validation → add as limitation | ⏳ |
| 3 | SOTA RAG comparison → discuss + future work | ⏳ |
| 4 | Chunk size ablation table | ⏳ Waiting for `ablation_chunk_size_fast.py` |
| 5 | Faithfulness 0.6198 analysis | ⏳ |
| 6 | Implementation details subsection | ⏳ |

#### Reviewer 2 Comments
| # | Comment | Status |
|---|---|---|
| 1 | Explain low BLEU/chrF (generative vs extractive) | ⏳ |
| 2 | Clarify explainability scope + future work | ⏳ |
| 3 | Scalability table | ⏳ Waiting for `scalability_bench.py` |
| 4 | Embedding model comparison table | ✅ Results ready |
| 5 | Robustness analysis | ⏳ Waiting for `robustness_test.py` |
| 6 | User study → future work | ⏳ |

---

## Original Evaluation Results (from Gemini run — for reference)

| Benchmark | Recall@5 | MRR | BERTScore F1 | Latency |
|---|---|---|---|---|
| cuad.json | 100% | 1.000 | 0.7936 | 0.21s |
| privacy_qa.json | 100% | 1.000 | 0.8084 | 0.26s |
| contractnli.json | 100% | 1.000 | 0.8010 | 0.21s |
| maud.json | 100% | 1.000 | 0.7936 | 0.22s |
| **Overall (6,881 queries)** | **100%** | **1.000** | **0.7951** | **0.21s** |

**LLM-as-Judge (Gemini + phi3):** Faithfulness = 0.6198, Hallucination Rate = **0.76%**

---

## Key File Locations

```
/run/media/tsl/Hari/CivicSense/CivicSense/
├── app.py                          ← Flask backend (LM Studio version)
├── .env                            ← LM Studio model config
├── requirements.txt                ← Updated (openai, no google/ollama)
├── evaluate1.py                    ← Stage 1: generate answers
├── evaluate2.py                    ← Stage 2: compute metrics
├── evaluate3.py                    ← Stage 3: LLM-as-Judge (LM Studio)
├── ablation_chunk_size_fast.py     ← NEW: fast retrieval-only ablation
├── evaluate_model_comparison.py    ← NEW: per-model evaluation
├── compile_model_results.py        ← NEW: aggregate model results
├── run_all.sh                      ← NEW: master orchestrator
├── model_results/                  ← Per-model result JSONs (created on run)
├── embedding_comparison_summary.txt ← ✅ DONE
├── ablation_results_fast.json      ← pending
├── ablation_summary_fast.txt       ← pending
├── robustness_results.json         ← pending
├── scalability_results.json        ← pending

/run/media/tsl/Hari/CivicSense/
└── conference_101719.tex           ← LaTeX paper (needs revision)
```

---

## Recommended Next Actions (in order)

1. **Now:** Wait for robustness test to finish (~5 min), check `/tmp/robustness.log`
2. **Now:** Switch LM Studio model → `microsoft/phi-4`, run model comparison eval
3. **Later:** Run `scalability_bench.py` when LM Studio is not loaded (saves GPU)
4. **After all models:** Run `compile_model_results.py` → paper table ready
5. **After experiments:** Begin LaTeX revision (need author emails first for Chair #2)
6. **Final:** LaTeX compile → PDF review → similarity report → submit
