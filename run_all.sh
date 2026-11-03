#!/usr/bin/env bash
# =============================================================================
# CivicSense AI — Full Experiment Runner
# =============================================================================
# Runs ALL experiments in the correct order.
# Prerequisites:
#   1. LM Studio running on localhost:1234 with desired model loaded
#   2. Python venv at ~/.venv-civicsense
# Usage:
#   bash run_all.sh [--model MODEL_NAME] [--skip-full-eval]
#
# Flags:
#   --model MODEL_NAME   Override LM_STUDIO_MODEL in .env for this session
#   --skip-full-eval     Skip the 6881-query full eval (just run ablation/rob/scale)
# =============================================================================

set -e
PYTHON="${CIVICSENSE_PYTHON:-python}"
export CIVICSENSE_BACKEND=lmstudio
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Parse args ---
MODEL_OVERRIDE=""
SKIP_FULL_EVAL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL_OVERRIDE="$2"; shift 2 ;;
        --skip-full-eval) SKIP_FULL_EVAL=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -n "$MODEL_OVERRIDE" ]; then
    export LM_STUDIO_MODEL="$MODEL_OVERRIDE"
    echo "▶ Model override: $MODEL_OVERRIDE"
fi

# --- Check LM Studio is up ---
echo ""
echo "============================================================"
echo "  CivicSense AI — Full Experiment Pipeline"
echo "============================================================"
echo ""

if ! curl -s http://localhost:1234/v1/models > /dev/null 2>&1; then
    echo "❌ LM Studio is NOT running on localhost:1234"
    echo "   Please start LM Studio and enable the local server, then retry."
    exit 1
fi
echo "✅ LM Studio is running"

# --- Check Flask app ---
if ! curl -s http://localhost:5001/ > /dev/null 2>&1; then
    echo ""
    echo "▶ Starting Flask app (app.py) in background..."
    $PYTHON app.py &
    FLASK_PID=$!
    sleep 4
    if ! curl -s http://localhost:5001/ > /dev/null 2>&1; then
        echo "❌ Flask app failed to start. Check app.py output."
        kill $FLASK_PID 2>/dev/null
        exit 1
    fi
    echo "✅ Flask app started (PID: $FLASK_PID)"
else
    echo "✅ Flask app already running"
    FLASK_PID=""
fi

cleanup() {
    if [ -n "$FLASK_PID" ]; then
        echo ""
        echo "▶ Stopping Flask app (PID: $FLASK_PID)..."
        kill "$FLASK_PID" 2>/dev/null
    fi
}
trap cleanup EXIT

# =============================================================================
# PHASE 0: LLM-free experiments (no model needed, run anytime)
# =============================================================================
echo ""
echo "------------------------------------------------------------"
echo "  PHASE 0: LLM-Free Experiments"
echo "------------------------------------------------------------"

echo ""
echo "▶ [0a] Embedding model comparison..."
$PYTHON compare_embeddings.py
echo "✅ Done → embedding_comparison_summary.txt"

echo ""
echo "▶ [0b] Scalability benchmark..."
$PYTHON scalability_bench.py
echo "✅ Done → scalability_summary.txt"

# =============================================================================
# PHASE 1: Ablation study (chunk size)
# =============================================================================
echo ""
echo "------------------------------------------------------------"
echo "  PHASE 1: Chunk Size Ablation Study"
echo "------------------------------------------------------------"
echo "▶ Testing 5 chunk sizes × 4 overlap values (20 configs, ~50 queries each)..."
echo "  This will take ~30-60 minutes."
echo ""
$PYTHON ablation_chunk_size_fast.py
echo "✅ Done → ablation_summary_fast.txt"

# =============================================================================
# PHASE 2: Robustness testing
# =============================================================================
echo ""
echo "------------------------------------------------------------"
echo "  PHASE 2: Robustness Testing"
echo "------------------------------------------------------------"
$PYTHON robustness_test.py
echo "✅ Done → robustness_summary.txt"

# =============================================================================
# PHASE 3: Multi-model comparison (current model only)
# =============================================================================
echo ""
echo "------------------------------------------------------------"
echo "  PHASE 3: Multi-Model Evaluation (current model)"
echo "------------------------------------------------------------"
ACTIVE_MODEL=$(curl -s http://localhost:1234/v1/models | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "${LM_STUDIO_MODEL:-unknown}")
echo "▶ Running model comparison eval for: $ACTIVE_MODEL"
$PYTHON previous_codes/evaluate_model_comparison.py --model "$ACTIVE_MODEL"
echo "✅ Done → model_results/result_${ACTIVE_MODEL/\//_}.json"

# =============================================================================
# PHASE 4: Full evaluation pipeline (optional, ~2-3 hours)
# =============================================================================
if [ "$SKIP_FULL_EVAL" = false ]; then
    echo ""
    echo "------------------------------------------------------------"
    echo "  PHASE 4: Full Evaluation Pipeline (6881 queries)"
    echo "------------------------------------------------------------"
    echo "⚠️  This will take 2-3+ hours. Press Ctrl+C to skip."
    echo ""

    echo "▶ [4a] Stage 1 — Generating results..."
    $PYTHON evaluate1.py
    echo "✅ Done → evaluation_results_stage1.json"

    echo ""
    echo "▶ [4b] Stage 2 — Calculating BERTScore, BLEU, chrF..."
    $PYTHON evaluate2.py
    echo "✅ Done → evaluation_results_full.json + evaluation_summary.txt"

    echo ""
    echo "▶ [4c] Stage 3 — LLM-as-Judge faithfulness scoring..."
    $PYTHON evaluate3.py
    echo "✅ Done → evaluation_results_judged.json + evaluation_summary_final.txt"
else
    echo ""
    echo "⏭  Skipping full 6881-query eval (--skip-full-eval flag set)"
fi

# =============================================================================
# DONE
# =============================================================================
echo ""
echo "============================================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "============================================================"
echo ""
echo "Output files:"
echo "  embedding_comparison_summary.txt   (Reviewer 2 #4)"
echo "  scalability_summary.txt            (Reviewer 2 #3)"
echo "  ablation_summary.txt               (Reviewer 1 #4)"
echo "  robustness_summary.txt             (Reviewer 2 #5)"
echo "  model_results/                     (LLM comparison table)"
if [ "$SKIP_FULL_EVAL" = false ]; then
echo "  evaluation_summary.txt             (Full eval metrics)"
echo "  evaluation_summary_final.txt       (Faithfulness / hallucination rate)"
fi
echo ""
echo "Next steps:"
echo "  1. Load the next model in LM Studio"
echo "  2. Run: python evaluate_model_comparison.py --model <name>"
echo "  3. After all models: python previous_codes/compile_model_results.py"
echo ""
