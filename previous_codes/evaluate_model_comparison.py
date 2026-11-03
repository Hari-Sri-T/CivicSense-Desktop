"""
MULTI-MODEL RAG EVALUATION
===========================
Tests a single LM Studio model through the full RAG pipeline on a subset of
LegalBench-RAG benchmarks. Run once per model (load the model in LM Studio
first, then run this script).

Usage:
  1. Load your model in LM Studio and start the server (port 1234)
  2. Start Flask app:  python app.py
  3. Run this script:  python evaluate_model_comparison.py --model gemma-4-12b-qat

Results are saved as:  model_results/result_{model_name}.json
After running all models, run:  python compile_model_results.py

Models to test (load each in LM Studio and run):
  - gemma-4-12b-qat          (best quality, already downloaded)
  - phi-4                    (Microsoft reasoning)
  - deepseek-r1-0528-qwen3-8b (DeepSeek R1 reasoning)
  - ministral-3-14b-reasoning (Mistral reasoning)
  - nemotron-3-nano-4b       (NVIDIA compact)
  - ministral-3-3b           (lightweight)
  - qwen3.5-9b               (Qwen latest)
"""

import os
import sys
# Force UTF-8 output on Windows terminals to avoid cp1252 codec errors
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import glob
import json
import time
import argparse
import requests
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import warnings
from transformers import logging as transformers_logging
import sacrebleu
try:
    import bert_score
    BERT_SCORE_AVAILABLE = True
except (ImportError, OSError) as e:
    print(f"[WARNING] bert_score/torch unavailable: {e}")
    print("[WARNING] BERTScore will be skipped. All other metrics will run normally.")
    BERT_SCORE_AVAILABLE = False

# Suppress verbose warnings
transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)

# ---- CONFIG ----
BASE_URL            = "http://localhost:5001"
CLEAR_ENDPOINT      = f"{BASE_URL}/clear_index"
LOAD_FILE_ENDPOINT  = f"{BASE_URL}/load_file"
QUERY_ENDPOINT      = f"{BASE_URL}/query"
LM_STUDIO_URL       = "http://localhost:1234/v1/models"

CORPUS_DIR          = "LegalBench-RAG/corpus"
BENCHMARK_DIR       = "LegalBench-RAG/benchmarks"
RESULTS_DIR         = "model_results"

# How many queries to run per benchmark (4 benchmarks × 10 = 40 queries total per model)
# 10 is a good middle ground: gives enough statistical significance without taking 50 mins per model.
MAX_QUERIES_PER_BENCHMARK = 10

SECONDS_BETWEEN_QUERIES   = 0.5  # local model, can go faster than cloud API

os.makedirs(RESULTS_DIR, exist_ok=True)


# ---- Helpers ----

def recall_at_k(retrieved_sources, gold_file_path, k=5):
    if not retrieved_sources or not gold_file_path:
        return 0.0
    norm_gold = gold_file_path.replace("\\", "/")
    for src in retrieved_sources[:k]:
        if norm_gold in src.replace("\\", "/"):
            return 1.0
    return 0.0


def mean_reciprocal_rank(retrieved_sources, gold_file_path):
    if not retrieved_sources or not gold_file_path:
        return 0.0
    norm_gold = gold_file_path.replace("\\", "/")
    for i, src in enumerate(retrieved_sources):
        if norm_gold in src.replace("\\", "/"):
            return 1.0 / (i + 1)
    return 0.0


def token_f1(pred, gold):
    if not pred or not gold:
        return 0.0
    p_toks = set(pred.lower().split())
    g_toks = set(gold.lower().split())
    common = p_toks & g_toks
    if not common:
        return 0.0
    prec = len(common) / (len(p_toks) + 1e-8)
    rec  = len(common) / (len(g_toks) + 1e-8)
    return 2 * prec * rec / (prec + rec)


def get_active_model():
    """Ask LM Studio which model is currently loaded."""
    try:
        r = requests.get(LM_STUDIO_URL, timeout=3)
        data = r.json()
        models = data.get("data", [])
        if models:
            return models[0].get("id", "unknown")
    except Exception:
        pass
    return "unknown"


def check_flask_running():
    try:
        r = requests.get(f"{BASE_URL}/", timeout=3)
        return True
    except Exception:
        return False


# ---- Main ----

def run_evaluation(model_name: str):
    print("=" * 65)
    print(f"  MULTI-MODEL EVALUATION — {model_name}")
    print("=" * 65)

    # Verify Flask app is running
    if not check_flask_running():
        print(f"\n[ERROR] Flask app not reachable at {BASE_URL}")
        print("   Start it with:  python app.py")
        return

    # Verify LM Studio is running and log active model
    active = get_active_model()
    print(f"\n[OK] Flask app running at {BASE_URL}")
    print(f"   LM Studio active model: {active}")
    if active != "unknown" and model_name.lower() not in active.lower():
        print(f"\n[WARNING] LM Studio shows '{active}' but you passed --model '{model_name}'")
        print("   Make sure you loaded the correct model in LM Studio before running!")
        confirm = input("   Continue anyway? [y/N]: ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            return

    bench_paths = glob.glob(os.path.join(BENCHMARK_DIR, "*.json"))
    if not bench_paths:
        print(f"[ERROR] No benchmark files found in {BENCHMARK_DIR}")
        return

    session = requests.Session()
    all_results = []
    all_preds, all_golds = [], []

    for bench_path in tqdm(bench_paths, desc="Benchmarks"):
        bench_name = os.path.basename(bench_path)
        with open(bench_path, "r", encoding="utf-8") as f:
            bench_data = json.load(f)

        tests = bench_data.get("tests", [])
        if not tests:
            continue

        questions_by_file = defaultdict(list)
        for item in tests:
            q       = item.get("query")
            snippet = item.get("snippets", [{}])[0]
            gt      = snippet.get("answer")
            fp      = snippet.get("file_path")
            if q and gt and fp:
                questions_by_file[fp].append({"query": q, "gold_answer": gt})

        query_count = 0

        for file_path, qa_list in questions_by_file.items():
            if MAX_QUERIES_PER_BENCHMARK and query_count >= MAX_QUERIES_PER_BENCHMARK:
                break

            full_path = os.path.join(CORPUS_DIR, file_path)
            if not os.path.exists(full_path):
                continue

            # Clear + load file
            try:
                session.post(CLEAR_ENDPOINT).raise_for_status()
                resp = session.post(LOAD_FILE_ENDPOINT, json={"file_path": full_path})
                resp.raise_for_status()
                if resp.json().get("status") != "success":
                    continue
            except Exception as e:
                print(f"\n  Skip {file_path}: {e}")
                continue

            for item in qa_list:
                if MAX_QUERIES_PER_BENCHMARK and query_count >= MAX_QUERIES_PER_BENCHMARK:
                    break

                q  = item["query"]
                gt = item["gold_answer"]
                if isinstance(gt, list):
                    gt = " | ".join(str(g) for g in gt)
                else:
                    gt = str(gt)
                gt = str(item["gold_answer"]) if item["gold_answer"] is not None else ""

                t0 = time.time()
                # --- Generate LLM Answer ---
                model_answer = ""
                try:
                    resp = session.post(QUERY_ENDPOINT, json={"query": q, "top_k": 5})
                    if resp.status_code == 200:
                        data = resp.json()
                        # ENSURE IT IS A STRING (no None type)
                        model_answer = str(data.get("answer", ""))
                    else:
                        print(f"Error {resp.status_code}")
                except Exception as e:
                    print(f"Request failed: {e}")

                if not model_answer.strip():
                    model_answer = "Empty"

                retrieved_srcs = data.get("sources", []) if "data" in locals() else []
                
                res_dict = {
                    "benchmark":     bench_name,
                    "file_path":     file_path,
                    "query":         q,
                    "gold_answer":   gt,
                    "model_answer":  model_answer,
                    "latency_sec":   round(time.time() - t0, 4),
                    "recall_at_5":   recall_at_k(retrieved_srcs, file_path),
                    "mrr":           mean_reciprocal_rank(retrieved_srcs, file_path),
                    "token_f1":      token_f1(model_answer, gt),
                    "exact_match":   1.0 if model_answer.strip().lower() == gt.strip().lower() else 0.0,
                }
                all_results.append(res_dict)
                all_preds.append(model_answer)
                all_golds.append(gt)
                
                # Save incremental backup so we never lose 40 queries again
                backup_path = f"model_results/backup_{model_name.replace('/', '_')}.json"
                with open(backup_path, "w", encoding="utf-8") as f:
                    json.dump({"summary": "IN PROGRESS", "detailed_results": all_results}, f, indent=2)

                query_count += 1
                time.sleep(SECONDS_BETWEEN_QUERIES)

    if not all_results:
        print("No results collected. Check server status.")
        return

    if BERT_SCORE_AVAILABLE:
        print(f"\nCalculating BERTScore for {len(all_preds)} predictions...")
        try:
            _, _, bert_f1 = bert_score.score(all_preds, all_golds, lang="en", verbose=False)
            bert_scores = bert_f1.tolist()
        except Exception as e:
            print(f"BERTScore failed: {e}")
            bert_scores = [None] * len(all_preds)
    else:
        print("\n[SKIPPED] BERTScore - torch not available on this environment.")
        bert_scores = [None] * len(all_preds)

    print("Calculating BLEU and chrF...")
    for i, res in enumerate(all_results):
        res["bert_score_f1"] = round(bert_scores[i], 4) if bert_scores[i] is not None else None
        res["bleu"]  = round(sacrebleu.corpus_bleu([res["model_answer"]], [[res["gold_answer"]]]).score, 4)
        res["chrf"]  = round(sacrebleu.corpus_chrf([res["model_answer"]], [[res["gold_answer"]]]).score, 4)

    # ---- Aggregate ----
    metrics = defaultdict(list)
    for r in all_results:
        for k in ["recall_at_5", "mrr", "token_f1", "exact_match",
                  "bert_score_f1", "bleu", "chrf", "latency_sec"]:
            v = r[k]
            if v is not None:
                metrics[k].append(v)

    def safe_mean(key):
        vals = metrics.get(key, [])
        return round(float(np.mean(vals)), 4) if vals else None

    summary = {
        "model":            model_name,
        "lm_studio_model":  active,
        "total_queries":    len(all_results),
        "recall_at_5":      round(safe_mean("recall_at_5") * 100, 2) if safe_mean("recall_at_5") is not None else None,
        "mrr":              safe_mean("mrr"),
        "bert_score_f1":    safe_mean("bert_score_f1"),  # None if torch unavailable
        "bleu":             safe_mean("bleu"),
        "chrf":             safe_mean("chrf"),
        "token_f1_pct":     round(safe_mean("token_f1") * 100, 2) if safe_mean("token_f1") is not None else None,
        "exact_match_pct":  round(safe_mean("exact_match") * 100, 2) if safe_mean("exact_match") is not None else None,
        "avg_latency_sec":  safe_mean("latency_sec"),
        "bert_score_available": BERT_SCORE_AVAILABLE,
    }

    # ---- Save ----
    safe_name = model_name.replace("/", "_").replace(":", "_")
    out_file = os.path.join(RESULTS_DIR, f"result_{safe_name}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "detailed_results": all_results}, f, indent=2)

    print(f"\n[DONE] Results saved to {out_file}")
    print("\n" + "=" * 65)
    print(f"  SUMMARY - {model_name}")
    print("=" * 65)
    print(f"  Total Queries:   {summary['total_queries']}")
    print(f"  Recall@5:        {summary['recall_at_5']}%")
    print(f"  MRR:             {summary['mrr']}")
    bert_str = str(summary['bert_score_f1']) if summary['bert_score_f1'] is not None else "N/A (torch unavailable)"
    print(f"  BERTScore F1:    {bert_str}")
    print(f"  BLEU:            {summary['bleu']}")
    print(f"  chrF:            {summary['chrf']}")
    print(f"  Token F1:        {summary['token_f1_pct']}%")
    print(f"  Avg Latency:     {summary['avg_latency_sec']}s")
    print("=" * 65)
    if not BERT_SCORE_AVAILABLE:
        print("\n[NOTE] BERTScore was skipped. Fix torch install, then re-run evaluate2.py to add it.")
    print("\nNext: load the next model in LM Studio and re-run with --model <name>")
    print("Then run:  python compile_model_results.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a single LM Studio model on LegalBench-RAG.")
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model name/identifier (used for output filename). "
             "Should match the model loaded in LM Studio. "
             "Example: --model gemma-4-12b-qat"
    )
    args = parser.parse_args()
    run_evaluation(args.model)
