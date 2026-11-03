"""
load_and_eval_legalbench_targeted.py

Usage:
1) pip install sacrebleu bert-score transformers
2) Ensure your Flask app (app.py) is updated with /clear_index and /load_file endpoints.
3) Start your Flask app: python app.py (listening at http://localhost:5001)
4) Run this script: python evaluate_civicsense.py

What it does:
- Iterates through each benchmark JSON in the BENCHMARK_DIR.
- For each benchmark:
  - Groups all questions by their source 'file_path'.
  - Iterates through ALL unique files.
  - For each file:
    1) Calls the /clear_index endpoint to clear the RAG's in-memory vector store.
    2) Calls the /load_file endpoint to load *only* that specific file into the vector store.
    3) Asks ALL questions related to that file.
    4) Records latency, exact match, F1, Recall@5, MRR, chrF, BLEU, and BERTScore-F1.
- Saves detailed JSON/CSV results and a final summary_report.txt.
"""

import os
import glob
import json
import time
import csv
import requests
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import itertools
import sacrebleu  # For chrF and BLEU scores
import bert_score # <-- ADDED for BERTScore
import warnings  # <-- NEW: To suppress warnings
from transformers import logging as transformers_logging # <-- NEW: To suppress warnings

# ---- NEW: Suppress verbose warnings ----
# Suppress "Some weights..." warnings from transformers
transformers_logging.set_verbosity_error()
# Suppress other general UserWarnings (often from bert-score)
warnings.filterwarnings("ignore", category=UserWarning)
# ----------------------------------------


# ---- CONFIG ----
BASE_URL = "http://localhost:5001"
# NEW: Endpoint to clear the vector store (must be added to app.py)
CLEAR_ENDPOINT = f"{BASE_URL}/clear_index"
# NEW: Endpoint to load a *single* file (must be added to app.py)
LOAD_FILE_ENDPOINT = f"{BASE_URL}/load_file"
# MODIFIED: /query endpoint (must return 'sources' for Recall/MRR)
QUERY_ENDPOINT = f"{BASE_URL}/query"

CORPUS_DIR = "LegalBench-RAG/corpus"
BENCHMARK_DIR = "LegalBench-RAG/benchmarks"

# --- REMOVED Evaluation Limits to run on ALL questions ---
# ------------------------------

# --- NEW: Output Files ---
SAVE_JSON = "evaluation_results_targeted.json"
SAVE_CSV = "evaluation_results_targeted.csv"
SAVE_SUMMARY = "evaluation_summary.txt"
# -------------------------

# Rate limit queries to stay within API quotas
# Set to 1.2 seconds for a 50 QPM rate (60 / 1.2 = 50)
SECONDS_BETWEEN_QUERIES = 1.2  # 50 QPM

# ---- 1. Helper Functions for Metrics ----

def token_f1_score(pred, gold):
    """Calculates token-level F1 score."""
    if not pred or not gold:
        return 0.0
    ptoks = pred.lower().split()
    gtoks = gold.lower().split()
    common = set(ptoks) & set(gtoks)
    if not gtoks or not ptoks:
        return 0.0
    
    precision = len(common) / (len(ptoks) + 1e-8) # Avoid zero division
    recall = len(common) / (len(gtoks) + 1e-8) # Avoid zero division
    
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def recall_at_k(retrieved_sources, gold_file_path, k=5):
    """
    Calculates Recall@k.
    Checks if the 'gold_file_path' is present in the top-k 'retrieved_sources'.
    Assumes sources are file paths or have a 'file_path' key.
    """
    if not retrieved_sources or not gold_file_path:
        return 0.0
    
    # The 'gold_file_path' from the benchmark is relative, e.g., "cuad/file.txt"
    # The 'retrieved_sources' from app.py are absolute, e.g., "LegalBench-RAG/corpus/cuad/file.txt"
    # We must check if the relative path is *contained in* the absolute path.
    
    top_k_sources = retrieved_sources[:k]
    
    relevant_found = False
    for source_path in top_k_sources:
        # Normalize paths for comparison
        norm_source = source_path.replace("\\", "/")
        norm_gold = gold_file_path.replace("\\", "/")
        if norm_gold in norm_source:
            relevant_found = True
            break
            
    return 1.0 if relevant_found else 0.0

def mean_reciprocal_rank(retrieved_sources, gold_file_path):
    """
    Calculates MRR.
    Finds the rank of the first relevant 'gold_file_path' in 'retrieved_sources'.
    """
    if not retrieved_sources or not gold_file_path:
        return 0.0

    # Normalize gold path once
    norm_gold = gold_file_path.replace("\\", "/")

    for i, source_path in enumerate(retrieved_sources):
        rank = i + 1
        
        # Normalize source path for comparison
        norm_source = source_path.replace("\\", "/")
        
        if norm_gold in norm_source:
            return 1.0 / rank # Found it, return reciprocal rank
            
    return 0.0 # Not found

def calculate_chrf(pred, gold):
    """Calculates the chrF score for semantic similarity."""
    if not pred or not gold:
        return 0.0
    # sacrebleu expects lists of strings for hypotheses and references
    # We use .score which returns the score directly (default is chrF2)
    return sacrebleu.corpus_chrf([pred], [[gold]]).score

def calculate_bleu(pred, gold):
    """Calculates the BLEU score for semantic similarity."""
    if not pred or not gold:
        return 0.0
    # sacrebleu expects lists of strings for hypotheses and references
    return sacrebleu.corpus_bleu([pred], [[gold]]).score

def calculate_bertscore(pred, gold):
    """Calculates the BERTScore F1 for semantic similarity."""
    if not pred or not gold:
        return 0.0
    try:
        # Returns (Precision, Recall, F1) tensors
        (P, R, F1) = bert_score.score([pred], [[gold]], lang='en', verbose=False)
        return F1.item() # Return the F1 score as a float
    except Exception as e:
        # We don't print warnings anymore, just return 0
        return 0.0

# ---- 2. Run Benchmarks (New Logic) ----
print(f"Running targeted benchmarks from {BENCHMARK_DIR}...")
bench_paths = glob.glob(os.path.join(BENCHMARK_DIR, "*.json"))

if not bench_paths:
    print(f"Error: No benchmark .json files found in {BENCHMARK_DIR}.")
    exit(1)

all_results = []
session = requests.Session()

# Pre-load BERTScore model once to avoid re-loading
# This will trigger the model download on first run.
print("Loading BERTScore model (this may take a moment on first run)...")
try:
    calculate_bertscore("pre-warm", "pre-warm")
    print("BERTScore model loaded.")
except Exception as e:
    print(f"Could not pre-warm BERTScore: {e}. It will load on the first query.")


# Main loop over each benchmark file (cuad.json, maud.json, etc.)
for bench_path in tqdm(bench_paths, desc="Running benchmarks"):
    bench_name = os.path.basename(bench_path)
    try:
        with open(bench_path, "r", encoding="utf-8") as f:
            bench_data = json.load(f)
    except Exception as e:
        print(f"\nError loading benchmark file {bench_path}: {e}")
        continue

    tests = bench_data.get("tests", [])
    if not tests:
        print(f"\nWarning: No 'tests' found in {bench_path}")
        continue

    # --- NEW: Group questions by file_path ---
    questions_by_file = defaultdict(list)
    for item in tests:
        q = item.get("query")
        # Use the *first* snippet as ground truth
        snippet = item.get("snippets", [{}])[0]
        gt = snippet.get("answer")
        file_path = snippet.get("file_path") # e.g., "cuad/file.txt"

        if not q or not gt or not file_path:
            continue
        
        questions_by_file[file_path].append({"query": q, "gold_answer": gt})
    # ----------------------------------------

    if not questions_by_file:
        print(f"\nWarning: No valid questions with file_paths found in {bench_name}")
        continue

    # --- UPDATED: Loop over ALL files ---
    file_pbar_desc = f"  - {bench_name} ({len(questions_by_file)} files)"
    file_pbar = tqdm(
        questions_by_file.items(), # Removed itertools.islice to run ALL files
        desc=file_pbar_desc,
        total=len(questions_by_file), # Updated total to reflect all files
        leave=False
    )

    for file_path, qa_list in file_pbar:
        
        # Construct the full path relative to the corpus directory
        full_corpus_path = os.path.join(CORPUS_DIR, file_path)
        
        if not os.path.exists(full_corpus_path):
            print(f"\nSkipping file: Not found at {full_corpus_path}")
            continue
            
        # --- NEW: Step 1 - Clear RAG Index ---
        try:
            resp_clear = session.post(CLEAR_ENDPOINT)
            resp_clear.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"\nFATAL: Failed to clear index. Check app.py. Error: {e}")
            print("Aborting benchmark run.")
            break # Stop processing this benchmark
            
        # --- NEW: Step 2 - Load *Single* File into RAG ---
        try:
            resp_load = session.post(LOAD_FILE_ENDPOINT, json={"file_path": full_corpus_path})
            resp_load.raise_for_status()
            if resp_load.json().get("status") != "success":
                print(f"\nServer failed to load file {full_corpus_path}")
                continue # Skip this file
        except requests.exceptions.RequestException as e:
            print(f"\nFATAL: Failed to load file. Check app.py. Error: {e}")
            continue # Skip this file

        # --- UPDATED: Step 3 - Ask ALL Questions for this file ---
        for item in qa_list: # Removed slicing to run ALL questions
            q = item["query"]
            gt = item["gold_answer"]
            
            start_time = time.time()
            try:
                response = session.post(QUERY_ENDPOINT, json={"query": q})
                response.raise_for_status()
                data = response.json()
                
                # ------ FIX IS HERE ------
                # Was: data.get("response", "")
                model_answer = data.get("answer", "") 
                # -------------------------
                
                explanation = data.get("explanation", "")
                retrieved_sources = data.get("sources", []) 
                
            except requests.exceptions.RequestException as e:
                print(f"\nError querying server for: {q}\n{e}")
                model_answer = "[ERROR: No response from server]"
                explanation = f"[ERROR: {e}]"
                retrieved_sources = []
                data = {"error": str(e)}
            
            latency = time.time() - start_time

            # --- Calculate all metrics ---
            exact = 1.0 if model_answer.strip().lower() == gt.strip().lower() else 0.0
            f1 = token_f1_score(model_answer, gt)
            chrf = calculate_chrf(model_answer, gt) 
            bleu = calculate_bleu(model_answer, gt) 
            bertscore_f1 = calculate_bertscore(model_answer, gt) # <-- ADDED
            
            # Recall/MRR check if the *correct file* was retrieved.
            # 'file_path' (e.g., "cuad/file.txt") is the gold ID.
            recall_5 = recall_at_k(retrieved_sources, file_path, k=5)
            mrr = mean_reciprocal_rank(retrieved_sources, file_path)

            all_results.append({
                "benchmark_file": bench_name,
                "source_file": file_path,
                "question": q,
                "gold_answer": gt,
                "model_answer": model_answer,
                "retrieved_sources": retrieved_sources,
                "latency_sec": latency,
                "exact_match": exact,
                "token_f1": f1,
                "chrf_score": chrf, 
                "bleu_score": bleu, 
                "bert_score_f1": bertscore_f1, # <-- ADDED
                "recall_at_5": recall_5,
                "mrr": mrr,
                "raw_response": data
            })

            time.sleep(SECONDS_BETWEEN_QUERIES)

# ---- 3. Save Detailed Results ----
try:
    with open(SAVE_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSuccessfully saved detailed results to {SAVE_JSON}")
except Exception as e:
    print(f"\nError saving JSON results: {e}")

try:
    if all_results:
        keys = all_results[0].keys()
        with open(SAVE_CSV, "w", encoding="utf-8", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_results)
        print(f"Successfully saved CSV summary to {SAVE_CSV}")
except Exception as e:
    print(f"\nError saving CSV results: {e}")

# ---- 4. Calculate Aggregate Metrics and Save Summary Report ----
print("Calculating summary report...")
benchmark_scores = defaultdict(lambda: defaultdict(list))
all_metrics = defaultdict(list)

for res in all_results:
    bench_name = res["benchmark_file"]
    metrics_to_avg = ["latency_sec", "exact_match", "token_f1", "chrf_score", "bleu_score", "bert_score_f1", "recall_at_5", "mrr"] # <-- ADDED metrics
    
    for metric in metrics_to_avg:
        benchmark_scores[bench_name][metric].append(res[metric])
        all_metrics[metric].append(res[metric])

summary_lines = ["--- LegalBench Targeted Evaluation Summary ---"]

# Calculate scores for each benchmark
for bench_name, metrics in benchmark_scores.items():
    summary_lines.append(f"\nBenchmark: {bench_name}")
    summary_lines.append("-" * (len(bench_name) + 12))
    summary_lines.append(f"Recall@5:       {np.mean(metrics['recall_at_5']) * 100:.2f}%")
    summary_lines.append(f"MRR:            {np.mean(metrics['mrr']):.3f}")
    summary_lines.append(f"BERTScore F1:   {np.mean(metrics['bert_score_f1']):.4f}") # <-- ADDED
    summary_lines.append(f"chrF Score:     {np.mean(metrics['chrf_score']):.2f}") 
    summary_lines.append(f"BLEU Score:     {np.mean(metrics['bleu_score']):.2f}") 
    summary_lines.append(f"Token F1:       {np.mean(metrics['token_f1']) * 100:.2f}%")
    summary_lines.append(f"Exact Match:    {np.mean(metrics['exact_match']) * 100:.2f}%")
    summary_lines.append(f"Avg Latency:    {np.mean(metrics['latency_sec']):.2f} sec")
    summary_lines.append(f"(Total Questions: {len(metrics['exact_match'])})")

# Calculate overall average scores
if all_metrics:
    summary_lines.append("\n\n--- Overall Average Scores ---")
    summary_lines.append(f"Recall@5:       {np.mean(all_metrics['recall_at_5']) * 100:.2f}%")
    summary_lines.append(f"MRR:            {np.mean(all_metrics['mrr']):.3f}")
    summary_lines.append(f"BERTScore F1:   {np.mean(all_metrics['bert_score_f1']):.4f}") # <-- ADDED
    summary_lines.append(f"chrF Score:     {np.mean(all_metrics['chrf_score']):.2f}") 
    summary_lines.append(f"BLEU Score:     {np.mean(all_metrics['bleu_score']):.2f}") 
    summary_lines.append(f"Token F1:       {np.mean(all_metrics['token_f1']) * 100:.2f}%")
    summary_lines.append(f"Exact Match:    {np.mean(all_metrics['exact_match']) * 100:.2f}%")
    summary_lines.append(f"Avg Latency:    {np.mean(all_metrics['latency_sec']):.2f} sec")
    summary_lines.append(f"(Total Questions: {len(all_metrics['exact_match'])})")
else:
    summary_lines.append("\n\n--- No results to average. ---")

# Add note about missing metrics
summary_lines.append("\n\n--- Metric Notes ---")
summary_lines.append(" - 'Grounded Answers' and 'Hallucination Rate' require a separate evaluation model")
summary_lines.append("   (e.g., an LLM judge) to check if the answer is supported by the context.")
summary_lines.append("   These are not calculated by this script.")
summary_lines.append(" - chrF/BLEU scores range from 0-100. Higher is better.")
summary_lines.append(" - BERTScore F1 ranges from 0-1. Higher is better. (A score of 0.90+ is excellent).")


# Save and print the summary
summary_report = "\n".join(summary_lines)
try:
    with open(SAVE_SUMMARY, "w", encoding="utf-8") as f:
        f.write(summary_report)
    print(f"Successfully saved summary report to {SAVE_SUMMARY}")
except Exception as e:
    print(f"\nError saving summary report: {e}")

print("\n" + summary_report)
print("\nTargeted evaluation complete.")

