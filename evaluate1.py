"""
STAGE 1: GENERATE RESULTS

Usage:
1) Make sure your Flask app (app.py) is running.
2) Run this script: python evaluate_stage_1_generate.py

What it does:
- Runs all 1,219 queries against your RAG server as fast as possible.
- Calculates "fast" metrics (Recall, MRR, Latency, Exact Match).
- Saves all raw text (question, gold answer, model answer, sources)
  to 'evaluation_results_stage1.json'.
"""

import os
import glob
import json
import time
import requests
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import itertools
import warnings
from transformers import logging as transformers_logging

# --- Suppress verbose warnings ---
transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)
# ---------------------------------

# ---- CONFIG ----
BASE_URL = "http://localhost:5001"
CLEAR_ENDPOINT = f"{BASE_URL}/clear_index"
LOAD_FILE_ENDPOINT = f"{BASE_URL}/load_file"
QUERY_ENDPOINT = f"{BASE_URL}/query"

CORPUS_DIR = "LegalBench-RAG/corpus"
BENCHMARK_DIR = "LegalBench-RAG/benchmarks"
SAVE_JSON = "evaluation_results_stage1.json" # <-- STAGE 1 OUTPUT

# Set to 1.2 seconds for a 50 QPM rate (60 / 1.2 = 50)
SECONDS_BETWEEN_QUERIES = 1.2

# ---- Helper Functions (Fast Metrics) ----

def recall_at_k(retrieved_sources, gold_file_path, k=5):
    if not retrieved_sources or not gold_file_path:
        return 0.0
    top_k_sources = retrieved_sources[:k]
    relevant_found = False
    for source_path in top_k_sources:
        norm_source = source_path.replace("\\", "/")
        norm_gold = gold_file_path.replace("\\", "/")
        if norm_gold in norm_source:
            relevant_found = True
            break
    return 1.0 if relevant_found else 0.0

def mean_reciprocal_rank(retrieved_sources, gold_file_path):
    if not retrieved_sources or not gold_file_path:
        return 0.0
    norm_gold = gold_file_path.replace("\\", "/")
    for i, source_path in enumerate(retrieved_sources):
        rank = i + 1
        norm_source = source_path.replace("\\", "/")
        if norm_gold in norm_source:
            return 1.0 / rank
    return 0.0

# ---- Run Benchmarks (Generation) ----
print(f"--- STAGE 1: GENERATING RESULTS ---")
print(f"Running targeted benchmarks from {BENCHMARK_DIR}...")
bench_paths = glob.glob(os.path.join(BENCHMARK_DIR, "*.json"))

if not bench_paths:
    print(f"Error: No benchmark .json files found in {BENCHMARK_DIR}.")
    exit(1)

all_results = []
session = requests.Session()

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

    # --- Group questions by file_path ---
    questions_by_file = defaultdict(list)
    for item in tests:
        q = item.get("query")
        snippet = item.get("snippets", [{}])[0]
        gt = snippet.get("answer")
        file_path = snippet.get("file_path")
        if not q or not gt or not file_path:
            continue
        questions_by_file[file_path].append({"query": q, "gold_answer": gt})

    if not questions_by_file:
        print(f"\nWarning: No valid questions with file_paths found in {bench_name}")
        continue

    # --- Loop over ALL files ---
    file_pbar_desc = f"  - {bench_name} ({len(questions_by_file)} files)"
    file_pbar = tqdm(
        questions_by_file.items(),
        desc=file_pbar_desc,
        total=len(questions_by_file),
        leave=False
    )

    for file_path, qa_list in file_pbar:
        full_corpus_path = os.path.join(CORPUS_DIR, file_path)
        if not os.path.exists(full_corpus_path):
            print(f"\nSkipping file: Not found at {full_corpus_path}")
            continue
            
        # --- Step 1 - Clear RAG Index ---
        try:
            resp_clear = session.post(CLEAR_ENDPOINT)
            resp_clear.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"\nFATAL: Failed to clear index. Check app.py. Error: {e}")
            break
            
        # --- Step 2 - Load *Single* File into RAG ---
        try:
            resp_load = session.post(LOAD_FILE_ENDPOINT, json={"file_path": full_corpus_path})
            resp_load.raise_for_status()
            if resp_load.json().get("status") != "success":
                print(f"\nServer failed to load file {full_corpus_path}")
                continue
        except requests.exceptions.RequestException as e:
            print(f"\nFATAL: Failed to load file. Check app.py. Error: {e}")
            continue

        # --- Step 3 - Ask ALL Questions for this file ---
        for item in qa_list:
            q = item["query"]
            gt = item["gold_answer"]
            
            start_time = time.time()
            try:
                response = session.post(QUERY_ENDPOINT, json={"query": q})
                response.raise_for_status()
                data = response.json()
                
                model_answer = data.get("answer", "") 
                retrieved_sources = data.get("sources", []) 
                
            except requests.exceptions.RequestException as e:
                print(f"\nError querying server for: {q}\n{e}")
                model_answer = "[ERROR: No response from server]"
                retrieved_sources = []
            
            latency = time.time() - start_time

            # --- Calculate "fast" metrics ---
            exact = 1.0 if model_answer.strip().lower() == gt.strip().lower() else 0.0
            recall_5 = recall_at_k(retrieved_sources, file_path, k=5)
            mrr = mean_reciprocal_rank(retrieved_sources, file_path)

            # --- Save RAW data (no slow metrics yet) ---
            all_results.append({
                "benchmark_file": bench_name,
                "source_file": file_path,
                "question": q,
                "gold_answer": gt,
                "model_answer": model_answer,
                "retrieved_sources": retrieved_sources,
                "latency_sec": latency,
                "exact_match": exact,
                "recall_at_5": recall_5,
                "mrr": mrr
                # "raw_response" is removed to save space, we have the answer
            })

            time.sleep(SECONDS_BETWEEN_QUERIES)

# ---- Save Stage 1 Results ----
try:
    with open(SAVE_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n--- STAGE 1 COMPLETE ---")
    print(f"Successfully saved {len(all_results)} results to {SAVE_JSON}")
    print(f"\nNow run 'python evaluate_stage_2_calculate.py' to get full metrics.")
except Exception as e:
    print(f"\nError saving JSON results: {e}")
