"""
ABLATION STUDY: Chunk Size and Overlap Analysis

Tests the impact of different chunk sizes and overlap values on
retrieval and generation performance using LegalBench-RAG benchmarks.

Configurations tested:
- Chunk sizes: [150, 250, 350, 500, 750] words
- Overlap values: [0, 25, 50, 75] words

Usage:
1) Start the Flask app: python app.py
2) Run this script: python ablation_chunk_size.py

Output:
- ablation_results.json: Full results per configuration
- ablation_summary.txt: Formatted summary table for paper
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
import warnings
from transformers import logging as transformers_logging

# Suppress warnings
transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)

# ---- CONFIG ----
BASE_URL = "http://localhost:5001"
CLEAR_ENDPOINT = f"{BASE_URL}/clear_index"
LOAD_FILE_ENDPOINT = f"{BASE_URL}/load_file"
QUERY_ENDPOINT = f"{BASE_URL}/query"

CORPUS_DIR = "LegalBench-RAG/corpus"
BENCHMARK_DIR = "LegalBench-RAG/benchmarks"

# Ablation configurations
CHUNK_SIZES = [150, 250, 350, 500, 750]
OVERLAP_VALUES = [0, 25, 50, 75]

# Rate limiting
SECONDS_BETWEEN_QUERIES = 1.2

# Limit queries per benchmark for faster ablation (set to None for full run)
MAX_QUERIES_PER_BENCHMARK = 50  # Set to None to evaluate all queries

# Output files
SAVE_JSON = "ablation_results.json"
SAVE_SUMMARY = "ablation_summary.txt"


# ---- Metric Functions ----

def recall_at_k(retrieved_sources, gold_file_path, k=5):
    """Calculate Recall@K: whether the gold document appears in top-K results."""
    if not retrieved_sources or not gold_file_path:
        return 0.0
    top_k_sources = retrieved_sources[:k]
    for source_path in top_k_sources:
        norm_source = source_path.replace("\\", "/")
        norm_gold = gold_file_path.replace("\\", "/")
        if norm_gold in norm_source:
            return 1.0
    return 0.0


def mean_reciprocal_rank(retrieved_sources, gold_file_path):
    """Calculate MRR: reciprocal rank of the first relevant result."""
    if not retrieved_sources or not gold_file_path:
        return 0.0
    norm_gold = gold_file_path.replace("\\", "/")
    for i, source_path in enumerate(retrieved_sources):
        norm_source = source_path.replace("\\", "/")
        if norm_gold in norm_source:
            return 1.0 / (i + 1)
    return 0.0


def run_ablation_config(chunk_size, overlap, bench_paths, session):
    """Run evaluation for a single chunk_size + overlap configuration."""
    config_results = []
    config_metrics = defaultdict(list)

    for bench_path in bench_paths:
        bench_name = os.path.basename(bench_path)
        try:
            with open(bench_path, "r", encoding="utf-8") as f:
                bench_data = json.load(f)
        except Exception as e:
            print(f"\nError loading {bench_path}: {e}")
            continue

        tests = bench_data.get("tests", [])
        if not tests:
            continue

        # Group questions by file
        questions_by_file = defaultdict(list)
        for item in tests:
            q = item.get("query")
            snippet = item.get("snippets", [{}])[0]
            gt = snippet.get("answer")
            file_path = snippet.get("file_path")
            if not q or not gt or not file_path:
                continue
            questions_by_file[file_path].append({"query": q, "gold_answer": gt})

        query_count = 0
        for file_path, qa_list in questions_by_file.items():
            if MAX_QUERIES_PER_BENCHMARK and query_count >= MAX_QUERIES_PER_BENCHMARK:
                break

            full_corpus_path = os.path.join(CORPUS_DIR, file_path)
            if not os.path.exists(full_corpus_path):
                continue

            # Clear index
            try:
                session.post(CLEAR_ENDPOINT).raise_for_status()
            except Exception:
                continue

            # Load file with specific chunk_size and overlap
            try:
                resp = session.post(LOAD_FILE_ENDPOINT, json={
                    "file_path": full_corpus_path,
                    "chunk_size": chunk_size,
                    "overlap": overlap
                })
                resp.raise_for_status()
                load_result = resp.json()
                num_chunks = load_result.get("chunk_count", 0)
            except Exception:
                continue

            for item in qa_list:
                if MAX_QUERIES_PER_BENCHMARK and query_count >= MAX_QUERIES_PER_BENCHMARK:
                    break

                q = item["query"]
                gt = item["gold_answer"]

                start_time = time.time()
                try:
                    response = session.post(QUERY_ENDPOINT, json={"query": q})
                    response.raise_for_status()
                    data = response.json()
                    model_answer = data.get("answer", "")
                    retrieved_sources = data.get("sources", [])
                except Exception:
                    model_answer = "[ERROR]"
                    retrieved_sources = []

                latency = time.time() - start_time

                recall_5 = recall_at_k(retrieved_sources, file_path, k=5)
                mrr = mean_reciprocal_rank(retrieved_sources, file_path)

                config_metrics["recall_at_5"].append(recall_5)
                config_metrics["mrr"].append(mrr)
                config_metrics["latency"].append(latency)
                config_metrics["num_chunks"].append(num_chunks)

                query_count += 1
                time.sleep(SECONDS_BETWEEN_QUERIES)

    # Calculate aggregate metrics for this configuration
    return {
        "chunk_size": chunk_size,
        "overlap": overlap,
        "total_queries": len(config_metrics["recall_at_5"]),
        "recall_at_5": float(np.mean(config_metrics["recall_at_5"])) if config_metrics["recall_at_5"] else 0.0,
        "mrr": float(np.mean(config_metrics["mrr"])) if config_metrics["mrr"] else 0.0,
        "avg_latency": float(np.mean(config_metrics["latency"])) if config_metrics["latency"] else 0.0,
        "avg_chunks_per_doc": float(np.mean(config_metrics["num_chunks"])) if config_metrics["num_chunks"] else 0.0,
    }


# ---- Main Execution ----
if __name__ == "__main__":
    print("=" * 60)
    print("ABLATION STUDY: Chunk Size & Overlap Analysis")
    print("=" * 60)

    bench_paths = glob.glob(os.path.join(BENCHMARK_DIR, "*.json"))
    if not bench_paths:
        print(f"Error: No benchmark files found in {BENCHMARK_DIR}")
        exit(1)

    print(f"Found {len(bench_paths)} benchmark files")
    print(f"Chunk sizes to test: {CHUNK_SIZES}")
    print(f"Overlap values to test: {OVERLAP_VALUES}")
    print(f"Total configurations: {len(CHUNK_SIZES) * len(OVERLAP_VALUES)}")

    if MAX_QUERIES_PER_BENCHMARK:
        print(f"Max queries per benchmark: {MAX_QUERIES_PER_BENCHMARK}")

    session = requests.Session()
    all_ablation_results = []

    total_configs = len(CHUNK_SIZES) * len(OVERLAP_VALUES)
    config_idx = 0

    for chunk_size in CHUNK_SIZES:
        for overlap in OVERLAP_VALUES:
            # Skip invalid configs where overlap >= chunk_size
            if overlap >= chunk_size:
                print(f"\nSkipping invalid config: chunk_size={chunk_size}, overlap={overlap}")
                continue

            config_idx += 1
            print(f"\n--- Config {config_idx}/{total_configs}: "
                  f"chunk_size={chunk_size}, overlap={overlap} ---")

            result = run_ablation_config(chunk_size, overlap, bench_paths, session)
            all_ablation_results.append(result)

            print(f"  Recall@5: {result['recall_at_5']*100:.2f}%, "
                  f"MRR: {result['mrr']:.3f}, "
                  f"Latency: {result['avg_latency']:.2f}s, "
                  f"Avg Chunks: {result['avg_chunks_per_doc']:.1f}")

    # ---- Save Results ----
    with open(SAVE_JSON, "w", encoding="utf-8") as f:
        json.dump(all_ablation_results, f, indent=2)
    print(f"\nResults saved to {SAVE_JSON}")

    # ---- Generate Summary Table ----
    summary_lines = [
        "=" * 80,
        "ABLATION STUDY: Impact of Chunk Size and Overlap on Retrieval Performance",
        "=" * 80,
        "",
        f"{'Chunk Size':>12} | {'Overlap':>8} | {'Recall@5':>10} | {'MRR':>8} | {'Latency(s)':>11} | {'Avg Chunks':>11}",
        "-" * 80,
    ]

    for r in all_ablation_results:
        summary_lines.append(
            f"{r['chunk_size']:>12} | {r['overlap']:>8} | "
            f"{r['recall_at_5']*100:>9.2f}% | {r['mrr']:>8.3f} | "
            f"{r['avg_latency']:>11.3f} | {r['avg_chunks_per_doc']:>11.1f}"
        )

    summary_lines.extend([
        "",
        "-" * 80,
        f"Total configurations tested: {len(all_ablation_results)}",
        f"Queries per config limit: {MAX_QUERIES_PER_BENCHMARK or 'All'}",
        "",
        "Analysis:",
        "- Smaller chunks increase granularity but may lose context.",
        "- Larger chunks preserve context but reduce retrieval precision.",
        "- Overlap helps maintain context continuity at chunk boundaries.",
        "- The optimal configuration balances retrieval accuracy with chunk count.",
    ])

    summary = "\n".join(summary_lines)

    with open(SAVE_SUMMARY, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"\n{summary}")
    print(f"\nSummary saved to {SAVE_SUMMARY}")
    print("\nAblation study complete.")
