"""
STAGE 2: CALCULATE METRICS

Usage:
1) Make sure 'evaluation_results_stage1.json' exists.
2) Run this script: python evaluate_stage_2_calculate.py
   (This does NOT need the Flask server to be running)

What it does:
- Loads 'evaluation_results_stage1.json'.
- Calculates the "slow" metrics (BERTScore, BLEU, chrF, Token F1).
- Saves the final, complete results to:
  - 'evaluation_results_full.json'
  - 'evaluation_results_full.csv'
- Prints the final summary report to 'evaluation_summary.txt'
"""

import json
import csv
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import sacrebleu  # For chrF and BLEU scores
import bert_score # For BERTScore
import warnings
from transformers import logging as transformers_logging

# --- Suppress verbose warnings ---
transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)
# ---------------------------------

# ---- Input/Output Files ----
STAGE1_RESULTS_FILE = "evaluation_results_stage1.json"
SAVE_FULL_JSON = "evaluation_results_full.json"
SAVE_FULL_CSV = "evaluation_results_full.csv"
SAVE_SUMMARY = "evaluation_summary.txt"

# ---- Helper Functions (Slow Metrics) ----

def token_f1_score(pred, gold):
    if not pred or not gold:
        return 0.0
    ptoks = pred.lower().split()
    gtoks = gold.lower().split()
    common = set(ptoks) & set(gtoks)
    if not gtoks or not ptoks:
        return 0.0
    precision = len(common) / (len(ptoks) + 1e-8)
    recall = len(common) / (len(gtoks) + 1e-8)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def calculate_chrf(pred, gold):
    if not pred or not gold:
        return 0.0
    return sacrebleu.corpus_chrf([pred], [[gold]]).score

def calculate_bleu(pred, gold):
    if not pred or not gold:
        return 0.0
    return sacrebleu.corpus_bleu([pred], [[gold]]).score

def calculate_bertscore(preds, golds):
    """Calculates BERTScore F1 in batches for efficiency."""
    try:
        (P, R, F1) = bert_score.score(preds, golds, lang='en', verbose=False)
        return F1.tolist() # Return a list of F1 scores
    except Exception as e:
        print(f"\nWarning: BERTScore calculation failed. {e}")
        return [0.0] * len(preds)

# ---- Main Calculation Script ----

print(f"--- STAGE 2: CALCULATING METRICS ---")
print(f"Loading results from {STAGE1_RESULTS_FILE}...")

try:
    with open(STAGE1_RESULTS_FILE, "r", encoding="utf-8") as f:
        all_results = json.load(f)
except FileNotFoundError:
    print(f"Error: File not found: {STAGE1_RESULTS_FILE}")
    print("Please run 'evaluate_stage_1_generate.py' first.")
    exit(1)
except json.JSONDecodeError:
    print(f"Error: Could not read {STAGE1_RESULTS_FILE}. Is it corrupted?")
    exit(1)

if not all_results:
    print("No results found in file. Exiting.")
    exit(1)

print(f"Loaded {len(all_results)} results. Calculating slow metrics...")

# --- Pre-load BERTScore model ---
print("Loading BERTScore model (this may take a moment on first run)...")
try:
    calculate_bertscore(["pre-warm"], [["pre-warm"]])
    print("BERTScore model loaded.")
except Exception as e:
    print(f"Could not pre-warm BERTScore: {e}.")
    pass

# --- Calculate metrics in batches (much faster for BERTScore) ---
batch_size = 16
full_results = []
all_preds = [res["model_answer"] for res in all_results]
all_golds = [[res["gold_answer"]] for res in all_results] # Note: sacrebleu/bert_score expect list-of-lists

bert_scores_f1 = []

# Process in batches for BERTScore
for i in tqdm(range(0, len(all_results), batch_size), desc="Calculating BERTScore"):
    batch_preds = all_preds[i : i + batch_size]
    batch_golds = all_golds[i : i + batch_size]
    
    bert_scores_f1.extend(calculate_bertscore(batch_preds, batch_golds))

# Process all metrics for each result
for i, res in enumerate(tqdm(all_results, desc="Calculating other metrics")):
    pred = res["model_answer"]
    gold = res["gold_answer"]

    # Add the new "slow" metrics
    res["token_f1"] = token_f1_score(pred, gold)
    res["chrf_score"] = calculate_chrf(pred, gold)
    res["bleu_score"] = calculate_bleu(pred, gold)
    res["bert_score_f1"] = bert_scores_f1[i]
    
    full_results.append(res)

print("All metrics calculated.")

# ---- Save Full Results (JSON and CSV) ----
try:
    with open(SAVE_FULL_JSON, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    print(f"Successfully saved full results to {SAVE_FULL_JSON}")
except Exception as e:
    print(f"\nError saving full JSON results: {e}")

try:
    if full_results:
        keys = full_results[0].keys()
        with open(SAVE_FULL_CSV, "w", encoding="utf-8", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(full_results)
        print(f"Successfully saved full CSV summary to {SAVE_FULL_CSV}")
except Exception as e:
    print(f"\nError saving full CSV results: {e}")

# ---- Calculate Aggregate Metrics and Save Summary Report ----
print("Calculating summary report...")
benchmark_scores = defaultdict(lambda: defaultdict(list))
all_metrics = defaultdict(list)

for res in full_results:
    bench_name = res["benchmark_file"]
    metrics_to_avg = [
        "latency_sec", "exact_match", "token_f1", "chrf_score", 
        "bleu_score", "bert_score_f1", "recall_at_5", "mrr"
    ]
    
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
    summary_lines.append(f"BERTScore F1:   {np.mean(metrics['bert_score_f1']):.4f}")
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
    summary_lines.append(f"BERTScore F1:   {np.mean(all_metrics['bert_score_f1']):.4f}")
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
summary_lines.append(" - 'Grounded Answers' / 'Hallucination Rate' (Faithfulness) metrics were not calculated.")
summary_lines.append("   This requires a separate 'LLM-as-a-Judge' step to score the model answer")
summary_lines.append("   against the retrieved sources, which is a different, more complex evaluation.")
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
print("\n--- STAGE 2 COMPLETE ---")
print("Targeted evaluation complete.")
