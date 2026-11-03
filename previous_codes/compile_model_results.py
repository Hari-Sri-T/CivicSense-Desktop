"""
COMPILE MODEL COMPARISON RESULTS
==================================
Aggregates all per-model result files from model_results/ into a single
comparison table for inclusion in the paper.

Usage:
  python compile_model_results.py

Input:  model_results/result_*.json  (one per model)
Output: model_comparison_summary.txt
        model_comparison_results.json
"""

import os
import json
import glob

RESULTS_DIR   = "model_results"
SUMMARY_FILE  = "model_comparison_summary.txt"
JSON_OUT      = "model_comparison_results.json"

# Human-readable model display names (maps filename key -> paper label)
MODEL_LABELS = {
    "gemma-4-12b-qat":             "Gemma-4-12B-QAT (Google, 2026)",
    "phi-4":                       "Phi-4 14B (Microsoft, 2025)",
    "deepseek-r1-0528-qwen3-8b":   "DeepSeek-R1 Qwen3-8B (2025)",
    "ministral-3-14b-reasoning":   "Ministral-3 14B Reasoning (Mistral, 2025)",
    "nemotron-3-nano-4b":          "Nemotron-3 Nano-4B (NVIDIA, 2025)",
    "ministral-3-3b":              "Ministral-3 3B (Mistral, 2025)",
    "qwen3.5-9b":                  "Qwen3.5-9B (Alibaba, 2025)",
}


def load_results():
    result_files = glob.glob(os.path.join(RESULTS_DIR, "result_*.json"))
    if not result_files:
        print(f"No result files found in {RESULTS_DIR}/")
        print("Run evaluate_model_comparison.py for each model first.")
        return []

    summaries = []
    for fp in sorted(result_files):
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        summary = data.get("summary", {})
        if summary:
            summaries.append(summary)
            print(f"Loaded: {os.path.basename(fp)}  ({summary['total_queries']} queries)")
    return summaries


def generate_table(summaries):
    lines = [
        "=" * 110,
        "TABLE: LLM Comparison on CivicSense AI RAG Pipeline (LegalBench-RAG Subset)",
        "=" * 110,
        "",
        f"{'Model':<42} | {'Recall@5':>9} | {'MRR':>6} | {'BERTScore':>10} | {'BLEU':>6} | {'chrF':>6} | {'Latency(s)':>11}",
        "-" * 110,
    ]

    for s in summaries:
        model_key   = s.get("model", "unknown")
        label       = MODEL_LABELS.get(model_key, model_key)
        lines.append(
            f"{label:<42} | {s.get('recall_at_5') or 0.0:>8.2f}% | {s.get('mrr') or 0.0:>6.4f} | "
            f"{s.get('bert_score_f1') or 0.0:>10.4f} | {s.get('bleu') or 0.0:>6.2f} | {s.get('chrf') or 0.0:>6.2f} | "
            f"{s.get('avg_latency_sec') or 0.0:>10.3f}s"
        )

    lines.extend([
        "-" * 110,
        "",
        "Notes:",
        "  - Recall@5: Whether the correct source document appears in top-5 retrieved chunks.",
        "  - MRR: Mean Reciprocal Rank of the correct source document.",
        "  - BERTScore F1: Semantic similarity between model answer and gold answer (range 0-1).",
        "  - BLEU/chrF: Lexical overlap metrics (lower due to generative/explanatory style).",
        "  - Latency: Average end-to-end query response time (embedding + retrieval + generation).",
        f"  - Queries per model: {summaries[0]['total_queries'] if summaries else 'N/A'} (subset of LegalBench-RAG)",
        "",
        "Embedding model fixed across all runs: all-MiniLM-L6-v2 (384-dim)",
        "Chunk size fixed across all runs: 350 words, 50-word overlap, Top-K=5",
    ])

    return "\n".join(lines)


def main():
    print("=" * 65)
    print("  COMPILING MODEL COMPARISON RESULTS")
    print("=" * 65)

    summaries = load_results()
    if not summaries:
        return

    # Sort by BERTScore descending
    summaries.sort(key=lambda x: x.get("bert_score_f1") or 0.0, reverse=True)

    table = generate_table(summaries)
    print("\n" + table)

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(table)
    print(f"\n✅ Summary saved to {SUMMARY_FILE}")

    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"✅ JSON saved to {JSON_OUT}")

    # Print best model recommendation
    if summaries:
        best = summaries[0]
        print(f"\n🏆 Best model by BERTScore F1: {best['model']}  ({best['bert_score_f1']})")


if __name__ == "__main__":
    main()
