"""
CivicSense AI — Model Comparison Compiler v2
Reads result_v2_*.json files and produces a clean comparison table + CSV.
"""
import os, json, glob, csv

RESULTS_DIR = "model_results"

def main():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "result_v2_*.json")))
    if not files:
        print(f"No result_v2_*.json files found in {RESULTS_DIR}/")
        return

    summaries = []
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        s = data.get("summary", {})
        if isinstance(s, dict):
            summaries.append(s)
            print(f"Loaded: {os.path.basename(fp)}  ({s.get('total_queries',0)} queries, {s.get('error_rate_pct',0)}% errors)")

    if not summaries:
        print("No valid summaries found.")
        return

    # Sort by BERTScore, fall back to corpus_bleu
    summaries.sort(
        key=lambda x: (x.get("bert_score_f1") or 0.0, x.get("corpus_bleu") or 0.0),
        reverse=True
    )

    cols = [
        ("Model",          "model",            "<42", "s"),
        ("Errors%",        "error_rate_pct",   ">7",  ".1f"),
        ("BERTScore",      "bert_score_f1",    ">9",  ".4f"),
        ("Corpus BLEU",    "corpus_bleu",      ">11", ".2f"),
        ("Corpus chrF",    "corpus_chrf",      ">11", ".2f"),
        ("Token F1%",      "token_f1_pct",     ">9",  ".1f"),
        ("Exact Match%",   "exact_match_pct",  ">12", ".1f"),
        ("Avg Latency(s)", "avg_latency_sec",  ">14", ".2f"),
    ]

    # Header
    header = " | ".join(f"{label:{col[2]}}" for label, _, col, _ in cols)
    sep    = "-" * len(header)
    print("\n\n" + "=" * len(header))
    print("  CivicSense AI — LLM Comparison (LegalBench-RAG, 24 queries each)")
    print("=" * len(header))
    print(header)
    print(sep)

    rows = []
    for s in summaries:
        cells = []
        for label, key, width, fmt in cols:
            val = s.get(key)
            if val is None:
                cells.append(f"{'N/A':>{width.strip('<>')}}")
            else:
                try:
                    cells.append(f"{val:{width}{fmt}}")
                except Exception:
                    cells.append(f"{str(val):{width}s}")
        row_str = " | ".join(cells)
        print(row_str)
        rows.append([s.get(key, "") for _, key, _, _ in cols])

    print("=" * len(header))
    print("\nNotes:")
    print("  Errors%     : Queries where JSON could not be recovered (even after json_repair).")
    print("  BERTScore   : Semantic similarity (RoBERTa-large). Computed on valid answers only.")
    print("  Corpus BLEU : Lexical overlap. Computed at corpus level on valid answers only.")
    print("  Corpus chrF : Character n-gram F1. More robust than BLEU for short answers.")
    print("  Token F1    : Unigram overlap between model and gold answer tokens.")
    print(f"\n  Queries per model : {RESULTS_DIR.replace('model_results','')}{24}")
    print("  Embedding model   : all-MiniLM-L6-v2 (384-dim, CPU)")
    print("  Chunk size        : 350 words, 50-word overlap, Top-K=5")

    # CSV export
    csv_path = os.path.join(RESULTS_DIR, "evaluation_summary_v2.csv")
    header_labels = [label for label, _, _, _ in cols]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header_labels)
        for s in summaries:
            w.writerow([s.get(key, "") for _, key, _, _ in cols])
    print(f"\nCSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
