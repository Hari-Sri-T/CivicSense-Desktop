"""
CIVICSENSE AI — AUTOMATED MULTI-MODEL EVALUATION v2 — PHASE 2
=============================================================
Identical to evaluate_v2.py EXCEPT the Flask /query endpoint
receives an extra flag `no_reasoning=True` that swaps in a
hardened system prompt forbidding chain-of-thought tokens.

This tests whether stripping reasoning overhead lets verbose
models (Llama, Nemotron) recover accuracy.

Results saved as:  model_results/result_v2_phase2_<model>.json
"""

import os, sys, json, time, subprocess, argparse, glob, warnings
import numpy as np
import requests
import sacrebleu
from collections import defaultdict
from tqdm import tqdm

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

try:
    import bert_score
    BERT_SCORE_AVAILABLE = True
except Exception:
    BERT_SCORE_AVAILABLE = False

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
FLASK_BASE_URL    = "http://localhost:5001"
LM_STUDIO_URL     = "http://localhost:1234/v1/models"
CORPUS_DIR        = "LegalBench-RAG/corpus"
BENCHMARK_DIR     = "LegalBench-RAG/benchmarks"
RESULTS_DIR       = "model_results"
QUERIES_PER_BENCH = 6
CONTEXT_LENGTH    = 8192
API_RETRY_WAIT    = 4
MODEL_LOAD_WAIT   = 15
MODEL_LOAD_POLL   = 3

DEFAULT_MODELS = [
    "qwen2.5-3b-instruct",
    "llama-3.2-3b-instruct",
    "mistralai/ministral-3-3b",
    "google/gemma-3-4b",
    "nvidia/nemotron-3-nano-4b",
]

os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# LM STUDIO HELPERS (identical to Phase 1)
# ─────────────────────────────────────────────

def lms_unload_all():
    print("  [lms] Unloading all models...")
    result = subprocess.run(["lms", "unload", "--all"], capture_output=True, text=True)
    if result.returncode != 0:
        if "No models are currently loaded" in (result.stdout + result.stderr):
            print("  [lms] No models were loaded.")
        else:
            print(f"  [lms] unload warning: {result.stderr.strip()}")
    else:
        print("  [lms] All models unloaded.")
    time.sleep(2)


def lms_load_model(model_key: str):
    print(f"  [lms] Loading {model_key} (context={CONTEXT_LENGTH})...")
    result = subprocess.run(
        ["lms", "load", model_key, "--context-length", str(CONTEXT_LENGTH), "--gpu", "max", "-y"],
        capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        print(f"  [lms] load error: {result.stderr.strip()}")
        return False
    deadline = time.time() + MODEL_LOAD_WAIT * 10
    while time.time() < deadline:
        try:
            r = requests.get(LM_STUDIO_URL, timeout=5)
            models = r.json().get("data", [])
            if models:
                print(f"  [lms] Loaded: {models[0].get('id', '')}")
                time.sleep(2)
                return True
        except Exception:
            pass
        time.sleep(MODEL_LOAD_POLL)
    print("  [lms] Timed out waiting for model.")
    return False


def get_active_model() -> str:
    try:
        r = requests.get(LM_STUDIO_URL, timeout=3)
        models = r.json().get("data", [])
        return models[0].get("id", "unknown") if models else "none"
    except Exception:
        return "error"

# ─────────────────────────────────────────────
# FLASK / RAG HELPERS
# ─────────────────────────────────────────────

def flask_ok() -> bool:
    try:
        requests.get(f"{FLASK_BASE_URL}/", timeout=3)
        return True
    except Exception:
        return False


def clear_index(session: requests.Session):
    session.post(f"{FLASK_BASE_URL}/clear_index")


def load_file(session: requests.Session, full_path: str) -> bool:
    try:
        resp = session.post(f"{FLASK_BASE_URL}/load_file", json={"file_path": full_path}, timeout=60)
        resp.raise_for_status()
        return resp.json().get("status") == "success"
    except Exception as e:
        print(f"    [load_file] {e}")
        return False


def query_rag(session: requests.Session, query: str, top_k: int = 5):
    """Call Flask /query with no_reasoning=True flag and one automatic retry."""
    payload = {"query": query, "top_k": top_k, "no_reasoning": True}
    for attempt in range(2):
        try:
            resp = session.post(f"{FLASK_BASE_URL}/query", json=payload, timeout=180)
            if resp.status_code == 200:
                return resp.json()
            else:
                print(f"    [query] HTTP {resp.status_code} — {'retrying' if attempt == 0 else 'giving up'}")
        except Exception as e:
            print(f"    [query] Exception: {e} — {'retrying' if attempt == 0 else 'giving up'}")
        if attempt == 0:
            time.sleep(API_RETRY_WAIT)
    return None

# ─────────────────────────────────────────────
# METRIC HELPERS (identical to Phase 1)
# ─────────────────────────────────────────────

def is_error(answer: str) -> bool:
    return answer.strip().startswith("Error:") or answer.strip() == "Empty"


def token_f1(pred: str, gold: str) -> float:
    if not pred or not gold:
        return 0.0
    p_toks = set(pred.lower().split())
    g_toks = set(gold.lower().split())
    common = p_toks & g_toks
    if not common:
        return 0.0
    prec = len(common) / max(len(p_toks), 1)
    rec  = len(common) / max(len(g_toks), 1)
    return 2 * prec * rec / (prec + rec)


def recall_at_k(retrieved_sources, gold_file_path, k=5) -> float:
    if not retrieved_sources or not gold_file_path:
        return 0.0
    norm_gold = gold_file_path.replace("\\", "/")
    for src in retrieved_sources[:k]:
        if norm_gold in src.replace("\\", "/"):
            return 1.0
    return 0.0


def mrr(retrieved_sources, gold_file_path) -> float:
    if not retrieved_sources or not gold_file_path:
        return 0.0
    norm_gold = gold_file_path.replace("\\", "/")
    for i, src in enumerate(retrieved_sources):
        if norm_gold in src.replace("\\", "/"):
            return 1.0 / (i + 1)
    return 0.0

# ─────────────────────────────────────────────
# MAIN EVALUATION LOOP
# ─────────────────────────────────────────────

def evaluate_model(model_key: str):
    print("\n" + "=" * 65)
    print(f"  [PHASE 2] EVALUATING: {model_key}")
    print("=" * 65)

    if not flask_ok():
        print("[ERROR] Flask app not running at", FLASK_BASE_URL)
        return None

    lms_unload_all()
    loaded = lms_load_model(model_key)
    if not loaded:
        print(f"[ERROR] Could not load {model_key}. Skipping.")
        return None

    active = get_active_model()
    print(f"  LM Studio active: {active}\n")

    bench_paths = sorted(glob.glob(os.path.join(BENCHMARK_DIR, "*.json")))
    if not bench_paths:
        print("[ERROR] No benchmark files found.")
        return None

    session     = requests.Session()
    all_results = []

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
            snippet = (item.get("snippets") or [{}])[0]
            gt      = snippet.get("answer")
            fp      = snippet.get("file_path")
            if q and gt and fp:
                questions_by_file[fp].append({"query": q, "gold_answer": str(gt)})

        query_count = 0
        for file_path, qa_list in questions_by_file.items():
            if query_count >= QUERIES_PER_BENCH:
                break

            full_path = os.path.join(CORPUS_DIR, file_path)
            if not os.path.exists(full_path):
                continue

            clear_index(session)
            if not load_file(session, full_path):
                continue

            for item in qa_list:
                if query_count >= QUERIES_PER_BENCH:
                    break

                q  = item["query"]
                gt = item["gold_answer"]

                t0      = time.time()
                data    = query_rag(session, q)
                elapsed = round(time.time() - t0, 4)

                if data is None:
                    model_answer = "Error: API call failed after retry."
                    sources = []
                else:
                    raw_answer   = data.get("answer", "")
                    model_answer = str(raw_answer).strip() if raw_answer else "Empty"
                    sources      = data.get("sources", [])

                all_results.append({
                    "benchmark":    bench_name,
                    "file_path":    file_path,
                    "query":        q,
                    "gold_answer":  gt,
                    "model_answer": model_answer,
                    "latency_sec":  elapsed,
                    "is_error":     is_error(model_answer),
                    "recall_at_5":  recall_at_k(sources, file_path),
                    "mrr":          mrr(sources, file_path),
                    "token_f1":     token_f1(model_answer, gt) if not is_error(model_answer) else 0.0,
                    "exact_match":  1.0 if model_answer.strip().lower() == gt.strip().lower() else 0.0,
                })
                query_count += 1
                time.sleep(0.3)

        safe_key = model_key.replace("/", "_")
        with open(f"{RESULTS_DIR}/backup_v2_phase2_{safe_key}.json", "w", encoding="utf-8") as f:
            json.dump({"summary": "IN PROGRESS", "detailed_results": all_results}, f, indent=2)

    # ── Unload model, compute metrics ──
    print("\n  [lms] Unloading model to free RAM for BERTScore...")
    lms_unload_all()
    time.sleep(3)

    valid_results = [r for r in all_results if not r["is_error"]]
    error_count   = len(all_results) - len(valid_results)
    error_rate    = error_count / max(len(all_results), 1)

    print(f"\n  Total queries:  {len(all_results)}")
    print(f"  Valid answers:  {len(valid_results)}  |  Errors: {error_count}  ({error_rate*100:.1f}%)")

    bert_f1_list = []
    if BERT_SCORE_AVAILABLE and valid_results:
        preds = [r["model_answer"] for r in valid_results]
        golds = [r["gold_answer"]  for r in valid_results]
        print(f"  Computing BERTScore ({len(preds)} predictions)...")
        try:
            _, _, bert_f1 = bert_score.score(preds, golds, lang="en", verbose=False)
            bert_f1_list  = bert_f1.tolist()
            for i, r in enumerate(valid_results):
                r["bert_score_f1"] = round(bert_f1_list[i], 4)
            print(f"  BERTScore done. Mean F1 = {np.mean(bert_f1_list):.4f}")
        except Exception as e:
            print(f"  BERTScore FAILED: {e}")
            for r in valid_results:
                r["bert_score_f1"] = None
    else:
        for r in all_results:
            r["bert_score_f1"] = None

    if valid_results:
        preds_list  = [r["model_answer"] for r in valid_results]
        golds_list  = [r["gold_answer"]  for r in valid_results]
        corpus_bleu = sacrebleu.corpus_bleu(preds_list, [golds_list]).score
        corpus_chrf = sacrebleu.corpus_chrf(preds_list, [golds_list]).score
        for r in valid_results:
            r["bleu"] = round(sacrebleu.corpus_bleu([r["model_answer"]], [[r["gold_answer"]]]).score, 4)
            r["chrf"] = round(sacrebleu.corpus_chrf([r["model_answer"]], [[r["gold_answer"]]]).score, 4)
    else:
        corpus_bleu = 0.0
        corpus_chrf = 0.0

    for r in all_results:
        if "bleu" not in r:
            r["bleu"] = None
        if "chrf" not in r:
            r["chrf"] = None

    def smean(key, results=valid_results):
        vals = [r[key] for r in results if r.get(key) is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    summary = {
        "model":            model_key,
        "phase":            "phase2_no_reasoning",
        "lm_studio_model":  active,
        "total_queries":    len(all_results),
        "valid_answers":    len(valid_results),
        "error_count":      error_count,
        "error_rate_pct":   round(error_rate * 100, 2),
        "recall_at_5_pct":  round((smean("recall_at_5", all_results) or 0) * 100, 2),
        "mrr":              smean("mrr", all_results),
        "bert_score_f1":    smean("bert_score_f1") if bert_f1_list else None,
        "corpus_bleu":      round(corpus_bleu, 4),
        "corpus_chrf":      round(corpus_chrf, 4),
        "token_f1_pct":     round((smean("token_f1") or 0) * 100, 2),
        "exact_match_pct":  round((smean("exact_match") or 0) * 100, 2),
        "avg_latency_sec":  smean("latency_sec", all_results),
    }

    safe_key = model_key.replace("/", "_").replace(":", "_")
    out_path = os.path.join(RESULTS_DIR, f"result_v2_phase2_{safe_key}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "detailed_results": all_results}, f, indent=2)

    print(f"\n  [SAVED] {out_path}")
    print("\n" + "-" * 65)
    for label, key in [
        ("Recall@5",   "recall_at_5_pct"),
        ("MRR",        "mrr"),
        ("BERTScore",  "bert_score_f1"),
        ("Corpus BLEU","corpus_bleu"),
        ("Corpus chrF","corpus_chrf"),
        ("Token F1",   "token_f1_pct"),
        ("Error Rate", "error_rate_pct"),
        ("Avg Latency","avg_latency_sec"),
    ]:
        val = summary.get(key)
        unit = "%" if "pct" in key else ("s" if "latency" in key else "")
        print(f"  {label:<22} {val}{unit}")
    print("-" * 65)

    return summary


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CivicSense AI — Phase 2 Restricted Reasoning Evaluation")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    args = parser.parse_args()

    if not flask_ok():
        print("[ERROR] Flask app is not running. Start with:  python app.py")
        sys.exit(1)

    all_summaries = []
    for model in args.models:
        summary = evaluate_model(model)
        if summary:
            all_summaries.append(summary)
        time.sleep(2)

    if all_summaries:
        print("\n\n" + "=" * 90)
        print("  PHASE 2 FINAL COMPARISON TABLE (Restricted Reasoning)")
        print("=" * 90)
        print(f"{'Model':<42} | {'Errors':>6} | {'BERT':>6} | {'BLEU':>6} | {'chrF':>6} | {'TokF1':>6} | {'Lat(s)':>7}")
        print("-" * 90)
        for s in sorted(all_summaries, key=lambda x: x.get("bert_score_f1") or 0.0, reverse=True):
            label = s["model"][:42]
            bert  = f"{s['bert_score_f1']:.4f}" if s["bert_score_f1"] is not None else "  N/A "
            print(
                f"{label:<42} | {s['error_rate_pct']:>5.1f}% | {bert:>6} | "
                f"{s['corpus_bleu']:>6.2f} | {s['corpus_chrf']:>6.2f} | "
                f"{s['token_f1_pct']:>5.1f}% | {s['avg_latency_sec']:>7.1f}s"
            )
        print("=" * 90)
        print(f"\nAll results saved in: {RESULTS_DIR}/result_v2_phase2_*.json")
