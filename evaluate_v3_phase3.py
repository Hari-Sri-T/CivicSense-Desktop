"""
CIVICSENSE AI — PHASE 3: FULL REPRESENTATIVE DATASET RUN
=========================================================
Runs the two best-performing model/config combos from Phase 1+2
against a large representative sample (150 queries × 4 benchmarks = 600 total).

Models:
  1. qwen2.5-3b-instruct     → Phase 1 config (unrestricted reasoning)
  2. google/gemma-3-4b        → Phase 2 config (no_reasoning=True)

Results saved as:  model_results/result_v3_<model>_<config>.json
"""

import os, sys, json, time, subprocess, glob, warnings
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

try:
    from json_repair import repair_json
    JSON_REPAIR_AVAILABLE = True
except ImportError:
    JSON_REPAIR_AVAILABLE = False

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
FLASK_BASE_URL    = "http://localhost:5001"
LM_STUDIO_URL     = "http://localhost:1234/v1/models"
CORPUS_DIR        = "LegalBench-RAG/corpus"
BENCHMARK_DIR     = "LegalBench-RAG/benchmarks"
RESULTS_DIR       = "model_results"
QUERIES_PER_BENCH = 125         # 125 × 4 benchmarks = 500 total
CONTEXT_LENGTH    = 8192
API_RETRY_WAIT    = 4
MODEL_LOAD_POLL   = 3

# The two best models from Phase 1+2
PHASE3_RUNS = [
    {"model": "qwen2.5-3b-instruct", "no_reasoning": False, "label": "qwen_unrestricted"},
    {"model": "google/gemma-3-4b",   "no_reasoning": True,  "label": "gemma_restricted"}
]

os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# LM STUDIO HELPERS
# ─────────────────────────────────────────────

def lms_unload_all():
    print("  [lms] Unloading all models...")
    result = subprocess.run(["lms", "unload", "--all"], capture_output=True, text=True)
    if result.returncode != 0 and "No models" not in (result.stdout + result.stderr):
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
    deadline = time.time() + 150
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


def clear_index(session):
    try:
        session.post(f"{FLASK_BASE_URL}/clear_index", timeout=10)
    except Exception:
        pass


def load_file(session, full_path: str) -> bool:
    try:
        resp = session.post(f"{FLASK_BASE_URL}/load_file",
                            json={"file_path": full_path}, timeout=60)
        resp.raise_for_status()
        return resp.json().get("status") == "success"
    except Exception as e:
        print(f"    [load_file] {e}")
        return False


def query_rag(session, query: str, no_reasoning: bool, top_k: int = 5):
    payload = {"query": query, "top_k": top_k, "no_reasoning": no_reasoning}
    for attempt in range(2):
        try:
            resp = session.post(f"{FLASK_BASE_URL}/query", json=payload, timeout=180)
            if resp.status_code == 200:
                return resp.json()
            print(f"    [query] HTTP {resp.status_code} — {'retrying' if attempt == 0 else 'giving up'}")
        except Exception as e:
            print(f"    [query] Exception: {e} — {'retrying' if attempt == 0 else 'giving up'}")
        if attempt == 0:
            time.sleep(API_RETRY_WAIT)
    return None

# ─────────────────────────────────────────────
# METRIC HELPERS
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


def recall_at_k(sources, gold_fp, k=5) -> float:
    norm = gold_fp.replace("\\", "/")
    for src in (sources or [])[:k]:
        if norm in src.replace("\\", "/"):
            return 1.0
    return 0.0


def mrr_score(sources, gold_fp) -> float:
    norm = gold_fp.replace("\\", "/")
    for i, src in enumerate(sources or []):
        if norm in src.replace("\\", "/"):
            return 1.0 / (i + 1)
    return 0.0

# ─────────────────────────────────────────────
# MAIN EVALUATION LOOP FOR ONE RUN
# ─────────────────────────────────────────────

def evaluate_run(run: dict):
    model_key    = run["model"]
    no_reasoning = run["no_reasoning"]
    label        = run["label"]
    config_tag   = "restricted" if no_reasoning else "unrestricted"

    print("\n" + "=" * 70)
    print(f"  [PHASE 3] {model_key}  [{config_tag}]")
    print(f"  Queries per benchmark: {QUERIES_PER_BENCH}  |  Total: {QUERIES_PER_BENCH*4}")
    print("=" * 70)

    if not flask_ok():
        print("[ERROR] Flask not running. Start with:  python app.py")
        return None

    lms_unload_all()
    if not lms_load_model(model_key):
        print(f"[ERROR] Could not load {model_key}. Skipping.")
        return None

    active = get_active_model()
    print(f"  LM Studio active: {active}\n")

    bench_paths = sorted(glob.glob(os.path.join(BENCHMARK_DIR, "*.json")))
    session     = requests.Session()
    all_results = []
    bench_num   = 0

    for bench_path in tqdm(bench_paths, desc="Benchmarks"):
        bench_name = os.path.basename(bench_path)
        bench_num += 1
        with open(bench_path, "r", encoding="utf-8") as f:
            bench_data = json.load(f)

        tests = bench_data.get("tests", [])
        if not tests:
            continue

        # Group by file, maintain order
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
                data    = query_rag(session, q, no_reasoning)
                elapsed = round(time.time() - t0, 4)

                if data is None:
                    model_answer = "Error: API call failed after retry."
                    sources = []
                else:
                    raw          = data.get("answer", "")
                    model_answer = str(raw).strip() if raw else "Empty"
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
                    "mrr":          mrr_score(sources, file_path),
                    "token_f1":     token_f1(model_answer, gt) if not is_error(model_answer) else 0.0,
                    "exact_match":  1.0 if model_answer.strip().lower() == gt.strip().lower() else 0.0,
                })
                query_count += 1
                time.sleep(0.2)

            # Progress checkpoint save after every file
            safe = label.replace("/", "_")
            with open(f"{RESULTS_DIR}/backup_v3_{safe}.json", "w", encoding="utf-8") as f:
                json.dump({"summary": "IN PROGRESS", "detailed_results": all_results}, f, indent=2)

        done = sum(1 for r in all_results if r["benchmark"] == bench_name)
        errors = sum(1 for r in all_results if r["benchmark"] == bench_name and r["is_error"])
        print(f"  [{bench_num}/4] {bench_name}: {done} queries done, {errors} errors")

    # ── Unload model, compute metrics ──────────────────────────────
    print(f"\n  [lms] Unloading model to free RAM for BERTScore...")
    lms_unload_all()
    time.sleep(3)

    valid  = [r for r in all_results if not r["is_error"]]
    errors_count = len(all_results) - len(valid)
    error_rate   = errors_count / max(len(all_results), 1)

    print(f"\n  Total: {len(all_results)} | Valid: {len(valid)} | Errors: {errors_count} ({error_rate*100:.1f}%)")

    # BERTScore
    bert_f1_list = []
    if BERT_SCORE_AVAILABLE and valid:
        preds = [r["model_answer"] for r in valid]
        golds = [r["gold_answer"]  for r in valid]
        print(f"  Computing BERTScore ({len(preds)} predictions)...")
        try:
            _, _, bert_f1 = bert_score.score(preds, golds, lang="en", verbose=False)
            bert_f1_list  = bert_f1.tolist()
            for i, r in enumerate(valid):
                r["bert_score_f1"] = round(bert_f1_list[i], 4)
            print(f"  BERTScore done. Mean F1 = {np.mean(bert_f1_list):.4f}")
        except Exception as e:
            print(f"  BERTScore FAILED: {e}")

    if valid:
        preds_list  = [r["model_answer"] for r in valid]
        golds_list  = [r["gold_answer"]  for r in valid]
        corpus_bleu = sacrebleu.corpus_bleu(preds_list, [golds_list]).score
        corpus_chrf = sacrebleu.corpus_chrf(preds_list, [golds_list]).score
        for r in valid:
            r["bleu"] = round(sacrebleu.corpus_bleu([r["model_answer"]], [[r["gold_answer"]]]).score, 4)
            r["chrf"] = round(sacrebleu.corpus_chrf([r["model_answer"]], [[r["gold_answer"]]]).score, 4)
    else:
        corpus_bleu = corpus_chrf = 0.0

    for r in all_results:
        for k in ["bleu", "chrf", "bert_score_f1"]:
            if k not in r:
                r[k] = None

    def smean(key, res=valid):
        vals = [r[key] for r in res if r.get(key) is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    summary = {
        "model":            model_key,
        "config":           config_tag,
        "label":            label,
        "phase":            "phase3",
        "lm_studio_model":  active,
        "total_queries":    len(all_results),
        "valid_answers":    len(valid),
        "error_count":      errors_count,
        "error_rate_pct":   round(error_rate * 100, 2),
        "recall_at_5_pct":  round((smean("recall_at_5", all_results) or 0) * 100, 2),
        "mrr":              smean("mrr", all_results),
        "bert_score_f1":    smean("bert_score_f1") if bert_f1_list else None,
        "corpus_bleu":      round(corpus_bleu, 4),
        "corpus_chrf":      round(corpus_chrf, 4),
        "token_f1_pct":     round((smean("token_f1") or 0) * 100, 2),
        "exact_match_pct":  round((smean("exact_match") or 0) * 100, 2),
        "avg_latency_sec":  smean("latency_sec", all_results),
        "queries_per_bench": QUERIES_PER_BENCH,
    }

    safe = label.replace("/", "_").replace(":", "_")
    out  = os.path.join(RESULTS_DIR, f"result_v3_{safe}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "detailed_results": all_results}, f, indent=2)

    print(f"\n  [SAVED] {out}")
    print("\n" + "-" * 70)
    for k, v in [
        ("Recall@5",     f"{summary['recall_at_5_pct']}%"),
        ("MRR",          summary['mrr']),
        ("BERTScore F1", summary['bert_score_f1']),
        ("Corpus BLEU",  summary['corpus_bleu']),
        ("Corpus chrF",  summary['corpus_chrf']),
        ("Token F1",     f"{summary['token_f1_pct']}%"),
        ("Error Rate",   f"{summary['error_rate_pct']}%"),
        ("Avg Latency",  f"{summary['avg_latency_sec']}s"),
    ]:
        print(f"  {k:<22} {v}")
    print("-" * 70)

    return summary


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    if not flask_ok():
        print("[ERROR] Flask is not running. Start with:  python app.py")
        sys.exit(1)

    start_time = time.time()
    summaries  = []

    for run in PHASE3_RUNS:
        s = evaluate_run(run)
        if s:
            summaries.append(s)
        time.sleep(3)

    elapsed_h = (time.time() - start_time) / 3600

    if summaries:
        print("\n\n" + "=" * 80)
        print(f"  PHASE 3 FINAL TABLE  ({QUERIES_PER_BENCH} queries/benchmark = {QUERIES_PER_BENCH*4} total each)")
        print("=" * 80)
        print(f"{'Model + Config':<45} | {'Err%':>5} | {'BERT':>6} | {'BLEU':>6} | {'chrF':>6} | {'TokF1':>6} | {'Lat':>6}")
        print("-" * 80)
        for s in sorted(summaries, key=lambda x: x.get("bert_score_f1") or 0.0, reverse=True):
            label = f"{s['model']} [{s['config']}]"[:45]
            bert  = f"{s['bert_score_f1']:.4f}" if s["bert_score_f1"] else "  N/A"
            print(
                f"{label:<45} | {s['error_rate_pct']:>4.1f}% | {bert:>6} | "
                f"{s['corpus_bleu']:>6.2f} | {s['corpus_chrf']:>6.2f} | "
                f"{s['token_f1_pct']:>5.1f}% | {s['avg_latency_sec']:>5.1f}s"
            )
        print("=" * 80)
        print(f"\nTotal wall time: {elapsed_h:.2f} hours")
        print(f"Results saved in: {RESULTS_DIR}/result_v3_*.json")
