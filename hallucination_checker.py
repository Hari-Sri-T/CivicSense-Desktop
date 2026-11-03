"""
CIVICSENSE AI — LLM-AS-JUDGE HALLUCINATION CHECKER
====================================================
Uses Qwen2.5-3b-instruct as a faithfulness judge to evaluate
every valid model answer across Phase 1 and Phase 2 results.

For each valid answer it scores:
  - is_faithful   (bool)  : does the answer stay within what the gold answer says?
  - hallucination_score (0.0=faithful, 1.0=hallucination)
  - judge_reasoning (str) : brief explanation

Results are appended in-place to each result_v2*.json file.
A hallucination summary is appended to the summary dict.

Usage:
  python hallucination_checker.py
"""

import os, sys, json, glob, time, subprocess
import requests

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    from json_repair import repair_json
    JSON_REPAIR = True
except ImportError:
    JSON_REPAIR = False

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
JUDGE_MODEL       = "qwen2.5-3b-instruct"
LM_STUDIO_URL     = "http://localhost:1234/v1"
LM_STUDIO_MODELS  = "http://localhost:1234/v1/models"
RESULTS_DIR       = "model_results"
CONTEXT_LENGTH    = 4096       # Judge only needs modest context
MODEL_LOAD_POLL   = 3

JUDGE_SYSTEM_PROMPT = """You are a strict hallucination evaluator for a legal document RAG system.

Your job is to determine whether a MODEL ANSWER is faithful to the GOLD ANSWER extracted from the actual legal document.

FAITHFUL means: the model answer makes only claims that are supported by or consistent with the gold answer.
HALLUCINATION means: the model answer adds facts, numbers, names, or claims that are NOT in the gold answer, OR contradicts the gold answer.

Important rules:
- Minor paraphrasing or summarising is NOT hallucination.
- Saying "Yes" or "No" when the gold answer is a full sentence explanation may be incomplete but is not necessarily a hallucination — evaluate the factual content only.
- Adding legal analysis, opinions, or conclusions not present in the gold answer IS hallucination.
- If the model answer says it cannot answer or the information is not in the sources, that is faithful (not hallucination).

Respond in valid JSON only with exactly three keys:
{
  "is_faithful": true or false,
  "hallucination_score": a float from 0.0 (completely faithful) to 1.0 (complete hallucination),
  "judge_reasoning": "one sentence explanation"
}"""

def judge_prompt(query: str, gold: str, model_answer: str) -> str:
    return f"""QUERY:
{query}

GOLD ANSWER (from the actual legal document):
{gold}

MODEL ANSWER (to evaluate):
{model_answer}

Is the model answer faithful? Respond in JSON only."""

# ─────────────────────────────────────────────
# LMS HELPERS
# ─────────────────────────────────────────────

def lms_unload_all():
    print("  [lms] Unloading all models...")
    subprocess.run(["lms", "unload", "--all"], capture_output=True, text=True)
    time.sleep(2)


def lms_load_judge():
    print(f"  [lms] Loading judge model: {JUDGE_MODEL}...")
    result = subprocess.run(
        ["lms", "load", JUDGE_MODEL, "--context-length", str(CONTEXT_LENGTH), "--gpu", "max", "-y"],
        capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        print(f"  [lms] load error: {result.stderr.strip()}")
        return False
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            r = requests.get(LM_STUDIO_MODELS, timeout=5)
            models = r.json().get("data", [])
            if models:
                print(f"  [lms] Judge ready: {models[0].get('id', '')}")
                time.sleep(2)
                return True
        except Exception:
            pass
        time.sleep(MODEL_LOAD_POLL)
    return False


def call_judge(query: str, gold: str, model_answer: str) -> dict:
    """Call Qwen judge once with one automatic retry."""
    prompt = judge_prompt(query, gold, model_answer)
    for attempt in range(2):
        try:
            r = requests.post(
                f"{LM_STUDIO_URL}/chat/completions",
                json={
                    "model": "local-model",
                    "messages": [
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt}
                    ],
                    "temperature": 0.0,
                    "max_tokens": 256,
                },
                timeout=60
            )
            raw = r.json()["choices"][0]["message"]["content"].strip()

            # Parse JSON with tier fallback
            start = raw.find("{"); end = raw.rfind("}") + 1
            if start != -1 and end > start:
                raw = raw[start:end]
            try:
                result = json.loads(raw, strict=False)
                return result
            except json.JSONDecodeError:
                if JSON_REPAIR:
                    repaired = repair_json(raw, return_objects=True)
                    if isinstance(repaired, dict):
                        return repaired

        except Exception as e:
            if attempt == 0:
                print(f"      [judge] retry: {e}")
                time.sleep(3)
            else:
                print(f"      [judge] failed: {e}")

    return {
        "is_faithful": None,
        "hallucination_score": None,
        "judge_reasoning": "judge call failed"
    }

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def process_file(fp: str):
    with open(fp, encoding="utf-8") as f:
        data = json.load(f)

    results = data["detailed_results"]
    summary = data["summary"]
    model   = summary.get("model", "unknown")
    phase   = "Phase 2" if "phase2" in fp else "Phase 1"

    print(f"\n{'='*65}")
    print(f"  Judging: {model} ({phase})")
    print(f"  File:    {os.path.basename(fp)}")

    valid  = [r for r in results if not r.get("is_error")]
    errors = [r for r in results if r.get("is_error")]

    print(f"  Valid answers to judge: {len(valid)}")

    scores = []
    for i, r in enumerate(valid):
        verdict = call_judge(r["query"], r["gold_answer"], r["model_answer"])
        r["is_faithful"]        = verdict.get("is_faithful")
        r["hallucination_score"]= verdict.get("hallucination_score")
        r["judge_reasoning"]    = verdict.get("judge_reasoning", "")
        if verdict.get("hallucination_score") is not None:
            scores.append(verdict["hallucination_score"])

        status = "faithful" if verdict.get("is_faithful") else "HALLUCINATION"
        score  = f"{verdict.get('hallucination_score', '?'):.2f}" if verdict.get('hallucination_score') is not None else "?"
        print(f"  [{i+1:>3}/{len(valid)}] score={score} | {status} | {r['query'][:55]}")
        time.sleep(0.2)

    # Mark error answers as not evaluated
    for r in errors:
        r["is_faithful"]         = None
        r["hallucination_score"] = None
        r["judge_reasoning"]     = "skipped: model returned error"

    # Summarise
    if scores:
        faithful_count   = sum(1 for r in valid if r.get("is_faithful") is True)
        hallucin_count   = sum(1 for r in valid if r.get("is_faithful") is False)
        avg_halluc_score = round(sum(scores) / len(scores), 4)
        faithfulness_pct = round(faithful_count / len(valid) * 100, 2)
    else:
        faithful_count = hallucin_count = 0
        avg_halluc_score = None
        faithfulness_pct = None

    summary["judge_model"]         = JUDGE_MODEL
    summary["faithfulness_pct"]    = faithfulness_pct
    summary["avg_hallucination_score"] = avg_halluc_score
    summary["faithful_count"]      = faithful_count
    summary["hallucination_count"] = hallucin_count

    data["summary"]         = summary
    data["detailed_results"]= results

    with open(fp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"\n  Faithfulness: {faithfulness_pct}%  |  Avg Hallucination Score: {avg_halluc_score}")
    print(f"  Faithful={faithful_count}  Hallucinations={hallucin_count}  [SAVED]")
    return {
        "model":  model,
        "phase":  phase,
        "faithfulness_pct":  faithfulness_pct,
        "avg_hallucination_score": avg_halluc_score,
        "bert_score_f1": summary.get("bert_score_f1"),
        "corpus_bleu":   summary.get("corpus_bleu"),
    }


if __name__ == "__main__":
    lms_unload_all()
    if not lms_load_judge():
        print("[ERROR] Could not load judge model.")
        sys.exit(1)

    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "result_v2*.json")))
    if not files:
        print("[ERROR] No result_v2*.json files found.")
        sys.exit(1)

    print(f"\nFound {len(files)} result files to evaluate.")
    all_summaries = []
    for fp in files:
        s = process_file(fp)
        if s:
            all_summaries.append(s)

    lms_unload_all()

    print("\n\n" + "=" * 90)
    print("  HALLUCINATION JUDGE RESULTS — SUMMARY")
    print("=" * 90)
    print(f"{'Model':<38} | {'Phase':<7} | {'Faithful%':>9} | {'Halluc Score':>12} | {'BERTScore':>9} | {'BLEU':>6}")
    print("-" * 90)
    for s in sorted(all_summaries, key=lambda x: x.get("faithfulness_pct") or 0, reverse=True):
        fai = f"{s['faithfulness_pct']:.1f}%" if s["faithfulness_pct"] is not None else "N/A"
        hsc = f"{s['avg_hallucination_score']:.4f}" if s["avg_hallucination_score"] is not None else "N/A"
        bert = f"{s['bert_score_f1']:.4f}" if s.get("bert_score_f1") else "N/A"
        bleu = f"{s['corpus_bleu']:.2f}" if s.get("corpus_bleu") is not None else "N/A"
        print(f"{s['model']:<38} | {s['phase']:<7} | {fai:>9} | {hsc:>12} | {bert:>9} | {bleu:>6}")
    print("=" * 90)
    print("\nAll result files updated in-place with hallucination scores.")
    print("Run  python compile_v2.py  to regenerate the final comparison CSV.")
