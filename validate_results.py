"""
Quick validation script — reads all v2 result files and prints
a per-model, per-query breakdown so we can spot-check for hallucinations.
"""
import json, os, glob

RESULTS_DIR = "model_results"

files = sorted(glob.glob(os.path.join(RESULTS_DIR, "result_v2_*.json")))

for fp in files:
    with open(fp, encoding="utf-8") as f:
        data = json.load(f)

    s   = data["summary"]
    res = data["detailed_results"]

    errors  = [r for r in res if r.get("is_error")]
    valids  = [r for r in res if not r.get("is_error")]

    print(f"\n{'='*70}")
    print(f"MODEL: {s['model']}")
    print(f"  total={s['total_queries']}  valid={s['valid_answers']}  errors={s['error_count']}  error_rate={s['error_rate_pct']}%")
    print(f"  BERTScore={s['bert_score_f1']}  BLEU={s['corpus_bleu']}  chrF={s['corpus_chrf']}  TokF1={s['token_f1_pct']}%")
    print(f"\n  --- SAMPLE VALID ANSWERS (first 3) ---")
    for r in valids[:3]:
        print(f"  Q: {r['query'][:80]}")
        print(f"  A: {r['model_answer'][:120]}")
        print(f"  G: {r['gold_answer'][:120]}")
        print(f"  token_f1={r.get('token_f1',0):.2f}  bert={r.get('bert_score_f1','?')}")
        print()

    if errors:
        print(f"  --- ERROR ANSWERS (first 2) ---")
        for r in errors[:2]:
            print(f"  Q: {r['query'][:80]}")
            print(f"  A: {r['model_answer'][:120]}")
            print()
