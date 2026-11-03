import json, os

models = [
    'result_google_gemma-3-4b.json',
    'result_llama-3.2-3b-instruct.json',
    'result_mistralai_ministral-3-3b.json',
    'result_nvidia_nemotron-3-nano-4b.json',
    'result_qwen2.5-3b-instruct.json',
]

for m in models:
    path = os.path.join('model_results', m)
    with open(path) as f:
        d = json.load(f)
    s = d['summary']
    results = d['detailed_results']
    errors = sum(1 for r in results if 'Error' in str(r.get('model_answer', '')))
    missing_bert = sum(1 for r in results if r.get('bert_score_f1') is None)
    print(f"{m[:40]}")
    print(f"  queries={len(results)} errors={errors} missing_bert={missing_bert}")
    print(f"  bert={s.get('bert_score_f1')} bleu={s.get('bleu')} chrf={s.get('chrf')} token_f1={s.get('token_f1_pct')} latency={s.get('avg_latency_sec')}")
    print()
