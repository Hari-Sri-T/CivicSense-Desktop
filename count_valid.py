import json, glob

files = sorted(glob.glob('model_results/result_v2*.json'))
total_valid = 0
for fp in files:
    with open(fp, encoding='utf-8') as f:
        d = json.load(f)
    s = d['summary']
    valid = s.get('valid_answers', 0)
    phase = 'P2' if 'phase2' in fp else 'P1'
    model = s['model'][:35]
    print(f"{phase} | {model:<35} | valid={valid}")
    total_valid += valid
print(f"\nTotal valid answers to hallucination-check: {total_valid}")
print(f"Estimated time @ 10s/query (Qwen is fast): ~{total_valid*10//60} min")
