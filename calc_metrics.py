import json
import sacrebleu
import numpy as np
import collections
import sys

def main():
    try:
        with open("model_results/result_v3_qwen_unrestricted.json") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    benchmarks = ['cuad.json', 'privacy_qa.json', 'contractnli.json', 'maud.json']
    results_by_bench = collections.defaultdict(list)

    for r in data.get("detailed_results", []):
        bench = r.get("benchmark")
        if bench:
            results_by_bench[bench].append(r)

    print(f"{'Metric':<15} | {'CUAD':<10} | {'Privacy-QA':<10} | {'ContractNLI':<10} | {'MAUD':<10}")
    print("-" * 65)

    metrics_output = {
        'Corpus BLEU': {},
        'Corpus chrF': {},
        'Token F1': {},
        'Error Rate': {},
        'Avg Latency (s)': {}
    }

    for bench in benchmarks:
        results = results_by_bench[bench]
        valid = [r for r in results if not r.get("is_error")]
        errors = [r for r in results if r.get("is_error")]
        
        error_rate = (len(errors) / len(results)) * 100 if results else 0
        
        avg_latency = np.mean([r.get("latency_sec", 0) for r in results]) if results else 0
        token_f1 = np.mean([r.get("token_f1", 0) for r in valid]) * 100 if valid else 0
        
        refs = [[r.get("gold_answer", "") for r in valid]]
        sys_ans = [r.get("model_answer", "") for r in valid]
        
        if valid:
            corpus_bleu = sacrebleu.corpus_bleu(sys_ans, refs).score
            corpus_chrf = sacrebleu.corpus_chrf(sys_ans, refs).score
        else:
            corpus_bleu = 0
            corpus_chrf = 0
            
        b_name = bench
        metrics_output['Corpus BLEU'][b_name] = f"{corpus_bleu:.2f}"
        metrics_output['Corpus chrF'][b_name] = f"{corpus_chrf:.2f}"
        metrics_output['Token F1'][b_name] = f"{token_f1:.1f}%"
        metrics_output['Error Rate'][b_name] = f"{error_rate:.1f}%"
        metrics_output['Avg Latency (s)'][b_name] = f"{avg_latency:.1f}"

    for metric, vals in metrics_output.items():
        print(f"{metric:<15} | {vals.get('cuad.json', '-'):<10} | {vals.get('privacy_qa.json', '-'):<10} | {vals.get('contractnli.json', '-'):<10} | {vals.get('maud.json', '-'):<10}")

if __name__ == '__main__':
    main()
