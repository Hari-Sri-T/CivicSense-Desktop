import json
import sacrebleu
import bert_score
from collections import defaultdict

backup_path = "f:/CivicSense/CivicSense/model_results/backup_qwen2.5-3b-instruct.json"
with open(backup_path, "r", encoding="utf-8") as f:
    data = json.load(f)

results = data["detailed_results"]
metrics = defaultdict(list)

all_preds = []
all_golds = []

for res in results:
    pred_val = str(res.get("model_answer", "")).strip()
    if not pred_val: pred_val = "Empty"
    gold_val = str(res.get("gold_answer", "")).strip()
    if not gold_val: gold_val = "Empty"
    
    all_preds.append(pred_val)
    all_golds.append(gold_val)

print("Calculating BERTScore for backup...")
_, _, bert_f1 = bert_score.score(all_preds, all_golds, lang="en", verbose=False)
bert_scores = bert_f1.tolist()

for i, res in enumerate(results):
    res["bert_score_f1"] = bert_scores[i]
    pred_val = all_preds[i]
    gold_val = all_golds[i]
    
    res["bleu"] = round(sacrebleu.corpus_bleu([pred_val], [[gold_val]]).score, 4)
    res["chrf"] = round(sacrebleu.corpus_chrf([pred_val], [[gold_val]]).score, 4)
    
    for k in ["recall_at_5", "mrr", "token_f1", "exact_match", "latency_sec", "bleu", "chrf", "bert_score_f1"]:
        metrics[k].append(res.get(k, 0))

print("Total Queries:", len(results))
for k, v in metrics.items():
    print(f"{k}: {sum(v)/len(v):.4f}")

# Save the final resolved file for Qwen!
final_path = "f:/CivicSense/CivicSense/model_results/result_qwen2.5-3b-instruct.json"
with open(final_path, "w", encoding="utf-8") as f:
    summary_data = {
        "model": "qwen2.5-3b-instruct",
        "total_queries": len(results),
        "bert_score_f1": sum(metrics["bert_score_f1"])/len(metrics["bert_score_f1"]),
        "bleu": sum(metrics["bleu"])/len(metrics["bleu"]),
        "chrf": sum(metrics["chrf"])/len(metrics["chrf"]),
        "token_f1_pct": (sum(metrics["token_f1"])/len(metrics["token_f1"])) * 100,
        "avg_latency_sec": sum(metrics["latency_sec"])/len(metrics["latency_sec"]),
    }
    json.dump({"summary": summary_data, "detailed_results": results}, f, indent=2)
print(f"Final Qwen results saved to {final_path}!")

