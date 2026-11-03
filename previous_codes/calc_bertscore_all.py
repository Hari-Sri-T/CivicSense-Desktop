import json
import sacrebleu
import bert_score
import glob

files = [
    "f:/CivicSense/CivicSense/model_results/backup_qwen2.5-3b-instruct.json"
]

for file_path in files:
    print(f"\nProcessing {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["detailed_results"]
    all_preds = []
    all_golds = []

    for res in results:
        pred_val = str(res.get("model_answer", "")).strip()
        if not pred_val: pred_val = "Empty"
        gold_val = str(res.get("gold_answer", "")).strip()
        if not gold_val: gold_val = "Empty"
        
        all_preds.append(pred_val)
        all_golds.append(gold_val)

    print(f"Calculating BERTScore for {len(all_preds)} predictions in {file_path}...")
    try:
        _, _, bert_f1 = bert_score.score(all_preds, all_golds, lang="en", verbose=False)
        bert_scores = bert_f1.tolist()
    except Exception as e:
        print(f"Failed to calculate BERTScore: {e}")
        bert_scores = [None] * len(all_preds)

    sum_bert = 0
    valid_bert = 0
    for i, res in enumerate(results):
        score = bert_scores[i]
        res["bert_score_f1"] = score
        if score is not None:
            sum_bert += score
            valid_bert += 1

    # Only update the final path if it's the backup file being resolved, else update in-place
    final_path = file_path.replace("backup_", "result_")
    
    with open(final_path, "w", encoding="utf-8") as f:
        if "summary" in data and isinstance(data["summary"], dict):
            summary_data = data["summary"]
            if valid_bert > 0:
                summary_data["bert_score_f1"] = sum_bert / valid_bert
        else:
            # Recompute summary data if it was a backup without a full summary block
            model_name = "qwen2.5-3b-instruct" if "qwen" in file_path else "mistralai/ministral-3-3b"
            summary_data = {
                "model": model_name,
                "total_queries": len(results),
                "bert_score_f1": (sum_bert / valid_bert) if valid_bert > 0 else None,
                "bleu": sum(res.get("bleu", 0) for res in results)/len(results),
                "chrf": sum(res.get("chrf", 0) for res in results)/len(results),
                "token_f1_pct": (sum(res.get("token_f1", 0) for res in results)/len(results)) * 100,
                "avg_latency_sec": sum(res.get("latency_sec", 0) for res in results)/len(results),
            }
            
        json.dump({"summary": summary_data, "detailed_results": results}, f, indent=2)
    print(f"Successfully saved {final_path} with BERTScore!")
