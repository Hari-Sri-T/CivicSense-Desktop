"""
STAGE 3 : LLM AS JUDGE (LM Studio)

Gold Answer Based Evaluation
(No corpus needed)

Input:
evaluation_results_full.json

Output:
evaluation_results_judged.json
evaluation_summary_final.txt

Requires: LM Studio running on localhost:1234 with JUDGE_MODEL loaded.
"""

import json
import time
import os
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

INPUT_FILE  = "evaluation_results_full.json"
OUTPUT_FILE = "evaluation_results_judged.json"
SUMMARY_FILE = "evaluation_summary_final.txt"

# LM Studio config — reads from .env (JUDGE_MODEL=qwen2.5-14b-instruct)
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
JUDGE_MODEL        = os.getenv("JUDGE_MODEL", "qwen2.5-14b-instruct")

client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key="lm-studio")
print(f"Judge model: {JUDGE_MODEL} via {LM_STUDIO_BASE_URL}")

SLEEP_TIME = 0.1  # LM Studio is local, no rate limits needed


def build_prompt(question, gold, model_answer):

    return f"""
You are a STRICT LEGAL evaluator.

Evaluate correctness of MODEL ANSWER.

QUESTION:
{question}

EXPECTED ANSWER:
{gold}

MODEL ANSWER:
{model_answer}

Scoring:

1.0 = Correct and consistent with expected answer.

0.5 = Partially correct.

0.0 = Incorrect or hallucinated.

Return JSON ONLY:

{{
"faithfulness_score": number,
"reason": "short explanation"
}}

Do not add extra text.
"""


def judge(prompt):
    """Calls LM Studio judge model and returns (score, reason)."""
    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=256,
        )
        output = response.choices[0].message.content
    except Exception as e:
        return 0.0, f"lm_studio_error:{str(e)}"

    try:
        js = json.loads(output)
        score  = float(js.get("faithfulness_score", 0.0))
        reason = js.get("reason", "")
        return score, reason
    except Exception:
        return 0.0, "parse_failure"



print("\n--- STAGE 3 : LLM AS JUDGE ---")

with open(INPUT_FILE,"r",encoding="utf-8") as f:

    results = json.load(f)


scores = []

hallucinations = 0


for r in tqdm(results,desc="Judging"):

    prompt = build_prompt(

        r.get("question",""),

        r.get("gold_answer",""),

        r.get("model_answer","")

    )

    score,reason = judge(prompt)

    r["faithfulness_score"] = score

    r["hallucination_flag"] = 1 if score == 0 else 0

    r["judge_reason"] = reason

    scores.append(score)

    if score == 0:

        hallucinations += 1

    time.sleep(SLEEP_TIME)


avg = float(np.mean(scores))

halluc_rate = hallucinations / len(scores)


summary = f"""
--- LLM AS JUDGE SUMMARY ---

Average Faithfulness Score:
{avg:.4f}

Hallucination Rate:
{halluc_rate * 100:.2f} %

Total Questions:
{len(scores)}
"""


with open(OUTPUT_FILE,"w",encoding="utf-8") as f:

    json.dump(results,f,indent=2,ensure_ascii=False)


with open(SUMMARY_FILE,"w") as f:

    f.write(summary)


print(summary)

print("\nStage 3 Complete.")