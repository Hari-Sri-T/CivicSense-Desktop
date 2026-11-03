import os
import pandas as pd
import ollama  # Make sure you have the 'ollama' Python library: pip install ollama
import google.generativeai as genai
from transformers import pipeline
import json
import time

# --- 1. CONFIGURE YOUR MODELS ---

# a) Configure Gemini API Key
# (Replace with your actual API key)
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# b) Define the LLM prompt from your notes [cite: 183-185]
# [cite_start]b) Define the LLM prompt from your notes [cite: 183-185]
LLM_PROMPT_TEMPLATE = """You are an ESG/SDG analyst. Classify sustainability sentiment as one of: Positive, Neutral, Negative.
Return ONLY JSON: {{"label": ""}}
News: "{text}"
"""

# c) Define the label map for your fine-tuned sdg_bert model
# Based on your notebook's label_fn: 2=Positive, 1=Neutral, 0=Negative [cite: 41, 42, 43]
# Note: Your notebook had a typo 'return 8' for negative, I'm assuming it should be 0.
BERT_LABEL_MAP = {
    "LABEL_2": "Positive",
    "LABEL_1": "Neutral",
    "LABEL_0": "Negative"
}

# List of Ollama models to test
OLLAMA_MODELS = [
    "llama3.2",  # <-- Change this line
    "mistral",
    "phi3",
    "gemma"
]

# --- 2. DEFINE PREDICTION FUNCTIONS ---

def get_sdg_bert_prediction(text, classifier):
    """
    Gets a prediction from your locally saved fine-tuned model.
    """
    try:
        result = classifier(text)[0]
        # Map label (e.g., "LABEL_2") to the text ("Positive")
        return BERT_LABEL_MAP.get(result['label'], "Error")
    except Exception as e:
        print(f"Error (sdg_bert): {e}")
        return "Error"

def get_ollama_prediction(text, model_name):
    """
    Gets a prediction from a local Ollama model.
    """
    prompt = LLM_PROMPT_TEMPLATE.format(text=text)
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
            format='json' # Request JSON output
        )
        
        # The 'ollama' library with format='json' should parse the JSON for you
        content = response['message']['content']
        
        # In case it's still a string, try to parse it
        if isinstance(content, str):
            json_data = json.loads(content)
        else:
            json_data = content
            
        return json_data.get("label", "Parse Error")
    except Exception as e:
        print(f"Error (Ollama {model_name}): {e}")
        return "Error"

def get_gemini_prediction(text, gemini_model):
    """
    Gets a prediction from the Gemini Pro API.
    """
    prompt = LLM_PROMPT_TEMPLATE.format(text=text)
    try:
        response = gemini_model.generate_content(prompt)
        # Clean the response to extract only the JSON
        clean_response = response.text.strip().replace("```json", "").replace("```", "")
        json_data = json.loads(clean_response)
        return json_data.get("label", "Parse Error")
    except Exception as e:
        print(f"Error (Gemini): {e}")
        return "Error"

# --- 3. MAIN EXECUTION ---

print("Starting evaluation...")

# Load your fine-tuned model from the "./sdg_bert" directory [cite: 100]
print("Loading fine-tuned sdg_bert model...")
try:
    sdg_bert_classifier = pipeline("text-classification", model="./sdg_bert")
except Exception as e:
    print(f"Failed to load './sdg_bert'. Make sure the directory is correct. Error: {e}")
    exit()


# Load the Gemini model
try:
    gemini_pro_model = genai.GenerativeModel('gemini-flash-lite-latest')
except Exception as e:
    print(f"Failed to configure Gemini. Did you set your API key? Error: {e}")
    # You can choose to 'exit()' or just continue without Gemini
    gemini_pro_model = None


# Load the CSV file created by your notebook [cite: 163, 176]
df = pd.read_csv("llm_eval_samples.csv")
print(f"Loaded {len(df)} samples from llm_eval_samples.csv")

# Create new columns for predictions 
df["pred_sdg_bert"] = ""
for model_name in OLLAMA_MODELS:
    df[f"pred_{model_name}"] = ""
if gemini_pro_model:
    df["pred_gemini_pro"] = ""

# --- 4. RUN PREDICTIONS ---

print("Running predictions... This may take a while.")

for index, row in df.iterrows():
    text = row["text"]
    print(f"Processing sample {index+1}/{len(df)}...")

    # a) Baseline: Your fine-tuned model [cite: 180]
    df.at[index, "pred_sdg_bert"] = get_sdg_bert_prediction(text, sdg_bert_classifier)
    
    # b) Ollama Models [cite: 181, 182]
    for model_name in OLLAMA_MODELS:
        df.at[index, f"pred_{model_name}"] = get_ollama_prediction(text, model_name)
        time.sleep(1) # Simple rate limiting
        
    # c) Gemini Pro [cite: 182]
    if gemini_pro_model:
        df.at[index, "pred_gemini_pro"] = get_gemini_prediction(text, gemini_pro_model)
        time.sleep(2) # Avoid hitting API rate limits

# --- 5. SAVE RESULTS ---

output_file = "llm_eval_results.csv"
df.to_csv(output_file, index=False)

print(f"\n✅ Evaluation complete!")
print(f"Results saved to {output_file}")
print(df.head())