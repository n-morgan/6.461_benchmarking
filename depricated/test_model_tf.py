import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

import torch
import os
import gc

# ---- CONFIG ----

# Load environment variables
load_dotenv()
torch.cuda.empty_cache()
gc.collect()



HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
print("Token loaded:", HF_TOKEN is not None)

MODEL_NAME ="meta-llama/Llama-3.1-8B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_FILE = "model_responses.json"
INCREMENTAL_FILE = "model_responses_incremental.jsonl"

# ---- LOAD MODEL ----
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,token=HF_TOKEN, device_map="auto")
model.eval()

# ---- LOAD BENCHMARK ----
with open("benchmark_dataset_3000_base.json", "r") as f:
    final_benchmark = json.load(f)

# ---- LOAD PROGRESS ----
processed_indices = set()
results = []

if os.path.exists(INCREMENTAL_FILE):
    with open(INCREMENTAL_FILE, "r") as f:
        for line in f:
            entry = json.loads(line)
            results.append(entry)
            processed_indices.add(entry["question"])  # or use idx if unique
    print(f"Resuming, {len(processed_indices)} entries already processed")

# ---- PROCESS BENCHMARK ----
for idx, item in enumerate(final_benchmark):
    # Skip if already done
    if item["question"] in processed_indices:
        continue

    print("TURN:", idx)

    # Build prompt
    prompt = f"You are a helpful math assistant.\nUser: {item['prompt']}\nAssistant:"

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # Generate
    output_ids = model.generate(
        **inputs,
        max_new_tokens=10000,
        temperature=0.0,
        do_sample=False,
    )

    # Decode
    model_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if model_output.startswith(prompt):
        model_output = model_output[len(prompt):].strip()

    print(model_output)
    print()

    # Append simplified entry
    entry = {
        "category": item["category"],
        "question": item["question"],
        "answer": item["answer"],
        "model_output": model_output
    }
    results.append(entry)

    # Write incrementally as JSONL
    with open(INCREMENTAL_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

# ---- SAVE FINAL JSON LIST ----
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved final results to {OUTPUT_FILE}")

