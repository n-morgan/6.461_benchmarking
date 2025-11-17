import re
import json
import os
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict
from threading import Lock
from pathlib import Path

# Initialize client
client = OpenAI(
    base_url="http://0.0.0.0:30000/v1",
    api_key="sk",
)

# Load benchmark dataset (with question, answer, cot_prompt)
with open("benchmark_dataset.json", "r") as f:
    final_benchmark = json.load(f)

# Useful constants
N_RETRIES = 2
N_CORES = os.cpu_count() or 8
N_WORKERS = N_CORES * 2
N_PROMPTS = len(final_benchmark)

PROJECT_DIR = Path(__file__).parent

# Responses file
RESPONSES_FILE = PROJECT_DIR / "model_responses.json"

# Setup log directory/files
LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True, parents=True)

OUT_LOG = LOG_DIR / "model_responses.out"
ERR_LOG = LOG_DIR / "model_responses.err"

try: # remove log files if they already exist
    OUT_LOG.unlink()
    ERR_LOG.unlink()
except FileNotFoundError:
    pass

# Lock to prevent interleaving prints
out_lock, err_lock = Lock(), Lock()
def safe_out_log(*args: Any, **kwargs: Any) -> None:
    with out_lock, open(OUT_LOG, "a") as out: # to stdout and out log
        print(*args, **kwargs)
        print(*args, **kwargs, file=out)

def safe_err_log(*args: Any, **kwargs: Any) -> None:
    with err_lock, open(ERR_LOG, "a") as err:
        print(*args, **kwargs, file=err)

def extract_model_output(idx: int, item: Dict[str, Any]) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": "You are a helpful math assistant."},
        {"role": "user", "content": item["prompt"]}
    ]

    retries_left = N_RETRIES
    while retries_left >= 0:
        try:
            # Call the v1/chat/completions endpoint
            response = client.chat.completions.create(
                model="tei",
                messages=messages,
                temperature=0.0,
                max_tokens=8000
            )
            # Extract model output
            model_output = response.choices[0].message.content
            break  # success → exit loop
        except Exception as e:
            err_msg = f"[ERROR] Prompt {idx+1}: {e}"
            retries_left -= 1  # decrease retry counter
            if retries_left == -1:
                model_output = "<answer> response failed </answer>"
                safe_err_log(err_msg, f"Out of retries\n", sep="\n")
                break
            else:
                safe_err_log(err_msg, f"Retrying... ({retries_left} retries left)\n", sep="\n")
    
    safe_out_log(f"Prompt {idx+1}:", model_output, sep="\n", end="\n\n")

    return {
        # compile simplified result entry
        "category": item["category"],
        "question": item["question"],
        "answer": item["answer"],
        "model_output": model_output
    }

with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
    results = executor.map(extract_model_output, range(N_PROMPTS), final_benchmark)

# Save to new JSON file
with open(RESPONSES_FILE, "w") as f:
    json.dump(list(results), f, indent=2)

print(f"Saved question, answer, and model_output to \"{os.path.relpath(RESPONSES_FILE, PROJECT_DIR)}\"")
print(f"Saved output messages to \"{os.path.relpath(OUT_LOG, PROJECT_DIR)}\"")
print(f"Saved error messages to \"{os.path.relpath(ERR_LOG, PROJECT_DIR)}\"")
