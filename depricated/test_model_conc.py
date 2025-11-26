import json
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

client = OpenAI(
    base_url="http://0.0.0.0:30000/v1",
    api_key="sk",
)

with open("benchmark_dataset_300_scope.json", "r") as f:
    final_benchmark = json.load(f)

def ask_one(idx, item):
    messages = [
        {"role": "system", "content": "You are a helpful math assistant."},
        {"role": "user", "content": item["prompt"]},
    ]

    try:
        response = client.chat.completions.create(
            model="tei",
            messages=messages,
            temperature=0.0,
            max_tokens=8000,
            tools=[],
            tool_choice="none",
        )
        output = response.choices[0].message.content
    except Exception as e:
        output = f"<answer> failed: {str(e)} </answer>"

    return {
        "category": item["category"],
        "question": item["question"],
        "answer": item["answer"],
        "model_output": output,
        "index": idx
    }

# number of concurrent workers
WORKERS = 20

results = []
with ThreadPoolExecutor(max_workers=WORKERS) as executor:
    futures = [
        executor.submit(ask_one, idx, final_benchmark[idx])
        for idx in range(len(final_benchmark))
    ]

    for future in as_completed(futures):
        results.append(future.result())

# save results
with open("model_responses_trial.json", "w") as f:
    json.dump(results, f, indent=2)

print("saved model_responses.json")

