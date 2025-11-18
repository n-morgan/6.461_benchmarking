import re
import json
from openai import OpenAI

# Initialize client
client = OpenAI(
    base_url="http://0.0.0.0:30000/v1",
    api_key="sk",
)

# Load benchmark dataset (with question, answer, cot_prompt)
with open("benchmark_dataset_3000_react.json", "r") as f:
    final_benchmark = json.load(f)
# Collect simplified results
results = []

for idx,item in enumerate(final_benchmark):
    print("TURN: ", idx)
    messages = [
        {"role": "system", "content": "You are a helpful math assistant."},
        {"role": "user", "content": item["prompt"]}
    ]
    

    RETRYS = 0
    while RETRYS >= 0:
        try:
            # Call the v1/chat/completions endpoint
            response = client.chat.completions.create(
                model="tei",
                messages=messages,
                temperature=0.0,
                max_tokens=8000
            )
            break  # success → exit loop
        except Exception as e:
            print(f"Error: {e}")
            RETRYS -= 1  # decrease retry counter
            if RETRYS == -1:
                response = "<answer> response failed </answer>"
                break
            else:
                print(f"Retrying... ({RETRYS} left)")
    
    # Extract model iutput
    model_output = response.choices[0].message.content
    print(model_output)
    print()

    # Append simplified entry
    results.append({
        "category": item["category"],
        "question": item["question"],
        "answer": item["answer"],
        "model_output": model_output
    })

# Save to new JSON file
with open("model_responses_react.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved question, answer, and model_output to model_responses.json")

