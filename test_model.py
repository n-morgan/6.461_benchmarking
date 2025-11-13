import re
import json
from openai import OpenAI

# Initialize client
client = OpenAI(
    base_url="http://0.0.0.0:30000/v1",
    api_key="sk",
)

# Load benchmark dataset (with question, answer, cot_prompt)
with open("benchmark_dataset.json", "r") as f:
    final_benchmark = json.load(f)
print(1)
# Collect simplified results
results = []

for item in final_benchmark:
    print("HERE")
    messages = [
        {"role": "system", "content": "You are a helpful math assistant."},
        {"role": "user", "content": item["cot_prompt"]}
    ]
    print("PROMPT: ", item["cot_prompt"])
    
    # Call the v1/chat/completions endpoint
    response = client.chat.completions.create(
        model="tei",
        messages=messages,
        temperature=0.0
    )

    print("HERE AGAIN")
    
    # Extract model output
    model_output = response.choices[0].message.content
    print(model_output)

    # Append simplified entry
    results.append({
        "question": item["question"],
        "answer": item["answer"],
        "model_output": model_output
    })

# Save to new JSON file
with open("model_responses.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved question, answer, and model_output to model_responses.json")

