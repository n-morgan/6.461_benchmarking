import re
import json

def parse_model_answer(model_output: str) -> str:
    """Extract text inside <answer>...</answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", model_output, re.DOTALL)
    return match.group(1).strip() if match else ""

def exact_match(ref: str, pred: str) -> bool:
    """Simple exact match between reference and prediction."""
    return ref.strip() == pred.strip()

# Load the model-only benchmark dataset
with open("model_responses.json", "r") as f:
    data = json.load(f)

# Parse model answers and compute exact-match
num_correct = 0
for item in data:
    parsed = parse_model_answer(item.get("model_output", ""))
    # print("answer: " {item["answer"]} \n "parsed_answer: " {item["parsed_answer"]})
    # print(f"answer: {item['answer']}\nparsed_answer: {item['model_output']}")

    item["parsed_answer"] = parsed
    item["exact_match"] = exact_match(item["answer"], parsed)
    if item["exact_match"]:
        num_correct += 1



# Compute overall score
total = len(data)
score = num_correct / total * 100
print(f"Benchmark completed: {num_correct}/{total} correct")
print(f"Exact match accuracy: {score:.2f}%")

# Save scored dataset
with open("benchmark_dataset_scored.json", "w") as f:
    json.dump(data, f, indent=2)

print("Saved scored dataset to benchmark_dataset_scored.json")

