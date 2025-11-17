import re
import json
from collections import defaultdict


def parse_model_answer(model_output: str) -> str:
    """Extract text inside <answer>...</answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", model_output, re.DOTALL)
    return match.group(1).strip() if match else ""


def exact_match(ref: str, pred: str) -> bool:
    def normalize(s: str) -> str:
        return s.replace(",", "").split()
    return normalize(ref) == normalize(pred)


# Load the model-only benchmark dataset
with open("model_responses.json", "r") as f:
    data = json.load(f)

# Containers for score tracking
num_correct = 0
category_correct = defaultdict(int)
category_total = defaultdict(int)

# Parse model answers and compute exact-match
for item in data:

    category = item["category"]   # ← fixed the bug here

    parsed = parse_model_answer(item.get("model_output", ""))

    item["parsed_answer"] = parsed
    item["exact_match"] = exact_match(item["answer"], parsed)

    # global accuracy
    if item["exact_match"]:
        num_correct += 1

    # category-level accuracy
    category_total[category] += 1
    if item["exact_match"]:
        category_correct[category] += 1


# Compute overall score
total = len(data)
score = num_correct / total * 100

print(f"Benchmark completed: {num_correct}/{total} correct")
print(f"Exact match accuracy: {score:.2f}%\n")

# --- Per-category summary ---
print("Per-category accuracy:")
for cat in sorted(category_total.keys()):
    c = category_correct[cat]
    t = category_total[cat]
    print(f"  {cat}: {c}/{t} = {100*c/t:.2f}%")

# Save scored dataset
with open("benchmark_dataset_scored.json", "w") as f:
    json.dump(data, f, indent=2)

print("\nSaved scored dataset to benchmark_dataset_scored.json")

