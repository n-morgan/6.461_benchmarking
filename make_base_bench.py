from datasets import load_dataset
import random
import json
import re

# Load dataset
dataset = load_dataset("tomg-group-umd/CLRS-Text-test", split="test_1")  # or "train"
print(dataset)


total_examples = 300

# Algorithm groupings
algorithmsByGroup = {
    "sorting": ["insertion_sort", "bubble_sort", "heapsort", "quicksort"],
    "searching": ["minimum", "binary_search", "quickselect"],
    "divideAndConquer": ["find_maximum_subarray_kadane"],
    "greedy": ["activity_selector", "task_scheduling"],
    "dynamicProgramming": ["matrix_chain_order", "lcs_length", "optimal_bst"],
    "graphs": [
        "dfs", "bfs", "topological_sort", "articulation_points", "bridges",
        "strongly_connected_components", "mst_kruskal", "mst_prim",
        "bellman_ford", "dijkstra", "dag_shortest_paths", "floyd_warshall"
    ],
    "strings": ["naive_string_matcher", "kmp_matcher"],
    "geometry": ["segments_intersect", "graham_scan", "jarvis_march"]
}

# Compute number of algorithms and examples per algorithm
all_algorithms = [algo for group in algorithmsByGroup.values() for algo in group]

n_per_algorithm = total_examples // len(all_algorithms)
print(f"Sampling {n_per_algorithm} examples per algorithm")

def sample_by_category(dataset, algorithmsByGroup, n_per_algorithm):
    """
    Sample n_per_algorithm examples for each algorithm, organized by category.
    Returns: {category_name: {algorithm_id: [examples]}}
    """
    grouped_samples = {}

    for category, algorithms in algorithmsByGroup.items():
        grouped_samples[category] = {}
        for algo in algorithms:
            # Filter dataset for this algorithm
            
            algo_examples = [ex for ex in dataset if ex["algo_name"] == algo]
            
            # Sample n_per_algorithm randomly (or all if not enough)
            if len(algo_examples) >= n_per_algorithm:
                grouped_samples[category][algo] = random.sample(algo_examples, n_per_algorithm)
            else:
                grouped_samples[category][algo] = algo_examples
    
    return grouped_samples



samples_by_category = sample_by_category(dataset, algorithmsByGroup, n_per_algorithm)
# samples_by_category =  None


with open("sample_by_cat.json", "w") as f:

    json.dump(samples_by_category, f, indent=2)


#with open("sample_by_cat.json", "r") as f:
#    samples_by_category = json.load(f)
#
#
#
#print(samples_by_category["graphs"]["dfs"][0])


# for category, algos in samples_by_category.items():
#     total_in_category = sum(len(exs) for exs in algos.values())
#     print(f"{category}: {total_in_category} examples")
#     for algo, exs in algos.items():
#         print(f"  {algo}: {len(exs)} examples")

BASE_PROMPT = """You are a helpful math assistant adept at solving math problems


algorithm_name: {algorithm_name}
example_output: {example_output}
question: {question} 


On a new line at the end of your response. Output your answer in the form of example_output with answer tags. DO NOT INCLUDE YOUR REASONING IN THE ANSWER TAGS

<answer>[your answer here]</answer>

"""

# -----------------------------
# Parse dataset answers and create prompts
# -----------------------------
def clean_question(question_str):
    """Remove 'initial_trace: [...]' and 'trace | ...:' from question string."""
    # Remove initial_trace: [...] pattern
    question_clean = re.sub(r"initial_trace: \[.*?\]\n", "", question_str)
    # Remove trace | ...: pattern
    question_clean = re.sub(r"trace \| .*?:", "", question_clean)
    return question_clean.strip()

def parse_answer_to_list(answer_str):
    """
    Extract all bracketed or parenthesized steps from the answer string.
    Supports [] or ().
    """
    steps = re.findall(r"(\[.*?\]|\(.*?\))", answer_str)
    return steps


def parse_example_output(answer_str):
    """Pick a random intermediate step from the answer string (exclude last step)."""
    steps = parse_answer_to_list(answer_str)
    if len(steps) <= 1:
        return steps[0].split('|')[0].strip() if steps else ""
    # Random choice from trace (exclude final answer)
    return random.choice(steps[:-1]).split('|')[0].strip()

def parse_final_answer(answer_str):
    """Extract the final answer (after | in the last element)."""
    steps = parse_answer_to_list(answer_str)
    last = steps[-1] if steps else ""
    if '|' in last:
        return last.split('|')[1].strip()
    return last.strip()
# -----------------------------
# Build final benchmark dataset
# -----------------------------
final_benchmark = []

for category, algos in samples_by_category.items():
    for algo, examples in algos.items():
        for ex in examples:
            # Remove 'initial_trace trace | pred:' from question
            question_clean = clean_question(ex["question"])
            example_output = parse_example_output(ex["answer"])
            final_answer = parse_final_answer(ex["answer"])

            # Fill in COT prompt
            prompt = BASE_PROMPT.format(
                algorithm_name=algo,
                example_output=example_output,
                question=question_clean
            )

            final_benchmark.append({
                "category": category,
                "algorithm": algo,
                "question": question_clean,
                "example_output": example_output,
                "cot_prompt": prompt,
                "answer": final_answer
            })
# -----------------------------
# Save to JSON
# -----------------------------
with open("benchmark_dataset.json", "w") as f:
    json.dump(final_benchmark, f, indent=2)

print(f"Saved {len(final_benchmark)} benchmark examples to benchmark_dataset.json")






