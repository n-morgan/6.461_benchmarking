from datasets import load_dataset
import random
import json
import re

import os

SCHEMA_DIR = "scope_schemas"

def load_schema_files(schema_dir=SCHEMA_DIR):
    """
    Loads all *_schema.txt and *_example.txx files into a dictionary:

    {
        name: {
            "schema": "...",
            "example": "..."
        }
    }
    """
    data = {}

    for fname in os.listdir(schema_dir):
        fpath = os.path.join(schema_dir, fname)

        # skip directories
        if not os.path.isfile(fpath):
            continue

        # match *_schema.txt
        if fname.endswith("_schema.txt"):
            name = fname[:-len("_schema.txt")]
            with open(fpath, "r") as f:
                data.setdefault(name, {})["schema"] = f.read()

        # match *_example.txx
        elif fname.endswith("_example.txt"):
            name = fname[:-len("_example.txt")]
            with open(fpath, "r") as f:
                data.setdefault(name, {})["example"] = f.read()

    return data



category_schemas = load_schema_files()

# Load dataset
dataset = load_dataset("tomg-group-umd/CLRS-Text-test", split="test_1")  # or "train"
# print(dataset)




## OPTIONS: base, cot, react, scope 

# SELCTION = cot

total_examples = 30

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



# samples_by_category = sample_by_category(dataset, algorithmsByGroup, n_per_algorithm)
# samples_by_category =  None


# with open("sample_by_cat_30.json", "w") as f:
# 
#     json.dump(samples_by_category, f, indent=2)


with open("sample_by_cat_30.json", "r") as f:
    samples_by_category = json.load(f)



print(samples_by_category["searching"]["quickselect"])

# print("\n===== DATASET SAMPLE CHECK =====\n")
# 
# for category, algos in samples_by_category.items():
#     print(f"\n=== Category: {category} ===")
#     
#     for algo, examples in algos.items():
#         print(f"\nAlgorithm: {algo}")
#         if not examples:
#             print("  (No examples found)")
#             continue
#         
#         ex = examples[0]  # print ONE example only
#         
#         print("  Question:")
#         print("   ", ex.get("question", "").strip())
#         
#         print("  Answer:")
#         print("   ", ex.get("answer", "").strip())




# for category, algos in samples_by_category.items():
#     total_in_category = sum(len(exs) for exs in algos.values())
#     print(f"{category}: {total_in_category} examples")
#     for algo, exs in algos.items():
#         print(f"  {algo}: {len(exs)} examples")

BASE_PROMPT = """You are a helpful math assistant adept at solving math problems


algorithm_name: {algorithm_name}
example_output_A: {example_output_A}
example_output_B: {example_output_B}
question: {question} 


On a new line at the end of your response. Output your answer in the form of example_output_A or example_output_B  with answer tags. DO NOT INCLUDE YOUR REASONING IN THE ANSWER TAGS

<answer>[your answer here]</answer>

"""

COT_PROMPT = """

You are a helpful math assistant adept at solving math problems. Always show your reasoning **step by step** before giving the final answer.

algorithm_name: Solve Linear Equation
question: Solve 3x + 5 = 20
example_output_A: 86
example_output_B: 9

reasoning: 
Step 1: Identify the equation: 3x + 5 = 20
Step 2: Isolate the variable by subtracting 5 from both sides: 3x = 15
Step 3: Solve for x by dividing both sides by 3: x = 5

<answer>5</answer>

---

algorithm_name: Area of a Triangle
question: What is the area of a triangle with base 10 units and height 6 units?
example_output_A: 3
example_output_B: 47

reasoning:
Step 1: Identify the base and height: base = 10, height = 6
Step 2: Recall the area formula: A = 1/2 * base * height
Step 3: Substitute values: A = 1/2 * 10 * 6
Step 4: Multiply to find the area: A = 30

<answer>30</answer>

---

Now solve the following problem with clear, **step-by-step reasoning**. 

algorithm_name: {algorithm_name}
question: {question}
example_output_A: {example_output_A}
example_output_B: {example_output_B}

reasoning: 
[your reasoning here]

On a new line at the end of your response. Output your answer in the form of example_output_A or example_output_B  with answer tags. DO NOT INCLUDE YOUR REASONING IN THE ANSWER TAGS

<answer>[your answer here]</answer>
"""

REACT_PROMPT = """

You are a helpful math assistant adept at solving math problems. Use the **ReACT approach**: for each step, show your reasoning, decide if an action is needed, perform the action if necessary, then continue reasoning. Conclude with the final answer in <answer> tags.

algorithm_name: Solve Linear Equation with Reasoning and Action
question: Solve 3x + 5 = 20
example_output_A: 86
example_output_B: 9
reasoning: 
Step 1: Identify the equation: 3x + 5 = 20
Step 2: Reasoning: To solve for x, I need to isolate it.
Step 3: Action: Subtract 5 from both sides → 3x = 15
Step 4: Reasoning: Now divide both sides by 3 to find x
Step 5: Action: Divide 15 by 3 → x = 5

<answer>5</answer>

---

algorithm_name: Area of a Triangle with Reasoning and Action
question: What is the area of a triangle with base 10 units and height 6 units?
example_output_A: 3
example_output_B: 47
reasoning: 
Step 1: Identify base and height: base = 10, height = 6
Step 2: Reasoning: The formula for area is A = 1/2 * base * height
Step 3: Action: Substitute the values → A = 1/2 * 10 * 6
Step 4: Reasoning: Multiply to compute the area
Step 5: Action: 1/2 * 10 * 6 = 30

<answer>30</answer>

---

Now solve the following problem using the **ReACT approach**. Show step-by-step reasoning, take actions if needed. 

algorithm_name: {algorithm_name}
question: {question}
example_output_A: {example_output_A}
example_output_B: {example_output_B}


On a new line at the end of your response. Output your answer in the form of example_output_A or example_output_B  with answer tags. DO NOT INCLUDE YOUR REASONING IN THE ANSWER TAGS

<answer>[your answer here]</answer>


"""




SCOPE_PROMPT = """
You are a helpful math assistant adept at solving math problems. 
algorithm_name: {algorithm_name}
example_output_A: {example_output_A}
example_output_B: {example_output_B}



Use the following schema to work through your thought process

worked_example: {worked_example}
question: {question}
algorithm_schema: {algorithm_schema}


On a new line at the end of your response. Output your answer with answer tags. DO NOT INCLUDE YOUR REASONING IN THE ANSWER TAGS

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
    Split answer by commas until the final |.
    Example: "5, 5, 5, 5 | 1" -> ["5","5","5","5","1"]
    """
    if "|" not in answer_str:
        return [s.strip() for s in answer_str.split(",")]

    left, right = answer_str.split("|", 1)
    steps = [s.strip() for s in left.split(",") if s.strip()]
    steps.append(right.strip())
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
CHOSEN_PROMPT = REACT_PROMPT 
for category, algos in samples_by_category.items():
    for algo, examples in algos.items():
        for ex in examples:
            # Remove 'initial_trace trace | pred:' from question
            question_clean = clean_question(ex["question"])
            example_output_A = parse_example_output(ex["answer"])
            example_output_B = parse_example_output(ex["answer"])
            final_answer = parse_final_answer(ex["answer"])

            # Fill in COT prompt
            if CHOSEN_PROMPT == SCOPE_PROMPT:
                
                schema_text = category_schemas.get(category, {}).get("schema")
                example_text = category_schemas.get(category, {}).get("example")
                
                worked_example=schema_text
                algorithm_schema=example_text
                prompt = CHOSEN_PROMPT.format(
                    algorithm_name=algo,
                    example_output_A=example_output_A,
                    example_output_B=example_output_B, 
                    question=question_clean,
                    worked_example=worked_example,
                    algorithm_schema=algorithm_schema, 

    
                )
                
            else: 
                prompt = CHOSEN_PROMPT.format(
                    algorithm_name=algo,
                    example_output_A=example_output_A,
                    example_output_B=example_output_B,
                    question=question_clean
                )

            final_benchmark.append({
                "category": category,
                "algorithm": algo,
                "question": question_clean,
                "example_output_A": example_output_A,
                "example_output_B": example_output_A,
                "prompt": prompt,
                "answer": final_answer
            })
# -----------------------------
# Save to JSON
# -----------------------------
with open("benchmark_dataset.json", "w") as f:
    json.dump(final_benchmark, f, indent=2)

print(f"Saved {len(final_benchmark)} benchmark examples to benchmark_dataset.json")






