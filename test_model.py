import re
import json
import os
import time
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from threading import Lock
from pathlib import Path
from collections import defaultdict


# Initialize client
client = OpenAI(
    base_url="http://0.0.0.0:30000/v1",
    api_key="sk",
)


METHODS = ["base", "react", "cot", "scope"]

def load_datasets() -> tuple[int, dict[str, Any]]:
    datasets = {}
    for method in METHODS:
        with open(f"{method}_benchmark_dataset.json", "r") as f:
            datasets[method] = json.load(f)
    # number of prompts to test for each method
    n_prompts = min(map(len, datasets.values()))
    return n_prompts, datasets

N_PROMPTS, DATASETS = load_datasets()


PROJECT_DIR = Path(__file__).parent

def ensure_dir(dir: str) -> Path:
    path = PROJECT_DIR / dir
    path.mkdir(exist_ok=True)
    return path


def setup_log_files() -> tuple[dict[str, Path], dict[str, Path]]:
    # Setup log directory
    log_dir = ensure_dir("logs")

    base = "model_responses"

    out_log_files = list(map(lambda method: log_dir / f"{method}_{base}.out", METHODS))
    err_log_files = list(map(lambda method: log_dir / f"{method}_{base}.err", METHODS))

    # remove log files if they exist (we overwrite response files so dont need to be unlinked)
    for out, err in zip(out_log_files, err_log_files):
        try:
            out.unlink()
        except FileNotFoundError:
            pass
        try:
            err.unlink()
        except FileNotFoundError:
            pass

    return dict(zip(METHODS, out_log_files)), dict(zip(METHODS, err_log_files))

OUT_LOG_FILES, ERR_LOG_FILES = setup_log_files()


def setup_locks() -> tuple[Any]:
    out_locks = {method: Lock() for method in METHODS}
    err_locks = {method: Lock() for method in METHODS}

    return out_locks, err_locks

OUT_LOCKS, ERR_LOCKS = setup_locks()


# Useful constants
N_RETRIES = 2
N_CORES = os.cpu_count() or 8
N_WORKERS = N_CORES * 2


def safe_out_log(method: str, *args: Any, **kwargs: Any) -> None:
    with OUT_LOCKS[method], open(OUT_LOG_FILES[method], "a") as out: # to stdout and out_log
        print(f"[{method.upper()}]", *args, **kwargs)
        print(*args, **kwargs, file=out)

def safe_err_log(method: str, *args: Any, **kwargs: Any) -> None:
    with ERR_LOCKS[method], open(ERR_LOG_FILES[method], "a") as err:
        print(*args, **kwargs, file=err)


def aggregate_benchmarks() -> tuple[list[Any], list[Any]]:
    ids, aggregate_prompts = [], []
    for method, prompts in DATASETS.items():
        aggregate_prompts.extend(prompts[:N_PROMPTS])
        # (prompt index (1-indexed) in resp. dataset, method string (e.g. "base", "scope", "cot", etc.))
        ids.extend(zip(range(1, N_PROMPTS+1), [method] * N_PROMPTS))
    return ids, aggregate_prompts


def extract_model_output(id: tuple[int, str], item: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    idx, method = id

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
            err_msg = f"[ERROR] Prompt {idx}: {e}"
            retries_left -= 1  # decrease retry counter
            if retries_left == -1:
                model_output = "<answer> response failed </answer>"
                safe_err_log(method, err_msg, f"Out of retries\n", sep="\n")
                break
            else:
                safe_err_log(method, err_msg, f"Retrying... ({retries_left} retries left)\n", sep="\n")
    
    safe_out_log(method, f"Prompt {idx}:", model_output, sep="\n", end="\n\n")

    return (
        method,
        {
            # simplified result entry
            "category": item["category"],
            "question": item["question"],
            "answer": item["answer"],
            "model_output": model_output
        }
    )


def group_results_by_method(aggregate_results: list[tuple[str, dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    grouped_results = defaultdict(list)
    for method, result_entry in aggregate_results:
        grouped_results[method].append(result_entry)
    return grouped_results


def compute_and_save_results() -> None:
    ids, prompts = aggregate_benchmarks()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        aggregate_results = executor.map(extract_model_output, ids, prompts)
    grouped_results = group_results_by_method(aggregate_results)
    output_dir = ensure_dir("model_outputs")
    for method, result_entries in grouped_results.items():
        with open(output_dir / f"{method}_model_responses.json", "w") as f:
            json.dump(result_entries, f, indent=2)


if __name__ == "__main__":
    start_time = time.perf_counter()
    compute_and_save_results()
    duration = time.perf_counter() - start_time

    print(f"\nTook {duration:.3f} seconds!\n")

    print(f"Saved questions, answers, and model outputs to \"model_outputs/{{method_name}}_model_responses.json\"")
    print(f"Saved output messages to \"logs/{{method_name}}_model_responses.out\"")
    print(f"Saved error messages to \"logs/{{method_name}}_model_responses.err\"")
