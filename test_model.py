import re
import json
import os
import sys
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


def load_datasets(config: dict[str, Any]) -> dict[str, Any]:
    datasets = {}
    for method in config["methods"]:
        with open(f"{method}_benchmark_dataset.json", "r") as f:
            datasets[method] = json.load(f)

    # number of prompts to test for each method
    n_prompts = min(map(len, datasets.values()))
    config["n_prompts"] = min(config["n_prompts"], n_prompts) # save to config

    return datasets


def ensure_dir(dir: str) -> Path:
    project_dir = Path(__file__).parent
    path = project_dir / dir
    path.mkdir(exist_ok=True)
    return path


def setup_log_files(config: dict[str, Any]) -> None:
    # Setup log directory
    log_dir = ensure_dir("logs")

    base = "model_responses"
    out_log_files = list(map(lambda method: log_dir / f"{config["model"]}_{method}_{base}.out", config["methods"]))
    err_log_files = list(map(lambda method: log_dir / f"{config["model"]}_{method}_{base}.err", config["methods"]))

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

    config.update({
        "out_logs": dict(zip(config["methods"], out_log_files)),
        "err_logs": dict(zip(config["methods"], err_log_files))
    })


def setup_locks(config: dict[str, Any]) -> None:
    console_lock = Lock()

    out_locks = {method: Lock() for method in config["methods"]}
    err_locks = {method: Lock() for method in config["methods"]}

    config.update({
        "cons_lock": console_lock,
        "out_locks": out_locks,
        "err_locks": err_locks
    })


def safe_out_log(config: dict[str, Any], method: str, *args: Any, **kwargs: Any) -> None:
    with (
        config["cons_lock"], 
        config["out_locks"][method],
        open(config["out_logs"][method], "a") as out
    ): # to stdout and out_log
        print(f"[{method.upper()}]", *args, **kwargs)
        print(*args, **kwargs, file=out)


def safe_err_log(config: dict[str, Any], method: str, *args: Any, **kwargs: Any) -> None:
    with (
        config["err_locks"][method],
        open(config["err_logs"][method], "a") as err
    ):
        print(*args, **kwargs, file=err)


def aggregate_benchmarks(config: dict[str, Any], datasets: dict[str, Any]) -> tuple[list[Any], list[Any]]:
    ids, aggregate_prompts = [], []
    for method, prompts in datasets.items():
        aggregate_prompts.extend(prompts[:config["n_prompts"]])
        # (prompt index (1-indexed) in resp. dataset, method string (e.g. "base", "scope", "cot", etc.))
        ids.extend(zip(range(1, config["n_prompts"]+1), [method] * config["n_prompts"]))
    return ids, aggregate_prompts


def model_output_extractor(config: dict[str, Any]) -> Any:
    def extract_model_output(id: tuple[int, str], item: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        idx, method = id

        messages = [
            {"role": "system", "content": "You are a helpful math assistant."},
            {"role": "user", "content": item["prompt"]}
        ]

        retries_left = config["n_retries"]
        while retries_left >= 0:
            try:
                # Call the v1/chat/completions endpoint
                response = client.chat.completions.create(
                    model=config["model"],
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
                    safe_err_log(config, method, err_msg, f"Out of retries\n", sep="\n")
                    break
                else:
                    safe_err_log(config, method, err_msg, f"Retrying... ({retries_left} retries left)\n", sep="\n")
        
        safe_out_log(config, method, f"Prompt {idx}:", model_output, sep="\n", end="\n\n")

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

    # return a closure that captures model from outer scope for
    # use in executor.map in compute_and_save_results()
    return extract_model_output


def group_results_by_method(aggregate_results: list[tuple[str, dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    grouped_results = defaultdict(list)
    for method, result_entry in aggregate_results:
        grouped_results[method].append(result_entry)
    return grouped_results


def compute_and_save_results(config: dict[str, Any], datasets: dict[str, Any]) -> None:
    ids, prompts = aggregate_benchmarks(config, datasets)
    with ThreadPoolExecutor(max_workers=config["n_workers"]) as executor:
        aggregate_results = executor.map(model_output_extractor(config), ids, prompts)
    grouped_results = group_results_by_method(aggregate_results)
    output_dir = ensure_dir("model_outputs")
    for method, result_entries in grouped_results.items():
        with open(output_dir / f"{config["model"]}_{method}_model_responses.json", "w") as f:
            json.dump(result_entries, f, indent=2)


def print_usage() -> None:
    usage = """
USAGE:
python3 test_model.py --model | -m {model} [--datasets | -d [b][r][c][s]] [--n-prompts | -n {n}] [--n-workers | -w {w}] [--n-retries | -r {r}]

Configures script to run on {model} with the dataset flags (default: all) 
specifying which benchmark datasets to use for generating model outputs;
up to {n} (default: min(1000, min(map(len, datasets)))) prompts from each dataset are
used, with up to {w} (default: (os.cpu_count() or 8) * 2) concurrent workers, and up to
{r} (default: 0) retries on failure for each prompt.

Dataset flags:
    b => base benchmark dataset 
    r => react benchmark dataset
    c => cot benchmark dataset
    s => scope benchmark dataset
"""
    print(usage)


def parse_args() -> dict[str, Any]:
    args = sys.argv[1:]
    n_args = len(args)

    aliases = {
        "b": "base",
        "r": "react",
        "c": "cot",
        "s": "scope",
    }

    config = {}

    def finished_parsing(reqs: tuple[str, ...] = ("model",)) -> bool:
        return all(req in config for req in reqs)

    def apply_defaults(config: dict[str, Any]) -> None:
        defaults = {
            "methods": list(aliases.values()),
            "n_workers": (os.cpu_count() or 8) * 2,
            "n_retries": 0,
            "n_prompts": 1000
        }
        config.update({
            k: v
            for k, v in defaults.items()
            if k not in config
        })

    i = 0
    while i < n_args - 1:
        if args[i] in ("--model", "-m"):
            config["model"] = args[i+1]
        elif args[i] in ("--datasets", "-d"): 
            matching = set(aliases.keys()).intersection(set(args[i+1]))
            if (not matching or len(matching) != len(args[i+1])):
                break # no/bad flag passed in
            config["methods"] = list(map(lambda k: aliases[k], matching))
        elif args[i] in ("--n-prompts", "-n"):
            config["n_prompts"] = int(args[i+1])
        elif args[i] in ("--n-workers", "-w"):
            config["n_workers"] = int(args[i+1])
        elif args[i] in ("--n-retries", "-r"):
            config["n_retries"] = int(args[i+1])
        else:
            print(f"Unknown argument: {args[i]}")
            print_usage()
            exit(-1)
        i += 2

    apply_defaults(config)

    if finished_parsing():
        return config

    print("Bad/missing arguments!")
    print_usage()
    exit(-1)


def print_config(config: dict[str, Any], indent: str = " " * 4) -> None:
    print("Running with the following configuration:")
    print("{")
    for k, v in config.items():
        print(f"{indent}\"{k}\": {v}")
    print("}\n\n")


def main() -> None:
    config = parse_args()
    print_config(config) # for debugging

    datasets = load_datasets(config)
    setup_log_files(config)
    setup_locks(config)

    start_time = time.perf_counter()
    compute_and_save_results(config, datasets)
    duration = time.perf_counter() - start_time

    print(f"\nSaved questions, answers, and model outputs to \"model_outputs/{{model}}_{{method}}_model_responses.json\"")
    print(f"Saved output messages to \"logs/{{model}}_{{method}}_model_responses.out\"")
    print(f"Saved error messages to \"logs/{{model}}_{{method}}_model_responses.err\"")

    print(f"\nCompleted in {duration:.3f} seconds!")


if __name__ == "__main__":
    main()
