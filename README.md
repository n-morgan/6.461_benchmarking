# Math Algorithm Benchmark Project

This repository benchmarks models on algorithmic math problems using chain-of-thought prompts. It includes scripts to generate benchmark data, test the model, and evaluate outputs.

## File Structure

- `make_base_bench.py` – Generate the benchmark dataset with cleaned questions, example outputs, and COT prompts.
- `test_model.py` – Run the model on the benchmark dataset and collect model outputs.
- `eval.py` – Parse model outputs, extract answers, and compute exact-match scores.

## Workflow

1. **Generate the benchmark dataset**

```bash
python make_base_bench.py
```

This produces a JSON file (`benchmark_dataset.json`) containing:
- `question`
- `example_output`
- `cot_prompt`
- `answer` (ground truth)

2. **Run the model on the dataset**

```bash
python test_model.py
```

This produces a JSON file (`model_responses.json`) containing:
- `question`
- `answer` (ground truth)
- `model_output` (model’s response)

3. **Evaluate the model outputs**

```bash
python eval.py
```

This parses `<answer>...</answer>` from the model outputs, computes exact-match accuracy, and saves the scored dataset to `benchmark_dataset_scored.json`.

