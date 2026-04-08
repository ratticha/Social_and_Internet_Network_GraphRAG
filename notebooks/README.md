# Notebooks

This folder contains analysis notebooks for exploration and report preparation.

## Files

- `01_data_exploration.ipynb`: dataset overview, graph statistics, degree plots, split inspection
- `02_results_analysis.ipynb`: method comparison, Precision@K plots, prediction inspection

## How to use them

Run the pipeline first:

```bash
python scripts/download_data.py
python scripts/preprocess_data.py
python scripts/create_split.py --test-size 0.2 --negatives-per-positive 10 --seed 42
python scripts/run_experiments.py --top-k 10
```

Then open the notebooks in VS Code and select the project kernel.

## Current status

- `01_data_exploration.ipynb`: completed
- `02_results_analysis.ipynb`: completed

## Generated outputs

- `degree_distributions.png`
- `precision_at_k_comparison.png`
- `per_source_precision_boxplot.png`
- `positive_rate_by_rank.png`

## Environment note

Use the `C:\venvs\graphrag\Scripts\python.exe` interpreter or the matching notebook kernel in VS Code.
