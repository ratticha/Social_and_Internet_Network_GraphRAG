# Scripts

This folder contains the runnable entry-point scripts for the project pipeline.

## Scripts

- `download_data.py`: downloads and extracts the SNAP dataset
- `preprocess_data.py`: cleans the raw edge list into CSV format
- `create_split.py`: creates the train/test split and negative samples
- `run_experiments.py`: computes link prediction scores and evaluates Top-K performance
- `evaluate.py`: re-evaluates saved score files without rerunning scoring

## Recommended order

```bash
python scripts/download_data.py
python scripts/preprocess_data.py
python scripts/create_split.py --test-size 0.2 --negatives-per-positive 10 --seed 42
python scripts/run_experiments.py --top-k 10
```

## Your current progress

- `download_data.py`: completed
- `preprocess_data.py`: completed
- `create_split.py`: completed
- `run_experiments.py`: completed

## Notes

- The script now prints progress during the slow personalized PageRank stage
- The full run with all four methods completed successfully on this project
