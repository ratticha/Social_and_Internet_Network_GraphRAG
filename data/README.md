# Data Folder

This folder stores the dataset and the intermediate files created by the preprocessing and split stages.

## Subfolders

- `raw/`: original downloaded files from SNAP.
- `processed/`: cleaned edge-list CSV used by the pipeline.
- `splits/`: train/test split files and sampled negative edges.

## What each stage produced

### `raw/`

Produced by:

- `python scripts/download_data.py`

Files:

- `wiki-Vote.txt.gz`
- `wiki-Vote.txt`

Purpose:

- Keeps the original Wikipedia Vote dataset exactly as downloaded and extracted.

### `processed/`

Produced by:

- `python scripts/preprocess_data.py`

Files:

- `graph_edges.csv`

Purpose:

- Stores the cleaned directed edge list used in the project.

Observed output:

```text
Saved 103,689 cleaned edges to data/processed/graph_edges.csv
```

### `splits/`

Produced by:

- `python scripts/create_split.py --test-size 0.2 --negatives-per-positive 10 --seed 42`

Files:

- `train_edges.csv`
- `test_edges.csv`
- `negative_samples.csv`

Purpose:

- `train_edges.csv`: graph used by the link prediction methods
- `test_edges.csv`: hidden true edges for evaluation
- `negative_samples.csv`: non-edge samples for ranking and evaluation

Observed output:

```text
Train edges: 82,951
Test edges: 20,738
Negative samples: 207,380
```

## Data summary

- Cleaned edges: `103,689`
- Train edges: `82,951`
- Test edges: `20,738`
- Negative sampled edges: `207,380`

## Why this matters

The data pipeline creates a realistic evaluation setup for GraphRAG-style retrieval:

- the train graph acts as the known context graph
- the hidden test edges represent missing relevant links
- the negative samples represent irrelevant candidates that the model should rank lower
