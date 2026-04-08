# Scripts

This folder contains the runnable entry-point scripts for the full GraphRAG pipeline.

## Pipeline order

Run the scripts in this order:

```bash
python scripts/download_data.py
python scripts/preprocess_data.py
python scripts/create_split.py --test-size 0.2 --negatives-per-positive 10 --seed 42
python scripts/run_experiments.py --top-k 10
```

## Script summary

### `download_data.py`

Objective:

- Download the Wikipedia Vote Network from SNAP.
- Extract the compressed file into the local `data/raw/` folder.

Observed output:

```text
Downloading dataset from https://snap.stanford.edu/data/wiki-Vote.txt.gz
Extracting wiki-Vote.txt.gz to data\raw\wiki-Vote.txt
Download complete.
```

Generated files:

- `data/raw/wiki-Vote.txt.gz`
- `data/raw/wiki-Vote.txt`

### `preprocess_data.py`

Objective:

- Read the raw text file.
- Remove comment lines and invalid rows.
- Save a clean directed edge list as CSV.

Observed output:

```text
Saved 103,689 cleaned edges to data/processed/graph_edges.csv
```

Generated file:

- `data/processed/graph_edges.csv`

### `create_split.py`

Objective:

- Split the cleaned edge list into training and testing edges.
- Sample negative edges for evaluation.

Observed output:

```text
Train edges: 82,951
Test edges: 20,738
Negative samples: 207,380
```

Generated files:

- `data/splits/train_edges.csv`
- `data/splits/test_edges.csv`
- `data/splits/negative_samples.csv`

### `run_experiments.py`

Objective:

- Build candidate links from the train graph and evaluation data.
- Score candidates with all implemented methods.
- Rank the Top-K predictions.
- Evaluate Precision@10.

Observed output summary:

```text
Loaded train=82,951, test=20,738, negative=207,380 edges.
Candidates=228,118, nodes=7,115, edges=82,951.
Methods selected: common_neighbors, jaccard, adamic_adar, personalized_pagerank
Finished experiment pipeline in 886.4 seconds.
```

Generated files:

- `results/metrics/evaluation_summary.csv`
- `results/metrics/per_source_precision.csv`
- `results/predictions/scored_candidates.csv`
- `results/predictions/topk_predictions.csv`

### `evaluate.py`

Objective:

- Recompute evaluation metrics from saved prediction scores without rerunning the full scoring process.

Use case:

- Helpful if you want to reevaluate saved outputs or change evaluation settings after scoring is already complete.

## Notes

- The slowest part of the pipeline is `personalized_pagerank`.
- On this setup, the full experiment run took about `14.8` minutes.
- The progress messages inside `run_experiments.py` are normal and help show that PageRank is still running.
