# GraphRAG Link Prediction Project

This repository implements a simplified GraphRAG retrieval module for the Wikipedia Vote Network assignment. The project frames GraphRAG retrieval as a link prediction task: if user `A` is likely to connect to user `B`, then `B` is treated as relevant context for `A`.

## Project goals

- Process the SNAP Wikipedia Vote Network dataset
- Build a training and testing split from the edge list
- Implement multiple link prediction algorithms
- Rank candidate links for each source node
- Evaluate retrieval quality with Precision@K

## Repository structure

```text
.
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── notebooks/
├── reports/
│   └── figures/
├── results/
│   ├── metrics/
│   └── predictions/
├── scripts/
├── src/
│   └── graphrag/
└── tests/
```

## Implemented baselines

- `common_neighbors`
- `jaccard`
- `adamic_adar`
- `personalized_pagerank`

The local similarity baselines use an undirected projection of the training graph. The PageRank-style baseline uses personalized PageRank on the directed training graph.

## Current Project Status

- Dataset downloaded and extracted
- Edge list cleaned into CSV format
- 80/20 train-test split created with negative sampling
- All four baselines executed successfully
- Both analysis notebooks completed
- Report figures generated

## Dataset And Split Summary

- Cleaned directed edges: `103,689`
- Training edges: `82,951`
- Testing edges: `20,738`
- Negative sampled edges: `207,380`
- Source nodes evaluated: `2,807`

## Experiment Results

`Precision@10` on the held-out test edges:

| Method | Mean Precision@10 | Std. Dev. | Total Hits |
|---|---:|---:|---:|
| Personalized PageRank | 0.3369 | 0.2389 | 9456 |
| Common Neighbors | 0.3029 | 0.2328 | 8502 |
| Adamic/Adar | 0.3027 | 0.2320 | 8496 |
| Jaccard | 0.2941 | 0.2320 | 8254 |

The best-performing method is `personalized_pagerank`.

## Quick start

1. Create a virtual environment and install dependencies.

```bash
C:\Users\User\.pyenv\pyenv-win\versions\3.11.4\python.exe -m venv C:\venvs\graphrag
C:\venvs\graphrag\Scripts\activate
python -m pip install -r requirements.txt
```

On Windows, a short virtual-environment path such as `C:\venvs\graphrag` helps avoid long-path installation errors.

2. Download the dataset from SNAP.

```bash
python scripts/download_data.py
```

3. Preprocess the raw text file into a clean edge list.

```bash
python scripts/preprocess_data.py
```

4. Create a train/test split and sample negative edges.

```bash
python scripts/create_split.py --test-size 0.2 --negatives-per-positive 10 --seed 42
```

5. Run the baseline experiments.

```bash
python scripts/run_experiments.py --top-k 10
```

6. Re-run evaluation from saved scores if needed.

```bash
python scripts/evaluate.py --top-k 10
```

7. Open the notebooks in VS Code and use the `C:\venvs\graphrag\Scripts\python.exe` interpreter or kernel.

## Main outputs

- Processed edges: `data/processed/graph_edges.csv`
- Train edges: `data/splits/train_edges.csv`
- Test edges: `data/splits/test_edges.csv`
- Negative samples: `data/splits/negative_samples.csv`
- Scored candidates: `results/predictions/scored_candidates.csv`
- Top-K predictions: `results/predictions/topk_predictions.csv`
- Metrics summary: `results/metrics/evaluation_summary.csv`
- Per-source metrics: `results/metrics/per_source_precision.csv`
- Notebook figures: `reports/figures/*.png`

## Notes

- The `personalized_pagerank` baseline requires `scipy`, which is included in `requirements.txt`.
- If `run_experiments.py` fails with a `scipy` error, rerun `python -m pip install -r requirements.txt`.
- On this Windows setup, the short virtual environment path `C:\venvs\graphrag` worked better than the project-local `.venv`.

## Suggested report sections

- Problem statement
- Dataset description
- Preprocessing and split strategy
- Algorithms and assumptions
- Evaluation setup
- Results and comparison
- Limitations and future improvements
