# GraphRAG Link Prediction Project

This repository is a starter template for the Wikipedia Vote Network assignment on GraphRAG and context retrieval. The project frames GraphRAG retrieval as a link prediction task: if user `A` is likely to connect to user `B`, then `B` is treated as relevant context for `A`.

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

## Quick start

1. Create a virtual environment and install dependencies.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

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

## Main outputs

- Processed edges: `data/processed/graph_edges.csv`
- Train edges: `data/splits/train_edges.csv`
- Test edges: `data/splits/test_edges.csv`
- Negative samples: `data/splits/negative_samples.csv`
- Top-K predictions: `results/predictions/topk_predictions.csv`
- Metrics summary: `results/metrics/evaluation_summary.csv`

## Suggested report sections

- Problem statement
- Dataset description
- Preprocessing and split strategy
- Algorithms and assumptions
- Evaluation setup
- Results and comparison
- Limitations and future improvements

