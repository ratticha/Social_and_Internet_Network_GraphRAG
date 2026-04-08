# GraphRAG Link Prediction Project

This repository implements a simplified GraphRAG retrieval module for the Wikipedia Vote Network assignment. The main idea is to treat retrieval as a link prediction problem: if user `A` is likely to connect to user `B`, then `B` can be treated as useful context for `A`.

## Project logic

The pipeline follows this sequence:

1. Download the raw Wikipedia Vote Network from SNAP.
2. Clean the raw text file into a CSV edge list.
3. Split the graph into training edges and hidden testing edges.
4. Sample negative edges for evaluation.
5. Score candidate links with multiple graph-based methods.
6. Rank the Top-K predicted links for each source node.
7. Measure retrieval quality with Precision@10.
8. Analyze the outputs in notebooks and generate report figures.

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

## Folder objectives

- `data/`: stores the dataset and all intermediate files.
- `data/raw/`: original files downloaded from SNAP.
- `data/processed/`: cleaned graph edge list.
- `data/splits/`: training edges, testing edges, and negative samples.
- `scripts/`: runnable pipeline scripts.
- `src/graphrag/`: main source code for preprocessing, scoring, ranking, and evaluation.
- `results/`: experiment outputs.
- `results/metrics/`: evaluation summaries and per-source Precision@K files.
- `results/predictions/`: scored candidates and Top-K predictions.
- `notebooks/`: exploratory analysis and result interpretation notebooks.
- `reports/`: report draft and exported figures.
- `tests/`: small tests for the main pipeline logic.

## Implemented methods

- `common_neighbors`
- `jaccard`
- `adamic_adar`
- `personalized_pagerank`

The first three methods use local neighborhood structure on an undirected projection of the training graph. `personalized_pagerank` uses the directed training graph and captures more global graph structure.

## Environment setup

This project was run with Python `3.11.4`.

On Windows, a short virtual-environment path helped avoid long-path installation errors:

```bash
C:\Users\User\.pyenv\pyenv-win\versions\3.11.4\python.exe -m venv C:\venvs\graphrag
& "C:\venvs\graphrag\Scripts\Activate.ps1"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Step-by-step run

### 1. Download the dataset

Command:

```bash
python scripts/download_data.py
```

Observed terminal output:

```text
Downloading dataset from https://snap.stanford.edu/data/wiki-Vote.txt.gz
Extracting wiki-Vote.txt.gz to data\raw\wiki-Vote.txt
Download complete.
```

Generated files:

- `data/raw/wiki-Vote.txt.gz`
- `data/raw/wiki-Vote.txt`

### 2. Preprocess the raw data

Command:

```bash
python scripts/preprocess_data.py
```

Observed terminal output:

```text
Saved 103,689 cleaned edges to data/processed/graph_edges.csv
```

Generated file:

- `data/processed/graph_edges.csv`

Meaning:

- The raw SNAP text file was converted into a clean directed edge list.
- Each row represents a vote from one Wikipedia user to another.

### 3. Create the train/test split

Command:

```bash
python scripts/create_split.py --test-size 0.2 --negatives-per-positive 10 --seed 42
```

Observed terminal output:

```text
Train edges: 82,951
Test edges: 20,738
Negative samples: 207,380
```

Generated files:

- `data/splits/train_edges.csv`
- `data/splits/test_edges.csv`
- `data/splits/negative_samples.csv`

Meaning:

- `train_edges.csv` is the visible graph used to build the model.
- `test_edges.csv` contains hidden true edges used for evaluation.
- `negative_samples.csv` contains non-existent edges sampled as negative cases.

### 4. Run the experiments

Command:

```bash
python scripts/run_experiments.py --top-k 10
```

Observed terminal output:

```text
Loading input files...
Loaded train=82,951, test=20,738, negative=207,380 edges.
Building candidate set and graphs...
Candidates=228,118, nodes=7,115, edges=82,951.
Methods selected: common_neighbors, jaccard, adamic_adar, personalized_pagerank
Starting scoring stage...
Scoring candidates with common_neighbors...
Finished common_neighbors.
Scoring candidates with jaccard...
Finished jaccard.
Scoring candidates with adamic_adar...
Finished adamic_adar.
Scoring candidates with personalized_pagerank...
Starting personalized PageRank for 2,807 source nodes...
...
Finished personalized_pagerank.
Ranking Top-K predictions...
Evaluating Precision@K...
Finished experiment pipeline in 886.4 seconds.
```

Generated files:

- `results/metrics/evaluation_summary.csv`
- `results/metrics/per_source_precision.csv`
- `results/predictions/scored_candidates.csv`
- `results/predictions/topk_predictions.csv`

Meaning:

- `scored_candidates.csv` stores every candidate edge and its score.
- `topk_predictions.csv` stores the Top-10 ranked predictions for each source node.
- `evaluation_summary.csv` stores the overall method comparison.
- `per_source_precision.csv` stores Precision@10 for each evaluated source node.

## Final dataset summary

- Cleaned directed edges: `103,689`
- Training edges: `82,951`
- Testing edges: `20,738`
- Negative sampled edges: `207,380`
- Candidate edges scored: `228,118`
- Source nodes evaluated: `2,807`

## Final experiment results

`Precision@10` on the held-out test edges:

| Method | Mean Precision@10 | Std. Dev. | Total Hits |
|---|---:|---:|---:|
| Personalized PageRank | 0.3369 | 0.2389 | 9456 |
| Common Neighbors | 0.3029 | 0.2328 | 8502 |
| Adamic/Adar | 0.3027 | 0.2320 | 8496 |
| Jaccard | 0.2941 | 0.2320 | 8254 |

## Result interpretation

- `personalized_pagerank` achieved the best Precision@10.
- This suggests that global graph structure helped retrieve relevant hidden links better than purely local similarity methods.
- `common_neighbors` and `adamic_adar` performed very similarly, showing that local overlap between neighborhoods is still useful.
- `jaccard` performed slightly worse, which suggests that normalization by union size reduced some useful signal on this graph.
- Overall, the results support the GraphRAG idea that graph-based reasoning can recover relevant context beyond simple keyword-style similarity.

## Notebook outputs

Two notebooks were completed:

- `notebooks/01_data_exploration.ipynb`
  - explores graph statistics
  - plots in-degree and out-degree distributions
  - checks the processed data and split outputs
- `notebooks/02_results_analysis.ipynb`
  - compares Precision@10 across methods
  - analyzes per-source performance
  - generates figures for the report

Generated report figures:

- `reports/figures/degree_distributions.png`
- `reports/figures/precision_at_k_comparison.png`
- `reports/figures/per_source_precision_boxplot.png`
- `reports/figures/positive_rate_by_rank.png`

## Main output files

- Processed edges: `data/processed/graph_edges.csv`
- Train edges: `data/splits/train_edges.csv`
- Test edges: `data/splits/test_edges.csv`
- Negative samples: `data/splits/negative_samples.csv`
- Metrics summary: `results/metrics/evaluation_summary.csv`
- Per-source metrics: `results/metrics/per_source_precision.csv`
- Scored candidates: `results/predictions/scored_candidates.csv`
- Top-K predictions: `results/predictions/topk_predictions.csv`

## Notes

- The `personalized_pagerank` method requires `scipy`, which is already listed in `requirements.txt`.
- The full experiment run took about `886.4` seconds, or around `14.8` minutes, on this setup.
- If you rerun the scripts locally, the generated CSV files in `data/` and `results/` should be updated with the new outputs.
