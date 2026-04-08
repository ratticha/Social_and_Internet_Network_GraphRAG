# Results Folder

This folder stores the final outputs produced after running the experiment pipeline.

## Subfolders

- `metrics/`: evaluation summaries and per-source metric files.
- `predictions/`: scored candidate links and Top-K predictions.

## What is stored here

### `metrics/evaluation_summary.csv`

This file is the main comparison table across methods. It reports:

- method name
- `k`
- mean Precision@K
- standard deviation
- total hits
- number of evaluated source nodes

Final values from the current run:

| Method | Mean Precision@10 | Std. Dev. | Total Hits |
|---|---:|---:|---:|
| Personalized PageRank | 0.3369 | 0.2389 | 9456 |
| Common Neighbors | 0.3029 | 0.2328 | 8502 |
| Adamic/Adar | 0.3027 | 0.2320 | 8496 |
| Jaccard | 0.2941 | 0.2320 | 8254 |

### `metrics/per_source_precision.csv`

This file stores Precision@10 for each individual source node. It is useful for:

- measuring consistency across nodes
- plotting boxplots
- identifying which methods are more stable or more variable

### `predictions/scored_candidates.csv`

This file stores all candidate edges that were scored by the algorithms. It is useful for:

- debugging
- checking raw scores
- reusing saved predictions without recomputing everything

### `predictions/topk_predictions.csv`

This file stores the final Top-10 recommended links for each evaluated source node. It is the closest file to the actual retrieval output of the simplified GraphRAG system.

## Result interpretation

- `personalized_pagerank` is the best-performing method in this project.
- It outperformed the local similarity baselines on mean Precision@10.
- This suggests that global graph structure helped identify hidden relevant links better than using only local neighborhood overlap.
- The local baselines still performed reasonably well, especially `common_neighbors` and `adamic_adar`, which were very close to each other.

## Run summary

- Candidate edges scored: `228,118`
- Evaluated source nodes: `2,807`
- Experiment runtime: about `886.4` seconds
- Best method: `personalized_pagerank`
