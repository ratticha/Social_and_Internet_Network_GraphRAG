# Results Folder

This folder stores experiment outputs after scoring and evaluation.

## Subfolders

- `metrics/`: summary statistics such as Precision@K
- `predictions/`: scored candidates and Top-K link predictions

## Current status

The experiment step is complete.

## Final summary

- Candidate edges scored: `228,118`
- Methods evaluated: `common_neighbors`, `jaccard`, `adamic_adar`, `personalized_pagerank`
- Best method: `personalized_pagerank`
- Best mean Precision@10: `0.3369`
- Experiment runtime: about `886.4` seconds, or around `14.8` minutes

## Main files

- `metrics/evaluation_summary.csv`
- `metrics/per_source_precision.csv`
- `predictions/scored_candidates.csv`
- `predictions/topk_predictions.csv`
