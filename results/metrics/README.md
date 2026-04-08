# Metrics

This folder contains evaluation metrics for the link prediction experiments.

## Expected files

- `evaluation_summary.csv`: one row per method with mean Precision@K
- `per_source_precision.csv`: Precision@K for each source node

## Current status

The metrics have been generated successfully.

## Final ranking

1. `personalized_pagerank`: `0.3369`
2. `common_neighbors`: `0.3029`
3. `adamic_adar`: `0.3027`
4. `jaccard`: `0.2941`

## Notes

- `evaluation_summary.csv` is the best file to cite in the report
- `per_source_precision.csv` is useful for boxplots and stability analysis
