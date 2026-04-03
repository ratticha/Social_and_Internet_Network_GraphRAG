# graphrag Package

This package contains the main logic for the project.

## Modules

- `config.py`: common project paths and default parameters
- `data_loader.py`: reads raw SNAP data and CSV edge files
- `preprocessing.py`: cleans edges and removes duplicates or self-loops
- `graph_builder.py`: builds directed and undirected NetworkX graphs
- `split.py`: creates the train/test edge split
- `candidate_sampling.py`: creates negative candidate edges
- `link_prediction.py`: computes Common Neighbors, Jaccard, Adamic/Adar, and Personalized PageRank scores
- `ranking.py`: ranks candidate edges into Top-K predictions
- `evaluation.py`: computes Precision@K summaries

## Note

The `personalized_pagerank` baseline requires `scipy`, which is now listed in `requirements.txt`.

