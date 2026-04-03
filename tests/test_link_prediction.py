from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from graphrag.graph_builder import build_directed_graph, build_undirected_projection
from graphrag.link_prediction import score_candidates


def test_score_candidates_adds_requested_score_columns() -> None:
    train_edges = pd.DataFrame(
        [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4)],
        columns=["source", "target"],
    )
    candidates = pd.DataFrame(
        [(1, 4, 1), (1, 5, 0)],
        columns=["source", "target", "label"],
    )

    directed_graph = build_directed_graph(train_edges)
    undirected_graph = build_undirected_projection(directed_graph)
    scored = score_candidates(
        directed_graph,
        undirected_graph,
        candidates,
        methods=["common_neighbors", "jaccard", "adamic_adar"],
    )

    assert "common_neighbors" in scored.columns
    assert "jaccard" in scored.columns
    assert "adamic_adar" in scored.columns
    assert scored.loc[0, "common_neighbors"] > 0
    assert scored.loc[1, "jaccard"] == 0.0

