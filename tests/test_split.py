from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from graphrag.candidate_sampling import sample_negative_edges_by_source
from graphrag.graph_builder import build_directed_graph
from graphrag.split import random_edge_split


def test_random_edge_split_has_no_overlap() -> None:
    edges = pd.DataFrame(
        [
            (1, 2),
            (1, 3),
            (2, 3),
            (2, 4),
            (3, 4),
            (4, 5),
            (5, 1),
        ],
        columns=["source", "target"],
    )

    train_edges, test_edges = random_edge_split(edges, test_size=0.3, seed=7)
    train_set = set(train_edges.itertuples(index=False, name=None))
    test_set = set(test_edges.itertuples(index=False, name=None))

    assert train_set
    assert test_set
    assert train_set.isdisjoint(test_set)


def test_negative_sampling_avoids_known_edges() -> None:
    train_edges = pd.DataFrame(
        [(1, 2), (1, 3), (2, 3), (3, 4), (4, 5)],
        columns=["source", "target"],
    )
    test_edges = pd.DataFrame(
        [(1, 4), (2, 5)],
        columns=["source", "target"],
    )

    graph = build_directed_graph(train_edges)
    negative_edges = sample_negative_edges_by_source(graph, test_edges, negatives_per_positive=2, seed=1)

    train_set = set(train_edges.itertuples(index=False, name=None))
    test_set = set(test_edges.itertuples(index=False, name=None))
    negative_set = set(negative_edges.itertuples(index=False, name=None))

    assert negative_set
    assert negative_set.isdisjoint(train_set)
    assert negative_set.isdisjoint(test_set)

