from __future__ import annotations

import random
from typing import Iterable

import networkx as nx
import pandas as pd


def _iter_edges(extra_exclusions: pd.DataFrame | Iterable[tuple[int, int]] | None) -> Iterable[tuple[int, int]]:
    if extra_exclusions is None:
        return []
    if isinstance(extra_exclusions, pd.DataFrame):
        return extra_exclusions[["source", "target"]].itertuples(index=False, name=None)
    return extra_exclusions


def sample_negative_edges_by_source(
    graph: nx.DiGraph,
    positive_edges: pd.DataFrame,
    negatives_per_positive: int = 10,
    seed: int = 42,
    extra_exclusions: pd.DataFrame | Iterable[tuple[int, int]] | None = None,
) -> pd.DataFrame:
    if negatives_per_positive < 1:
        raise ValueError("negatives_per_positive must be at least 1.")

    rng = random.Random(seed)
    nodes = list(graph.nodes())
    excluded_targets: dict[int, set[int]] = {}

    for source, target in graph.edges():
        excluded_targets.setdefault(source, set()).add(target)

    for source, target in positive_edges[["source", "target"]].itertuples(index=False, name=None):
        excluded_targets.setdefault(source, set()).add(target)

    for source, target in _iter_edges(extra_exclusions):
        excluded_targets.setdefault(source, set()).add(target)

    rows: list[tuple[int, int]] = []
    positive_counts = positive_edges.groupby("source").size().to_dict()

    for source, count in positive_counts.items():
        blocked = set(excluded_targets.get(source, set()))
        blocked.add(source)
        available_targets = [node for node in nodes if node not in blocked]

        if not available_targets:
            continue

        sample_size = min(len(available_targets), count * negatives_per_positive)
        sampled_targets = (
            available_targets
            if sample_size == len(available_targets)
            else rng.sample(available_targets, sample_size)
        )

        for target in sampled_targets:
            rows.append((int(source), int(target)))

    negative_df = pd.DataFrame(rows, columns=["source", "target"])
    if negative_df.empty:
        return negative_df

    return negative_df.drop_duplicates().sort_values(["source", "target"]).reset_index(drop=True)


def build_candidate_set(test_edges: pd.DataFrame, negative_edges: pd.DataFrame) -> pd.DataFrame:
    positive_df = test_edges.copy()
    positive_df["label"] = 1

    negative_df = negative_edges.copy()
    negative_df["label"] = 0

    candidates = pd.concat([positive_df, negative_df], ignore_index=True)
    candidates = candidates.drop_duplicates(subset=["source", "target"], keep="first")
    return candidates.sort_values(["source", "target"]).reset_index(drop=True)

