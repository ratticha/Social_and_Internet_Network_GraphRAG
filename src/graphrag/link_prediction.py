from __future__ import annotations

import math

import networkx as nx
import pandas as pd


SUPPORTED_METHODS = (
    "common_neighbors",
    "jaccard",
    "adamic_adar",
    "personalized_pagerank",
)


def _neighbors(graph: nx.Graph, node: int) -> set[int]:
    if node not in graph:
        return set()
    return set(graph.neighbors(node))


def common_neighbors_score(graph: nx.Graph, source: int, target: int) -> float:
    return float(len(_neighbors(graph, source) & _neighbors(graph, target)))


def jaccard_score(graph: nx.Graph, source: int, target: int) -> float:
    source_neighbors = _neighbors(graph, source)
    target_neighbors = _neighbors(graph, target)
    union_size = len(source_neighbors | target_neighbors)
    if union_size == 0:
        return 0.0
    return float(len(source_neighbors & target_neighbors) / union_size)


def adamic_adar_score(graph: nx.Graph, source: int, target: int) -> float:
    shared_neighbors = _neighbors(graph, source) & _neighbors(graph, target)
    score = 0.0

    for neighbor in shared_neighbors:
        degree = graph.degree(neighbor)
        if degree > 1:
            score += 1.0 / math.log(degree)

    return float(score)


def _compute_personalized_pagerank(
    graph: nx.DiGraph,
    sources: list[int],
    alpha: float = 0.85,
) -> dict[int, dict[int, float]]:
    if graph.number_of_nodes() == 0:
        return {}

    nodes = list(graph.nodes())
    scores_by_source: dict[int, dict[int, float]] = {}

    for source in sources:
        if source not in graph:
            scores_by_source[source] = {}
            continue

        personalization = {node: 0.0 for node in nodes}
        personalization[source] = 1.0

        scores_by_source[source] = nx.pagerank(
            graph,
            alpha=alpha,
            personalization=personalization,
        )

    return scores_by_source


def score_candidates(
    directed_graph: nx.DiGraph,
    undirected_graph: nx.Graph,
    candidates: pd.DataFrame,
    methods: list[str] | tuple[str, ...] = SUPPORTED_METHODS,
    pagerank_alpha: float = 0.85,
) -> pd.DataFrame:
    unknown_methods = set(methods) - set(SUPPORTED_METHODS)
    if unknown_methods:
        raise ValueError(f"Unsupported methods requested: {sorted(unknown_methods)}")

    scored = candidates.copy()
    pairs = list(scored[["source", "target"]].itertuples(index=False, name=None))

    if "common_neighbors" in methods:
        scored["common_neighbors"] = [
            common_neighbors_score(undirected_graph, source, target)
            for source, target in pairs
        ]

    if "jaccard" in methods:
        scored["jaccard"] = [
            jaccard_score(undirected_graph, source, target)
            for source, target in pairs
        ]

    if "adamic_adar" in methods:
        scored["adamic_adar"] = [
            adamic_adar_score(undirected_graph, source, target)
            for source, target in pairs
        ]

    if "personalized_pagerank" in methods:
        pr_scores = _compute_personalized_pagerank(
            directed_graph,
            sources=sorted(scored["source"].unique().tolist()),
            alpha=pagerank_alpha,
        )
        scored["personalized_pagerank"] = [
            pr_scores.get(source, {}).get(target, 0.0)
            for source, target in pairs
        ]

    return scored

