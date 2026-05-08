from __future__ import annotations

import math
import time

import networkx as nx
import numpy as np
import pandas as pd


SUPPORTED_METHODS = (
    "common_neighbors",
    "jaccard",
    "adamic_adar",
    "personalized_pagerank",
    "katz",
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
    verbose: bool = False,
) -> dict[int, dict[int, float]]:
    if graph.number_of_nodes() == 0:
        return {}

    nodes = list(graph.nodes())
    scores_by_source: dict[int, dict[int, float]] = {}
    total_sources = len(sources)
    start_time = time.perf_counter()

    if verbose:
        print(
            f"Starting personalized PageRank for {total_sources:,} source nodes...",
            flush=True,
        )

    for index, source in enumerate(sources, start=1):
        if source not in graph:
            scores_by_source[source] = {}
            continue

        personalization = {node: 0.0 for node in nodes}
        personalization[source] = 1.0

        try:
            scores_by_source[source] = nx.pagerank(
                graph,
                alpha=alpha,
                personalization=personalization,
            )
        except ModuleNotFoundError as exc:
            if exc.name == "scipy":
                raise ModuleNotFoundError(
                    "personalized_pagerank requires scipy. "
                    "Install it with 'python -m pip install scipy' or rerun "
                    "'python -m pip install -r requirements.txt'."
                ) from exc
            raise

        if verbose and (index == 1 or index % 100 == 0 or index == total_sources):
            elapsed = time.perf_counter() - start_time
            print(
                f"  Personalized PageRank progress: {index:,}/{total_sources:,} "
                f"sources in {elapsed:.1f}s",
                flush=True,
            )

    return scores_by_source


def katz_score(
    graph: nx.Graph,
    source: int | str,
    target: int | str,
    beta: float = 0.05,
    max_path: int = 3,
) -> float:
    """
    Katz index between source and target.

    Sums contribution of all walks of length 1..max_path, each decayed by
    beta^length. Uses repeated sparse matrix-vector multiplication so it stays
    tractable for large graphs (only one source vector needed per source node).

    beta controls how fast longer paths are discounted.  beta=0.05 means a
    length-2 path contributes 0.05^2 = 0.0025 times a length-1 path.
    """
    if source not in graph or target not in graph:
        return 0.0

    nodes = list(graph.nodes())
    idx = {n: i for i, n in enumerate(nodes)}

    A = nx.to_numpy_array(graph, nodelist=nodes, weight=None).astype(float)
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    A = A / row_sums

    vec = np.zeros(len(nodes))
    vec[idx[source]] = 1.0

    score = 0.0
    current = vec.copy()
    for _ in range(max_path):
        current = beta * (A.T @ current)
        t_idx = idx.get(target)
        if t_idx is not None:
            score += current[t_idx]

    return float(score)


def score_candidates(
    directed_graph: nx.DiGraph,
    undirected_graph: nx.Graph,
    candidates: pd.DataFrame,
    methods: list[str] | tuple[str, ...] = SUPPORTED_METHODS,
    pagerank_alpha: float = 0.85,
    verbose: bool = False,
) -> pd.DataFrame:
    unknown_methods = set(methods) - set(SUPPORTED_METHODS)
    if unknown_methods:
        raise ValueError(f"Unsupported methods requested: {sorted(unknown_methods)}")

    scored = candidates.copy()
    pairs = list(scored[["source", "target"]].itertuples(index=False, name=None))

    def log(message: str) -> None:
        if verbose:
            print(message, flush=True)

    if "common_neighbors" in methods:
        log("Scoring candidates with common_neighbors...")
        scored["common_neighbors"] = [
            common_neighbors_score(undirected_graph, source, target)
            for source, target in pairs
        ]
        log("Finished common_neighbors.")

    if "jaccard" in methods:
        log("Scoring candidates with jaccard...")
        scored["jaccard"] = [
            jaccard_score(undirected_graph, source, target)
            for source, target in pairs
        ]
        log("Finished jaccard.")

    if "adamic_adar" in methods:
        log("Scoring candidates with adamic_adar...")
        scored["adamic_adar"] = [
            adamic_adar_score(undirected_graph, source, target)
            for source, target in pairs
        ]
        log("Finished adamic_adar.")

    if "personalized_pagerank" in methods:
        log("Scoring candidates with personalized_pagerank...")
        pr_scores = _compute_personalized_pagerank(
            directed_graph,
            sources=sorted(scored["source"].unique().tolist()),
            alpha=pagerank_alpha,
            verbose=verbose,
        )
        scored["personalized_pagerank"] = [
            pr_scores.get(source, {}).get(target, 0.0)
            for source, target in pairs
        ]
        log("Finished personalized_pagerank.")

    if "katz" in methods:
        log("Scoring candidates with katz...")
        scored["katz"] = [
            katz_score(undirected_graph, source, target)
            for source, target in pairs
        ]
        log("Finished katz.")

    return scored

