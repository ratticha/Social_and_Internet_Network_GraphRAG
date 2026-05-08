"""
GraphRAGPipeline — flexible, dataset-agnostic scoring and evaluation.

Works with any GraphDataset subclass. Automatically selects the right
evaluation metric based on the dataset's task tag:

  task="link_prediction"   → Precision@K  (one large graph, edge split)
  task="context_retrieval" → AUC-ROC + Average Precision (per-example graphs)

Quick examples
--------------
# ── Social network (edge list) ──────────────────────────────
from graphrag.dataset import WikiVoteDataset
from graphrag.pipeline import GraphRAGPipeline

dataset  = WikiVoteDataset("data/processed/graph_edges.csv")
pipeline = GraphRAGPipeline(dataset, methods=["ppr", "jaccard", "katz"], k=10)
results  = pipeline.run()
print(results)

# ── HotpotQA / knowledge graph ──────────────────────────────
from graphrag.dataset import ContextGraphDataset
from graphrag.pipeline import GraphRAGPipeline

records  = [{"graph": G, "seeds": [...], "positives": {...}}, ...]
dataset  = ContextGraphDataset(records, dataset_name="HotpotQA")
pipeline = GraphRAGPipeline(dataset, methods=["ppr", "katz"])
results  = pipeline.run()
print(results)
"""
from __future__ import annotations

from typing import Sequence

import networkx as nx
import numpy as np
import pandas as pd

from graphrag.dataset import ContextGraphDataset, GraphDataset
from graphrag.evaluation import evaluate_rankings
from graphrag.graph_builder import build_directed_graph, build_undirected_projection
from graphrag.link_prediction import (
    SUPPORTED_METHODS,
    adamic_adar_score,
    common_neighbors_score,
    jaccard_score,
    katz_score,
    score_candidates,
)
from graphrag.ranking import rank_all_methods

# Friendly aliases so callers can write "ppr" instead of "personalized_pagerank"
_ALIASES: dict[str, str] = {
    "ppr":               "personalized_pagerank",
    "pagerank":          "personalized_pagerank",
    "cn":                "common_neighbors",
    "aa":                "adamic_adar",
    "adamic":            "adamic_adar",
}


def _resolve_methods(methods: Sequence[str]) -> list[str]:
    resolved = [_ALIASES.get(m, m) for m in methods]
    unknown = [m for m in resolved if m not in SUPPORTED_METHODS]
    if unknown:
        raise ValueError(
            f"Unknown methods: {unknown}. "
            f"Supported: {list(SUPPORTED_METHODS)} "
            f"or aliases: {list(_ALIASES)}"
        )
    return resolved


# ── Context-retrieval scorer (per-example graphs) ─────────────────────────────

def _score_context_dataset(
    dataset: ContextGraphDataset,
    methods: list[str],
    verbose: bool,
) -> pd.DataFrame:
    """
    Score each example independently using its own per-example graph.
    Returns a flat DataFrame with columns:
      example_id, source_name, target_name, label, <method1>, <method2>, ...
    """
    rows: list[dict] = []
    total = len(dataset.records)

    for ex_id, rec in enumerate(dataset.records):
        G = rec["graph"]
        seeds = [s for s in rec["seeds"] if s in G.nodes]
        positives = {p.lower().strip() for p in rec["positives"]}

        if not seeds or G.number_of_nodes() < 3:
            continue

        # Build directed and undirected views of the per-example graph
        directed = G.to_directed() if not G.is_directed() else G
        undirected = G.to_undirected() if G.is_directed() else G

        candidates_here = [
            n for n in G.nodes() if n not in seeds
        ]

        if not candidates_here:
            continue

        for target in candidates_here:
            row: dict = {
                "example_id":  ex_id,
                "source_name": seeds[0],
                "target_name": str(target),
                "label":       1 if str(target) in positives else 0,
            }

            if "common_neighbors" in methods:
                row["common_neighbors"] = common_neighbors_score(undirected, seeds[0], target)
            if "jaccard" in methods:
                row["jaccard"] = jaccard_score(undirected, seeds[0], target)
            if "adamic_adar" in methods:
                row["adamic_adar"] = adamic_adar_score(undirected, seeds[0], target)
            if "katz" in methods:
                row["katz"] = katz_score(undirected, seeds[0], target)

            rows.append(row)

        if "personalized_pagerank" in methods and seeds:
            personalization = {n: 0.0 for n in G.nodes()}
            for s in seeds:
                if s in personalization:
                    personalization[s] = 1.0 / len(seeds)
            try:
                ppr = nx.pagerank(undirected, alpha=0.85, personalization=personalization)
            except Exception:
                ppr = {}
            for row in rows[-len(candidates_here):]:
                tgt = row["target_name"]
                row["personalized_pagerank"] = ppr.get(tgt, 0.0)

        if verbose and (ex_id + 1) % 50 == 0:
            print(f"  Scored {ex_id + 1}/{total} examples", flush=True)

    return pd.DataFrame(rows)


# ── AUC-ROC / Average Precision evaluation ───────────────────────────────────

def _evaluate_context(scored_df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """Compute AUC-ROC and Average Precision per method for context-retrieval tasks."""
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required for context-retrieval evaluation. "
            "Install it with: pip install scikit-learn"
        ) from e

    labels = scored_df["label"].values
    if labels.sum() == 0:
        raise ValueError("No positive examples found. Check your positives/seeds.")

    results = []
    for m in methods:
        if m not in scored_df.columns:
            continue
        scores = scored_df[m].fillna(0.0).values
        results.append({
            "method":            m,
            "auc_roc":           round(roc_auc_score(labels, scores), 4),
            "average_precision": round(average_precision_score(labels, scores), 4),
            "positives":         int(labels.sum()),
            "total_candidates":  len(labels),
        })

    return pd.DataFrame(results).sort_values("auc_roc", ascending=False).reset_index(drop=True)


# ── Main pipeline class ────────────────────────────────────────────────────────

class GraphRAGPipeline:
    """
    Flexible GraphRAG scoring and evaluation pipeline.

    Parameters
    ----------
    dataset : GraphDataset
        Any dataset adapter (EdgeListDataset, WikiVoteDataset,
        ContextGraphDataset, or your own subclass of GraphDataset).
    methods : list[str], optional
        Scoring methods to run. Defaults to all supported methods.
        Accepts full names or short aliases:
          "ppr" / "personalized_pagerank"
          "cn"  / "common_neighbors"
          "aa"  / "adamic_adar"
          "jaccard"
          "katz"
    k : int
        Top-K for Precision@K evaluation (link_prediction tasks only).
    pagerank_alpha : float
        Damping factor for Personalized PageRank.
    verbose : bool
        Print progress messages.
    """

    def __init__(
        self,
        dataset: GraphDataset,
        methods: Sequence[str] | None = None,
        k: int = 10,
        pagerank_alpha: float = 0.85,
        verbose: bool = True,
    ) -> None:
        self.dataset = dataset
        self.methods = _resolve_methods(
            methods if methods is not None else list(SUPPORTED_METHODS)
        )
        self.k = k
        self.pagerank_alpha = pagerank_alpha
        self.verbose = verbose

        self._scored_df: pd.DataFrame | None = None
        self._results: pd.DataFrame | None = None

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def run(self) -> pd.DataFrame:
        """
        Score all candidates and evaluate.

        Returns
        -------
        pd.DataFrame
            For link_prediction tasks:  columns = method, k, mean_precision_at_k, ...
            For context_retrieval tasks: columns = method, auc_roc, average_precision, ...
        """
        self._log(f"\n{'='*60}")
        self._log(f"Dataset : {self.dataset.name}")
        self._log(f"Task    : {self.dataset.task}")
        self._log(f"Methods : {', '.join(self.methods)}")
        self._log(f"{'='*60}\n")

        stats = self.dataset.summary()
        for key, val in stats.items():
            self._log(f"  {key:30s}: {val}")
        self._log("")

        if self.dataset.task == "context_retrieval":
            self._results = self._run_context()
        else:
            self._results = self._run_link_prediction()

        return self._results

    def _run_link_prediction(self) -> pd.DataFrame:
        self._log("Building graphs and scoring candidates...")
        graph      = self.dataset.get_graph()
        candidates = self.dataset.get_candidates()

        directed   = graph if graph.is_directed() else nx.DiGraph(graph)
        undirected = build_undirected_projection(directed)

        self._scored_df = score_candidates(
            directed_graph=directed,
            undirected_graph=undirected,
            candidates=candidates,
            methods=self.methods,
            pagerank_alpha=self.pagerank_alpha,
            verbose=self.verbose,
        )

        self._log(f"\nRanking top-{self.k} predictions per source node...")
        ranked = rank_all_methods(self._scored_df, methods=self.methods, k=self.k)

        self._log("Evaluating Precision@K...")
        summary_df, _ = evaluate_rankings(ranked, k=self.k)

        self._log("\n" + summary_df.to_string(index=False))
        return summary_df

    def _run_context(self) -> pd.DataFrame:
        assert isinstance(self.dataset, ContextGraphDataset)
        self._log("Scoring per-example context graphs...")
        self._scored_df = _score_context_dataset(
            self.dataset, self.methods, self.verbose
        )
        self._log("\nEvaluating AUC-ROC and Average Precision...")
        result = _evaluate_context(self._scored_df, self.methods)
        self._log("\n" + result.to_string(index=False))
        return result

    @property
    def scored_candidates(self) -> pd.DataFrame | None:
        """Raw scores DataFrame (available after run())."""
        return self._scored_df

    @property
    def results(self) -> pd.DataFrame | None:
        """Evaluation results (available after run())."""
        return self._results
