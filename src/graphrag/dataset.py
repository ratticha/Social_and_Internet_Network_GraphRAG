"""
Pluggable dataset adapters for the GraphRAG link prediction pipeline.

Every dataset adapter returns:
  - a NetworkX graph (directed or undirected)
  - a candidates DataFrame  with columns: source, target, label
  - a task tag: "link_prediction" or "context_retrieval"

Supported out of the box
-------------------------
EdgeListDataset   — any CSV / TSV with source,target columns
                    (covers wikivote and any other social network)
ContextGraphDataset — list of (graph, seed_entities, positive_entities)
                    (covers HotpotQA and any QA / knowledge-graph task)

Adding a new dataset
--------------------
Subclass GraphDataset and implement the three abstract methods.
The pipeline will handle scoring and evaluation automatically.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

import networkx as nx
import pandas as pd

from graphrag.candidate_sampling import (
    build_candidate_set,
    sample_negative_edges_by_source,
)
from graphrag.graph_builder import build_directed_graph, build_undirected_projection
from graphrag.preprocessing import clean_edges
from graphrag.split import random_edge_split


# ── Abstract base ──────────────────────────────────────────────────────────────

class GraphDataset(ABC):
    """Base class for all GraphRAG dataset adapters."""

    # Subclasses set this to "link_prediction" or "context_retrieval"
    task: str = "link_prediction"

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable dataset name."""

    @abstractmethod
    def get_graph(self) -> nx.Graph | nx.DiGraph:
        """Return the training graph used for scoring."""

    @abstractmethod
    def get_candidates(self) -> pd.DataFrame:
        """
        Return candidate pairs with columns: source, target, label.
        label=1 for positive (true) edges, label=0 for negative (non-) edges.
        """

    def summary(self) -> dict:
        """Return a dict of dataset statistics."""
        graph = self.get_graph()
        candidates = self.get_candidates()
        return {
            "dataset":    self.name,
            "task":       self.task,
            "nodes":      graph.number_of_nodes(),
            "edges":      graph.number_of_edges(),
            "directed":   graph.is_directed(),
            "candidates": len(candidates),
            "positives":  int(candidates["label"].sum()),
            "negatives":  int((candidates["label"] == 0).sum()),
        }


# ── Edge-list dataset (social networks, SNAP files, any CSV) ──────────────────

class EdgeListDataset(GraphDataset):
    """
    Works with any CSV/TSV that has source and target columns.
    Handles the full pipeline: clean → split → negative sample.

    Parameters
    ----------
    path : str or Path
        Path to the edge list file.
    directed : bool
        If True, build a DiGraph; otherwise build an undirected Graph.
    source_col, target_col : str
        Column names for the source and target node IDs.
    sep : str
        File separator.
    comment : str
        Lines beginning with this character are ignored (SNAP files use '#').
    test_size : float
        Fraction of edges to hold out for evaluation (default 0.2).
    negatives_per_positive : int
        Negative edges sampled per positive test edge (default 10).
    seed : int
        Random seed for reproducibility.
    """

    task = "link_prediction"

    def __init__(
        self,
        path: str | Path,
        directed: bool = True,
        source_col: str = "source",
        target_col: str = "target",
        sep: str = ",",
        comment: str = "#",
        test_size: float = 0.2,
        negatives_per_positive: int = 10,
        seed: int = 42,
    ) -> None:
        self._path = Path(path)
        self._directed = directed
        self._source_col = source_col
        self._target_col = target_col
        self._sep = sep
        self._comment = comment
        self._test_size = test_size
        self._negatives_per_positive = negatives_per_positive
        self._seed = seed

        self._train_graph: nx.DiGraph | None = None
        self._candidates: pd.DataFrame | None = None

    @property
    def name(self) -> str:
        return self._path.stem

    def _load_raw(self) -> pd.DataFrame:
        """Read the file regardless of format (CSV, TSV, SNAP space-separated)."""
        suffix = self._path.suffix.lower()

        if suffix in {".txt", ".tsv"} or self._comment == "#":
            rows = []
            with self._path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    stripped = line.strip()
                    if not stripped or stripped.startswith(self._comment):
                        continue
                    parts = stripped.split()
                    if len(parts) >= 2:
                        rows.append((parts[0], parts[1]))
            edges = pd.DataFrame(rows, columns=[self._source_col, self._target_col])
        else:
            edges = pd.read_csv(self._path, sep=self._sep, comment=self._comment)

        edges = edges[[self._source_col, self._target_col]].rename(
            columns={self._source_col: "source", self._target_col: "target"}
        )
        edges["source"] = pd.to_numeric(edges["source"], errors="coerce")
        edges["target"] = pd.to_numeric(edges["target"], errors="coerce")
        return edges.dropna().astype(int)

    def _build(self) -> None:
        if self._train_graph is not None:
            return

        raw = self._load_raw()
        clean = clean_edges(raw)
        train_edges, test_edges = random_edge_split(
            clean, test_size=self._test_size, seed=self._seed
        )

        if self._directed:
            self._train_graph = build_directed_graph(train_edges)
        else:
            self._train_graph = build_undirected_projection(
                build_directed_graph(train_edges)
            )

        negatives = sample_negative_edges_by_source(
            build_directed_graph(train_edges),
            test_edges,
            negatives_per_positive=self._negatives_per_positive,
            seed=self._seed,
        )
        self._candidates = build_candidate_set(test_edges, negatives)

    def get_graph(self) -> nx.DiGraph | nx.Graph:
        self._build()
        return self._train_graph  # type: ignore[return-value]

    def get_candidates(self) -> pd.DataFrame:
        self._build()
        return self._candidates  # type: ignore[return-value]


class WikiVoteDataset(EdgeListDataset):
    """
    Convenience wrapper for the SNAP Wikipedia Vote Network.
    Accepts either the raw wiki-Vote.txt file or a pre-cleaned CSV.
    """

    def __init__(
        self,
        path: str | Path = "data/processed/graph_edges.csv",
        test_size: float = 0.2,
        negatives_per_positive: int = 10,
        seed: int = 42,
    ) -> None:
        raw_path = Path(path)
        is_txt = raw_path.suffix.lower() == ".txt"
        super().__init__(
            path=raw_path,
            directed=True,
            sep="," if not is_txt else " ",
            comment="#" if is_txt else None,  # type: ignore[arg-type]
            test_size=test_size,
            negatives_per_positive=negatives_per_positive,
            seed=seed,
        )

    @property
    def name(self) -> str:
        return "Wikipedia Vote Network"


# ── Context-retrieval dataset (QA, knowledge graphs) ──────────────────────────

class ContextGraphDataset(GraphDataset):
    """
    Dataset adapter for context-retrieval tasks such as HotpotQA.

    Instead of one large graph with a train/test split, this adapter
    works with a list of independent per-example records. Each record is:

        {
          "graph":    nx.Graph  — entity co-occurrence graph for this example
          "seeds":    list[str] — query / question entities (scoring start points)
          "positives": set[str] — entities that are ground-truth relevant
        }

    This matches exactly what the AT3_SocialNetwork.ipynb notebook builds.

    Usage
    -----
    >>> records = [
    ...     {"graph": G1, "seeds": ["scott derrickson"], "positives": {"ed wood"}},
    ...     ...
    ... ]
    >>> dataset = ContextGraphDataset(records, dataset_name="HotpotQA")
    >>> pipeline = GraphRAGPipeline(dataset, methods=["ppr", "katz"])
    >>> results = pipeline.run()   # returns AUC-ROC and Average Precision

    Building records from HotpotQA
    --------------------------------
    See notebooks/03_universal_pipeline.ipynb for the full integration example.
    """

    task = "context_retrieval"

    def __init__(
        self,
        records: list[dict],
        dataset_name: str = "ContextGraph",
    ) -> None:
        if not records:
            raise ValueError("records list must not be empty.")
        self._records = records
        self._dataset_name = dataset_name
        self._combined_graph: nx.Graph | None = None
        self._candidates: pd.DataFrame | None = None

    @property
    def name(self) -> str:
        return self._dataset_name

    @property
    def records(self) -> list[dict]:
        return self._records

    def get_graph(self) -> nx.Graph:
        """Returns a combined graph merging all example graphs (for statistics only)."""
        if self._combined_graph is None:
            G = nx.Graph()
            for rec in self._records:
                G.add_edges_from(rec["graph"].edges(data=True))
            self._combined_graph = G
        return self._combined_graph

    def get_candidates(self) -> pd.DataFrame:
        """
        Builds a flat candidates DataFrame where:
          - source = seed entity (encoded as integer hash for compatibility)
          - target = candidate entity
          - label  = 1 if target is in positives else 0
          - example_id = which example this row belongs to
          - source_name / target_name = original string entity names
        """
        if self._candidates is not None:
            return self._candidates

        rows = []
        for ex_id, rec in enumerate(self._records):
            G = rec["graph"]
            seeds = [s for s in rec["seeds"] if s in G.nodes]
            positives = {p.lower().strip() for p in rec["positives"]}
            if not seeds:
                continue
            for node in G.nodes():
                if node in seeds:
                    continue
                rows.append({
                    "example_id":   ex_id,
                    "source_name":  seeds[0],
                    "target_name":  node,
                    "source":       abs(hash(f"{ex_id}_{seeds[0]}")) % (10 ** 9),
                    "target":       abs(hash(f"{ex_id}_{node}"))     % (10 ** 9),
                    "label":        1 if node in positives else 0,
                })

        self._candidates = pd.DataFrame(rows)
        return self._candidates

    def summary(self) -> dict:
        base = super().summary()
        base["examples"] = len(self._records)
        base["avg_nodes_per_example"] = round(
            sum(r["graph"].number_of_nodes() for r in self._records) / len(self._records), 1
        )
        return base
