from __future__ import annotations

import networkx as nx
import pandas as pd

from graphrag.preprocessing import clean_edges


def build_directed_graph(edges: pd.DataFrame) -> nx.DiGraph:
    clean_df = clean_edges(edges)
    graph = nx.DiGraph()
    graph.add_edges_from(clean_df.itertuples(index=False, name=None))
    return graph


def build_undirected_projection(graph_or_edges: nx.DiGraph | pd.DataFrame) -> nx.Graph:
    if isinstance(graph_or_edges, pd.DataFrame):
        return build_directed_graph(graph_or_edges).to_undirected()
    return graph_or_edges.to_undirected()


def graph_statistics(graph: nx.Graph) -> dict[str, float]:
    return {
        "nodes": float(graph.number_of_nodes()),
        "edges": float(graph.number_of_edges()),
        "density": float(nx.density(graph)),
    }

