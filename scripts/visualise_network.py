"""
Generate a focused subgraph visualisation of the Wikipedia Vote Network.

Shows the top hub nodes (highest in-degree) and their immediate neighbours,
illustrating the hub-and-spoke structure discussed in the report.

Output: reports/figures/network_subgraph.png
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

EDGES_PATH   = PROJECT_ROOT / "data" / "processed" / "graph_edges.csv"
FIGURES_DIR  = PROJECT_ROOT / "reports" / "figures"
OUTPUT_PATH  = FIGURES_DIR / "network_subgraph.png"

TOP_HUBS        = 6      # number of hub nodes to highlight
NEIGHBOURS_PER_HUB = 8  # max neighbours shown per hub
RANDOM_SEED     = 42


def build_subgraph(
    full_graph: nx.DiGraph,
    top_hubs: int,
    neighbours_per_hub: int,
    seed: int,
) -> tuple[nx.DiGraph, list[int], list[int]]:
    """Return (subgraph, hub_nodes, ordinary_nodes)."""
    rng = random.Random(seed)

    # Pick top hubs by in-degree (most-voted-for users)
    in_deg = sorted(full_graph.in_degree(), key=lambda x: x[1], reverse=True)
    hub_nodes = [node for node, _ in in_deg[:top_hubs]]

    # Collect a sample of their predecessors (voters) and successors (votees)
    selected = set(hub_nodes)
    for hub in hub_nodes:
        preds = list(full_graph.predecessors(hub))
        succs = list(full_graph.successors(hub))
        selected.update(rng.sample(preds, min(neighbours_per_hub, len(preds))))
        selected.update(rng.sample(succs, min(neighbours_per_hub // 2, len(succs))))

    sub = full_graph.subgraph(selected).copy()
    ordinary = [n for n in sub.nodes() if n not in hub_nodes]
    return sub, hub_nodes, ordinary


def draw(
    sub: nx.DiGraph,
    hub_nodes: list[int],
    ordinary_nodes: list[int],
    full_graph: nx.DiGraph,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("#f8fafc")

    # Layout: spring layout seeded for reproducibility
    pos = nx.spring_layout(sub, seed=RANDOM_SEED, k=2.2, iterations=60)

    hub_set = set(hub_nodes)

    # ── Node sizes and colours ──────────────────────────────────────────────
    in_deg_full = dict(full_graph.in_degree())

    node_sizes  = []
    node_colors = []
    for node in sub.nodes():
        deg = in_deg_full.get(node, 0)
        if node in hub_set:
            node_sizes.append(1800 + deg * 3)
            node_colors.append("#1F3864")   # dark navy — hub
        else:
            node_sizes.append(280 + deg * 8)
            node_colors.append("#5B9BD5")   # medium blue — ordinary

    # ── Edges ───────────────────────────────────────────────────────────────
    hub_edges = [(u, v) for u, v in sub.edges() if u in hub_set or v in hub_set]
    other_edges = [(u, v) for u, v in sub.edges() if (u, v) not in hub_edges]

    nx.draw_networkx_edges(
        sub, pos, edgelist=other_edges, ax=ax,
        edge_color="#94a3b8", alpha=0.45, width=0.8,
        arrows=True, arrowsize=10,
        connectionstyle="arc3,rad=0.08",
    )
    nx.draw_networkx_edges(
        sub, pos, edgelist=hub_edges, ax=ax,
        edge_color="#1F3864", alpha=0.65, width=1.4,
        arrows=True, arrowsize=14,
        connectionstyle="arc3,rad=0.08",
    )

    # ── Nodes ───────────────────────────────────────────────────────────────
    nx.draw_networkx_nodes(
        sub, pos, ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        linewidths=1.2,
        edgecolors="#ffffff",
    )

    # ── Labels (only hub nodes) ─────────────────────────────────────────────
    hub_labels = {
        n: f"User {n}\n({in_deg_full.get(n, 0)} votes)"
        for n in hub_nodes if n in sub
    }
    nx.draw_networkx_labels(
        sub, pos, labels=hub_labels, ax=ax,
        font_size=7.5, font_color="white", font_weight="bold",
    )

    # ── Legend ──────────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor="#1F3864", edgecolor="white",
                       label=f"Top-{len(hub_nodes)} hub nodes (highest in-degree)"),
        mpatches.Patch(facecolor="#5B9BD5", edgecolor="white",
                       label="Ordinary users (voters / votees)"),
    ]
    ax.legend(
        handles=legend_elements, loc="lower left",
        fontsize=9, framealpha=0.85,
        facecolor="white", edgecolor="#e2e8f0",
    )

    # ── Titles and stats ────────────────────────────────────────────────────
    ax.set_title(
        "Wikipedia Vote Network — Hub-and-Spoke Subgraph\n"
        f"Showing top-{len(hub_nodes)} hub nodes and their immediate neighbours  "
        f"({sub.number_of_nodes()} nodes, {sub.number_of_edges()} edges shown  |  "
        f"Full graph: {full_graph.number_of_nodes():,} nodes, "
        f"{full_graph.number_of_edges():,} edges)",
        fontsize=10, pad=14,
    )
    ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    print("Loading edge list...")
    edges = pd.read_csv(EDGES_PATH)
    full_graph = nx.from_pandas_edgelist(
        edges, source="source", target="target",
        create_using=nx.DiGraph(),
    )
    print(f"Full graph: {full_graph.number_of_nodes():,} nodes, "
          f"{full_graph.number_of_edges():,} edges")

    print(f"Building subgraph (top {TOP_HUBS} hubs, "
          f"up to {NEIGHBOURS_PER_HUB} neighbours each)...")
    sub, hub_nodes, ordinary_nodes = build_subgraph(
        full_graph, TOP_HUBS, NEIGHBOURS_PER_HUB, RANDOM_SEED,
    )
    print(f"Subgraph: {sub.number_of_nodes()} nodes, {sub.number_of_edges()} edges")

    print("Rendering visualisation...")
    draw(sub, hub_nodes, ordinary_nodes, full_graph, OUTPUT_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
