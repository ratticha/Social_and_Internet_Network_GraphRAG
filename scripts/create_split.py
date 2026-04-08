from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from graphrag.candidate_sampling import sample_negative_edges_by_source
from graphrag.data_loader import load_edges_csv, save_edges_csv
from graphrag.graph_builder import build_directed_graph
from graphrag.split import random_edge_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/test splits and negative samples.")
    parser.add_argument("--input", default="data/processed/graph_edges.csv", help="Processed edge CSV.")
    parser.add_argument("--train-output", default="data/splits/train_edges.csv", help="Train edges CSV.")
    parser.add_argument("--test-output", default="data/splits/test_edges.csv", help="Test edges CSV.")
    parser.add_argument(
        "--negative-output",
        default="data/splits/negative_samples.csv",
        help="Negative sampled edges CSV.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of edges held out for testing.")
    parser.add_argument(
        "--negatives-per-positive",
        type=int,
        default=10,
        help="Number of negative edges to sample per positive test edge.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    edges = load_edges_csv(args.input)
    train_edges, test_edges = random_edge_split(edges, test_size=args.test_size, seed=args.seed)

    train_graph = build_directed_graph(train_edges)
    negative_edges = sample_negative_edges_by_source(
        train_graph,
        test_edges,
        negatives_per_positive=args.negatives_per_positive,
        seed=args.seed,
    )

    save_edges_csv(train_edges, args.train_output)
    save_edges_csv(test_edges, args.test_output)
    save_edges_csv(negative_edges, args.negative_output)

    print(f"Train edges: {len(train_edges):,}")
    print(f"Test edges: {len(test_edges):,}")
    print(f"Negative samples: {len(negative_edges):,}")


if __name__ == "__main__":
    main()

