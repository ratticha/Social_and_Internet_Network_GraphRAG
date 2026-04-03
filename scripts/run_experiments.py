from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from graphrag.candidate_sampling import build_candidate_set
from graphrag.data_loader import load_edges_csv
from graphrag.evaluation import evaluate_rankings
from graphrag.graph_builder import build_directed_graph, build_undirected_projection
from graphrag.link_prediction import SUPPORTED_METHODS, score_candidates
from graphrag.ranking import rank_all_methods


def parse_methods(method_string: str) -> list[str]:
    methods = [item.strip() for item in method_string.split(",") if item.strip()]
    if not methods:
        raise ValueError("At least one scoring method is required.")
    return methods


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GraphRAG link prediction baselines.")
    parser.add_argument("--train", default="data/splits/train_edges.csv", help="Train edges CSV.")
    parser.add_argument("--test", default="data/splits/test_edges.csv", help="Test edges CSV.")
    parser.add_argument("--negative", default="data/splits/negative_samples.csv", help="Negative edges CSV.")
    parser.add_argument(
        "--methods",
        default=",".join(SUPPORTED_METHODS),
        help="Comma-separated list of methods to run.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="K value used in Precision@K.")
    parser.add_argument(
        "--scores-output",
        default="results/predictions/scored_candidates.csv",
        help="Output CSV for scored candidate edges.",
    )
    parser.add_argument(
        "--predictions-output",
        default="results/predictions/topk_predictions.csv",
        help="Output CSV for ranked Top-K predictions.",
    )
    parser.add_argument(
        "--metrics-output",
        default="results/metrics/evaluation_summary.csv",
        help="Output CSV for aggregated evaluation metrics.",
    )
    parser.add_argument(
        "--per-source-output",
        default="results/metrics/per_source_precision.csv",
        help="Output CSV for per-source Precision@K values.",
    )
    args = parser.parse_args()

    methods = parse_methods(args.methods)
    train_edges = load_edges_csv(args.train)
    test_edges = load_edges_csv(args.test)
    negative_edges = load_edges_csv(args.negative)

    candidates = build_candidate_set(test_edges, negative_edges)
    directed_graph = build_directed_graph(train_edges)
    undirected_graph = build_undirected_projection(directed_graph)

    scored_candidates = score_candidates(
        directed_graph=directed_graph,
        undirected_graph=undirected_graph,
        candidates=candidates,
        methods=methods,
    )

    ranked_predictions = rank_all_methods(scored_candidates, methods=methods, k=args.top_k)
    summary_df, per_source_df = evaluate_rankings(ranked_predictions, k=args.top_k)

    Path(args.scores_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.predictions_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.per_source_output).parent.mkdir(parents=True, exist_ok=True)

    scored_candidates.to_csv(args.scores_output, index=False)
    ranked_predictions.to_csv(args.predictions_output, index=False)
    summary_df.to_csv(args.metrics_output, index=False)
    per_source_df.to_csv(args.per_source_output, index=False)

    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

