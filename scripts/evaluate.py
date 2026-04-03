from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from graphrag.evaluation import evaluate_rankings
from graphrag.link_prediction import SUPPORTED_METHODS
from graphrag.ranking import rank_all_methods


def infer_methods(scores_df: pd.DataFrame) -> list[str]:
    return [column for column in SUPPORTED_METHODS if column in scores_df.columns]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved link prediction scores.")
    parser.add_argument(
        "--scores",
        default="results/predictions/scored_candidates.csv",
        help="CSV file containing candidate scores.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="K value used in Precision@K.")
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

    scores_df = pd.read_csv(args.scores)
    methods = infer_methods(scores_df)
    if not methods:
        raise ValueError("No supported score columns were found in the scores file.")

    ranked_predictions = rank_all_methods(scores_df, methods=methods, k=args.top_k)
    summary_df, per_source_df = evaluate_rankings(ranked_predictions, k=args.top_k)

    Path(args.predictions_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.per_source_output).parent.mkdir(parents=True, exist_ok=True)

    ranked_predictions.to_csv(args.predictions_output, index=False)
    summary_df.to_csv(args.metrics_output, index=False)
    per_source_df.to_csv(args.per_source_output, index=False)

    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
