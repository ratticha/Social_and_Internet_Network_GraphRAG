from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from graphrag.evaluation import evaluate_rankings


def test_evaluate_rankings_returns_expected_precision() -> None:
    ranked_predictions = pd.DataFrame(
        [
            ("common_neighbors", 1, 10, 1, 1, 0.9),
            ("common_neighbors", 1, 11, 0, 2, 0.6),
            ("common_neighbors", 2, 20, 0, 1, 0.8),
            ("common_neighbors", 2, 21, 1, 2, 0.4),
        ],
        columns=["method", "source", "target", "label", "rank", "score"],
    )

    summary_df, per_source_df = evaluate_rankings(ranked_predictions, k=2)

    assert len(per_source_df) == 2
    assert summary_df.loc[0, "method"] == "common_neighbors"
    assert summary_df.loc[0, "mean_precision_at_k"] == 0.5
