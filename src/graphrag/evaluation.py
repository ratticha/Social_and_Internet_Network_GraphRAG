from __future__ import annotations

import pandas as pd


def evaluate_rankings(ranked_predictions: pd.DataFrame, k: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_columns = {"method", "source", "target", "label", "rank", "score"}
    missing = required_columns - set(ranked_predictions.columns)
    if missing:
        raise ValueError(f"Missing columns for evaluation: {sorted(missing)}")

    topk_df = ranked_predictions[ranked_predictions["rank"] <= k].copy()

    per_source_df = (
        topk_df.groupby(["method", "source"], as_index=False)
        .agg(hits_at_k=("label", "sum"))
        .sort_values(["method", "source"])
        .reset_index(drop=True)
    )
    per_source_df["precision_at_k"] = per_source_df["hits_at_k"] / float(k)

    summary_df = (
        per_source_df.groupby("method", as_index=False)
        .agg(
            mean_precision_at_k=("precision_at_k", "mean"),
            std_precision_at_k=("precision_at_k", "std"),
            total_hits=("hits_at_k", "sum"),
            evaluated_sources=("source", "nunique"),
        )
        .sort_values("mean_precision_at_k", ascending=False)
        .reset_index(drop=True)
    )
    summary_df["std_precision_at_k"] = summary_df["std_precision_at_k"].fillna(0.0)
    summary_df.insert(1, "k", k)

    return summary_df, per_source_df

