from __future__ import annotations

import pandas as pd


def rank_top_k(scored_candidates: pd.DataFrame, method: str, k: int = 10) -> pd.DataFrame:
    required_columns = {"source", "target", "label", method}
    missing = required_columns - set(scored_candidates.columns)
    if missing:
        raise ValueError(f"Missing columns for ranking: {sorted(missing)}")

    ranked = (
        scored_candidates.sort_values(
            by=["source", method, "target"],
            ascending=[True, False, True],
        )
        .groupby("source", group_keys=False)
        .head(k)
        .copy()
    )

    ranked["rank"] = ranked.groupby("source").cumcount() + 1
    ranked["method"] = method
    ranked = ranked.rename(columns={method: "score"})

    return ranked[["method", "source", "target", "label", "rank", "score"]]


def rank_all_methods(scored_candidates: pd.DataFrame, methods: list[str], k: int = 10) -> pd.DataFrame:
    ranked_frames = [rank_top_k(scored_candidates, method=method, k=k) for method in methods]
    return pd.concat(ranked_frames, ignore_index=True)

