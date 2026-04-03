from __future__ import annotations

import pandas as pd


def clean_edges(edges: pd.DataFrame, drop_self_loops: bool = True, drop_duplicates: bool = True) -> pd.DataFrame:
    clean_df = edges[["source", "target"]].copy()
    clean_df["source"] = clean_df["source"].astype(int)
    clean_df["target"] = clean_df["target"].astype(int)

    if drop_self_loops:
        clean_df = clean_df[clean_df["source"] != clean_df["target"]]

    if drop_duplicates:
        clean_df = clean_df.drop_duplicates()

    return clean_df.reset_index(drop=True)

