from __future__ import annotations

from collections import Counter

import pandas as pd

from graphrag.preprocessing import clean_edges


def random_edge_split(edges: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    clean_df = clean_edges(edges)
    if len(clean_df) < 2:
        raise ValueError("At least two edges are required to create a split.")

    shuffled = clean_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    target_test_count = max(1, round(len(shuffled) * test_size))

    remaining_degree: Counter[int] = Counter()
    for source, target in shuffled.itertuples(index=False, name=None):
        remaining_degree[source] += 1
        remaining_degree[target] += 1

    test_flags = [False] * len(shuffled)
    selected = 0

    for idx, (source, target) in enumerate(shuffled.itertuples(index=False, name=None)):
        if selected >= target_test_count:
            break

        if remaining_degree[source] <= 1 or remaining_degree[target] <= 1:
            continue

        test_flags[idx] = True
        remaining_degree[source] -= 1
        remaining_degree[target] -= 1
        selected += 1

    if selected == 0:
        test_flags[0] = True

    test_mask = pd.Series(test_flags, dtype=bool)
    train_edges = shuffled[~test_mask].reset_index(drop=True)
    test_edges = shuffled[test_mask].reset_index(drop=True)
    return train_edges, test_edges
