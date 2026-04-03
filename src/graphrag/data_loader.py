from __future__ import annotations

from pathlib import Path

import pandas as pd


EDGE_COLUMNS = ["source", "target"]


def read_snap_wiki_vote(file_path: str | Path) -> pd.DataFrame:
    path = Path(file_path)
    rows: list[tuple[int, int]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            source, target = stripped.split()
            rows.append((int(source), int(target)))

    return pd.DataFrame(rows, columns=EDGE_COLUMNS)


def load_edges_csv(file_path: str | Path) -> pd.DataFrame:
    path = Path(file_path)
    edges = pd.read_csv(path)
    if edges.empty:
        return pd.DataFrame(columns=EDGE_COLUMNS)
    return edges[EDGE_COLUMNS].astype(int)


def save_edges_csv(edges: pd.DataFrame, file_path: str | Path) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    edges[EDGE_COLUMNS].to_csv(path, index=False)

