from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from graphrag.data_loader import read_snap_wiki_vote, save_edges_csv
from graphrag.preprocessing import clean_edges


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean the raw Wikipedia Vote edge list.")
    parser.add_argument("--input", default="data/raw/wiki-Vote.txt", help="Raw SNAP text file.")
    parser.add_argument(
        "--output",
        default="data/processed/graph_edges.csv",
        help="Output CSV path for cleaned edges.",
    )
    args = parser.parse_args()

    raw_edges = read_snap_wiki_vote(args.input)
    clean_df = clean_edges(raw_edges)
    save_edges_csv(clean_df, args.output)

    print(f"Saved {len(clean_df):,} cleaned edges to {args.output}")


if __name__ == "__main__":
    main()

