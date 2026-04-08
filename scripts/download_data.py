from __future__ import annotations

import argparse
import gzip
import shutil
import urllib.request
from pathlib import Path


DEFAULT_URL = "https://snap.stanford.edu/data/wiki-Vote.txt.gz"


def download_and_extract(url: str, archive_path: Path, output_path: Path, overwrite: bool = False) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        print(f"Dataset already exists: {output_path}")
        return

    print(f"Downloading dataset from {url}")
    urllib.request.urlretrieve(url, archive_path)

    print(f"Extracting {archive_path.name} to {output_path}")
    with gzip.open(archive_path, "rb") as source, output_path.open("wb") as target:
        shutil.copyfileobj(source, target)

    print("Download complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the SNAP Wikipedia Vote dataset.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Dataset URL.")
    parser.add_argument(
        "--output",
        default="data/raw/wiki-Vote.txt",
        help="Path for the extracted text file.",
    )
    parser.add_argument(
        "--archive",
        default="data/raw/wiki-Vote.txt.gz",
        help="Path for the downloaded archive.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files if they already exist.",
    )
    args = parser.parse_args()

    download_and_extract(
        url=args.url,
        archive_path=Path(args.archive),
        output_path=Path(args.output),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

