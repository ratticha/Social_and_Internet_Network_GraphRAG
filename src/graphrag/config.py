from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

DEFAULT_METHODS = (
    "common_neighbors",
    "jaccard",
    "adamic_adar",
    "personalized_pagerank",
)


@dataclass(frozen=True)
class ExperimentConfig:
    raw_path: Path = RAW_DIR / "wiki-Vote.txt"
    processed_path: Path = PROCESSED_DIR / "graph_edges.csv"
    train_path: Path = SPLITS_DIR / "train_edges.csv"
    test_path: Path = SPLITS_DIR / "test_edges.csv"
    negative_path: Path = SPLITS_DIR / "negative_samples.csv"
    scores_path: Path = PREDICTIONS_DIR / "scored_candidates.csv"
    topk_path: Path = PREDICTIONS_DIR / "topk_predictions.csv"
    metrics_path: Path = METRICS_DIR / "evaluation_summary.csv"
    per_source_metrics_path: Path = METRICS_DIR / "per_source_precision.csv"
    test_size: float = 0.2
    negatives_per_positive: int = 10
    top_k: int = 10
    random_seed: int = 42

