# Tests

This folder contains small checks for the main project logic.

## Files

- `test_split.py`: checks train/test splitting and negative sampling behavior
- `test_link_prediction.py`: checks that scoring columns are produced correctly
- `test_evaluation.py`: checks Precision@K aggregation logic

## How to run

```bash
python -m pytest -q
```
