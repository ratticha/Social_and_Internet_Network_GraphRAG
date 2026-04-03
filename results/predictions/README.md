# Predictions

This folder contains the scored candidate edges and final Top-K predictions.

## Expected files

- `scored_candidates.csv`: all candidate edges with algorithm scores
- `topk_predictions.csv`: Top-K ranked predictions for each method

## Current status

The prediction files have been generated successfully.

## Current result

- `scored_candidates.csv`: all `228,118` candidate edges with model scores
- `topk_predictions.csv`: Top-10 ranked predictions for each evaluated source node and method

## Use in analysis

- Inspect `topk_predictions.csv` for example retrieved links
- Use `scored_candidates.csv` if you want to compare raw scores across methods
