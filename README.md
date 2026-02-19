# KModel Core

Core machine learning pipeline for predicting which team reaches first 5 kills and first 10 kills in professional League of Legends matches.

This repo intentionally excludes Discord automation, live API polling, and GUI tooling. It focuses on the reproducible training and prediction pipeline.

## What Is Included

- Team/game feature construction from Oracle's Elixir CSV exports
- Time-based training split to reduce leakage risk
- Calibrated binary classifiers for `first5_blue` and `first10_blue`
- Batch prediction pipeline for upcoming matches
- Optional first10 backtesting mode with EV/Kelly staking

## Project Layout

```text
kmodel-core/
  main.py
  first_kills/
    __init__.py
    data_processing.py
    feature_engineering.py
    champion_tags.py
    models.py
    training.py
    predict.py
    betting_first10.py
  tests/
    test_predict_rolling_paths.py
```

## Quick Start

1. Create and activate an environment (Python 3.9+ recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run tests:

```bash
python -m pytest -q
```

## Training

```bash
python main.py --mode first_kills_train \
  --data 2024_LoL_esports_match_data_from_OraclesElixir.csv \
  --split-date 2024-06-01 \
  --output-dir output_first_kills \
  --rolling-window 10
```

## Predicting

```bash
python main.py --mode first_kills_predict \
  --data upcoming_games.csv \
  --historical-data 2024_LoL_esports_match_data_from_OraclesElixir.csv \
  --output-dir output_first_kills \
  --predictions-file predictions.csv
```

## Data

Use Oracle's Elixir exports as input CSVs. Keep raw data and generated artifacts out of version control for a lightweight portfolio repo.

## License

MIT (see `LICENSE`).
