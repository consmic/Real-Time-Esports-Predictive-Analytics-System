Esports Kill-Race Prediction System

A machine learning pipeline that predicts **which team reaches the first 5 and first 10 kills** in professional League of Legends matches, with a backtesting mode that sizes bets by expected value and Kelly staking.

## What it does

- **Features** — constructs team and game features from [Oracle's Elixir](https://oracleselixir.com/) match-data CSV exports.
- **Leakage control** — uses a **time-based train/test split** so the model is never evaluated on information that came before it in training, which is the easiest thing to get wrong with match data.
- **Models** — calibrated binary classifiers for `first5_blue` and `first10_blue` (predicting the blue side's kill-race outcomes). Calibration matters here: the downstream staking only works if the predicted probabilities are honest.
- **Prediction** — a batch pipeline that scores upcoming matches.
- **Backtesting** — an optional `first10` mode that evaluates a staking strategy using expected value and Kelly sizing.

## Quick start

Requires Python 3.9+.

```bash
pip install -r requirements.txt

# Train the kill-race classifiers
python main.py --mode first_kills_train

# Predict upcoming matches
python main.py --mode first_kills_predict
