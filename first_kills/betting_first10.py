"""
Betting strategy for First 10 kills predictions.

Implements high-confidence betting with EV calculation, Kelly staking, and backtesting.
"""

import pandas as pd
import numpy as np
import os
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

BetSide = Literal["BLUE", "RED"]


@dataclass
class BetConfig:
    """Configuration for First 10 kills betting strategy."""
    confidence_threshold: float = 0.65   # |p - 0.5| >= (threshold - 0.5)
    min_edge: float = 0.01              # minimum EV edge to bet
    stake_type: Literal["flat", "kelly", "fractional_kelly"] = "fractional_kelly"
    flat_stake: float = 1.0
    kelly_fraction: float = 0.25        # 1.0 = full Kelly, 0.25 default


def implied_prob(odds: float) -> float:
    """
    Convert decimal odds (e.g. 1.85) to implied win probability.
    
    Args:
        odds: Decimal odds (must be > 1.0)
        
    Returns:
        Implied probability (0.0 to 1.0)
    """
    if odds <= 0 or np.isnan(odds):
        return np.nan
    return 1.0 / odds


def expected_value(model_prob: float, odds: float) -> float:
    """
    Compute expected value per 1 unit stake.
    
    EV = p * (odds - 1) - (1 - p) * 1
    where p is model probability and odds is decimal odds.
    
    Args:
        model_prob: Model's probability of winning (0.0 to 1.0)
        odds: Decimal odds (must be > 1.0)
        
    Returns:
        Expected value per unit stake
    """
    if np.isnan(model_prob) or np.isnan(odds) or odds <= 1.0:
        return np.nan
    return model_prob * (odds - 1.0) - (1.0 - model_prob)


def kelly_stake(model_prob: float, odds: float) -> float:
    """
    Compute full Kelly fraction of bankroll for a +EV bet.
    
    For decimal odds O and true prob p:
      edge = p*(O-1) - (1-p)
      b = O - 1
      f* = (b*p - (1-p)) / b = edge / b
    
    Args:
        model_prob: Model's probability of winning (0.0 to 1.0)
        odds: Decimal odds (must be > 1.0)
        
    Returns:
        Kelly fraction (0.0 to 1.0, capped at 1.0)
    """
    if np.isnan(model_prob) or np.isnan(odds) or odds <= 1.0:
        return 0.0
    
    b = odds - 1.0
    if b <= 0:
        return 0.0
    
    edge = expected_value(model_prob, odds)
    f_star = edge / b
    
    # Cap at 1.0 (100% of bankroll max)
    return max(0.0, min(1.0, f_star))


def decide_bet_first10(
    p_blue: float,
    odds_blue: float,
    odds_red: float,
    config: BetConfig
) -> Tuple[Optional[BetSide], float, float]:
    """
    Decide whether to bet on First 10 kills.
    
    Returns:
        (side, edge, stake_fraction)
        - side: "BLUE", "RED", or None if no bet
        - edge: EV (per 1 unit) of chosen side
        - stake_fraction: Kelly fraction (for Kelly-based staking)
    
    Args:
        p_blue: Model probability that blue team gets first 10 kills
        odds_blue: Decimal odds for blue team
        odds_red: Decimal odds for red team
        config: Betting configuration
    """
    if np.isnan(p_blue) or np.isnan(odds_blue) or np.isnan(odds_red):
        return None, 0.0, 0.0
    
    if odds_blue <= 1.0 or odds_red <= 1.0:
        return None, 0.0, 0.0
    
    p_red = 1.0 - p_blue
    
    # Confidence filter: require |p - 0.5| >= (config.confidence_threshold - 0.5)
    conf_margin = config.confidence_threshold - 0.5
    if abs(p_blue - 0.5) < conf_margin:
        return None, 0.0, 0.0
    
    # Compute EV for both sides
    ev_blue = expected_value(p_blue, odds_blue)
    ev_red = expected_value(p_red, odds_red)
    
    # Choose side with higher EV
    if ev_blue <= 0 and ev_red <= 0:
        return None, 0.0, 0.0
    
    if ev_blue >= ev_red:
        chosen_side = "BLUE"
        chosen_prob = p_blue
        chosen_odds = odds_blue
        chosen_ev = ev_blue
    else:
        chosen_side = "RED"
        chosen_prob = p_red
        chosen_odds = odds_red
        chosen_ev = ev_red
    
    # Edge filter
    if chosen_ev < config.min_edge:
        return None, chosen_ev, 0.0
    
    # Kelly fraction
    kelly_f = kelly_stake(chosen_prob, chosen_odds)
    
    return chosen_side, chosen_ev, kelly_f


def compute_stake(
    stake_type: str,
    kelly_fraction: float,
    config: BetConfig,
    bankroll: float
) -> float:
    """
    Compute stake amount based on staking strategy.
    
    Args:
        stake_type: "flat", "kelly", or "fractional_kelly"
        kelly_fraction: Full Kelly fraction (from kelly_stake)
        config: Betting configuration
        bankroll: Current bankroll
        
    Returns:
        Stake amount
    """
    if stake_type == "flat":
        return config.flat_stake
    elif stake_type == "kelly":
        return bankroll * kelly_fraction
    elif stake_type == "fractional_kelly":
        return bankroll * kelly_fraction * config.kelly_fraction
    else:
        return config.flat_stake


def merge_odds_if_needed(
    predictions_df: pd.DataFrame,
    odds_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Merge odds data into predictions DataFrame if odds columns are missing.
    
    Args:
        predictions_df: Predictions DataFrame
        odds_path: Optional path to odds CSV file
        
    Returns:
        DataFrame with odds columns added
    """
    df = predictions_df.copy()
    
    # Check if odds columns exist
    has_odds = all(col in df.columns for col in ['odds_blue_first10', 'odds_red_first10'])
    
    if has_odds:
        return df
    
    if odds_path and os.path.exists(odds_path):
        print(f"Merging odds from: {odds_path}")
        odds_df = pd.read_csv(odds_path)
        # Merge on gameid or date
        merge_cols = ['gameid'] if 'gameid' in odds_df.columns else ['date']
        df = df.merge(odds_df, on=merge_cols, how='left', suffixes=('', '_odds'))
    else:
        print("Warning: Odds columns not found. Creating dummy odds (1.9 for both sides).")
        print("  For real backtesting, add odds_blue_first10 and odds_red_first10 columns.")
        df['odds_blue_first10'] = 1.9
        df['odds_red_first10'] = 1.9
    
    return df


def backtest_first10_high_confidence(
    predictions_path: str,
    config: BetConfig,
    initial_bankroll: float = 100.0,
    odds_path: Optional[str] = None
) -> dict:
    """
    Backtest a high-confidence First 10 kills strategy.
    
    Expects CSV with columns:
      - first10_blue (0/1 label)
      - p_first10_blue (model probability)
      - odds_blue_first10
      - odds_red_first10
    
    Args:
        predictions_path: Path to predictions CSV
        config: Betting configuration
        initial_bankroll: Starting bankroll
        
    Returns:
        Dictionary with backtest results
    """
    df = pd.read_csv(predictions_path)
    
    # Check for required columns
    required_cols = ["first10_blue", "p_first10_blue"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in predictions: {missing_cols}")
    
    # Merge odds if needed
    df = merge_odds_if_needed(df, odds_path)
    
    # Check for odds columns after merge
    if 'odds_blue_first10' not in df.columns or 'odds_red_first10' not in df.columns:
        raise ValueError("Odds columns (odds_blue_first10, odds_red_first10) are required for backtesting")
    
    bankroll = initial_bankroll
    peak_bankroll = bankroll
    bets = []
    equity_curve = [bankroll]
    
    for idx, row in df.iterrows():
        # Skip rows with missing labels (NaN)
        if pd.isna(row["first10_blue"]):
            equity_curve.append(bankroll)
            continue
        
        y_true = int(row["first10_blue"])  # 1 if Blue gets first 10 kills
        p_blue = row["p_first10_blue"]
        odds_blue = row["odds_blue_first10"]
        odds_red = row["odds_red_first10"]
        
        side, edge, kelly_f = decide_bet_first10(
            p_blue=p_blue,
            odds_blue=odds_blue,
            odds_red=odds_red,
            config=config
        )
        
        if side is None:
            equity_curve.append(bankroll)
            continue
        
        stake = compute_stake(
            stake_type=config.stake_type,
            kelly_fraction=kelly_f,
            config=config,
            bankroll=bankroll
        )
        
        if stake <= 0 or stake > bankroll:
            equity_curve.append(bankroll)
            continue
        
        # Determine outcome
        if side == "BLUE":
            won = (y_true == 1)
            odds = odds_blue
            model_prob = p_blue
        else:  # "RED"
            won = (y_true == 0)
            odds = odds_red
            model_prob = 1.0 - p_blue
        
        pnl = stake * (odds - 1.0) if won else -stake
        bankroll += pnl
        peak_bankroll = max(peak_bankroll, bankroll)
        
        bets.append({
            "gameid": row.get("gameid", idx),
            "date": row.get("date", None),
            "blue_team": row.get("blue_team", None),
            "red_team": row.get("red_team", None),
            "side": side,
            "stake": stake,
            "odds": odds,
            "won": int(won),
            "pnl": pnl,
            "bankroll": bankroll,
            "edge": edge,
            "model_prob": model_prob,
        })
        
        equity_curve.append(bankroll)
    
    bets_df = pd.DataFrame(bets)
    n_bets = len(bets_df)
    
    if n_bets > 0:
        hit_rate = bets_df["won"].mean()
        total_pnl = bets_df["pnl"].sum()
        total_staked = bets_df["stake"].sum()
        roi = total_pnl / total_staked if total_staked > 0 else 0.0
        
        # Max drawdown calculation
        equity_series = pd.Series(equity_curve)
        running_max = equity_series.expanding().max()
        drawdown = (running_max - equity_series) / running_max
        max_drawdown = drawdown.max() if len(drawdown) > 0 else 0.0
        
        # Confidence band breakdown
        bins = [0, 0.6, 0.7, 0.8, 0.9, 1.0]
        labels = ["0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]
        bets_df["conf_bin"] = pd.cut(bets_df["model_prob"], bins=bins, labels=labels, include_lowest=True)
        conf_summary = bets_df.groupby("conf_bin").agg({
            "won": ["count", "mean"],
            "pnl": "sum",
            "edge": "mean"
        }).round(4)
        
    else:
        hit_rate = 0.0
        total_pnl = 0.0
        roi = 0.0
        max_drawdown = 0.0
        conf_summary = pd.DataFrame()
    
    return {
        "initial_bankroll": initial_bankroll,
        "final_bankroll": bankroll,
        "n_bets": n_bets,
        "hit_rate": hit_rate,
        "total_pnl": total_pnl,
        "roi": roi,
        "max_drawdown": max_drawdown,
        "bets_df": bets_df,
        "equity_curve": equity_curve,
        "conf_summary": conf_summary,
    }

