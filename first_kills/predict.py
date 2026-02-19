"""
Prediction script for first kills models.
"""

import pandas as pd
import numpy as np
import os
from typing import Optional

from .models import FirstKillsModel
from .feature_engineering import (
    add_team_rolling_stats,
    get_first5_feature_cols,
    get_first10_feature_cols
)
from .data_processing import (
    load_raw_data,
    build_team_level,
    build_game_level
)


def predict_first_kills_for_games(
    upcoming_games_df: pd.DataFrame,
    models_dir: str = 'output_first_kills',
    rolling_window: int = 10,
    team_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Predict first5 and first10 probabilities for upcoming games.
    
    Args:
        upcoming_games_df: Game-level DataFrame for games to predict
        models_dir: Directory containing saved models
        rolling_window: Window size for rolling stats (should match training)
        team_df: Team-level DataFrame used to compute rolling stats if missing
        
    Returns:
        DataFrame with predictions (p_first5_blue, p_first10_blue, etc.)
    """
    # Load models
    first5_model_path = os.path.join(models_dir, 'first5_model.pkl')
    first5_calibrator_path = os.path.join(models_dir, 'first5_calibrator.pkl')
    first10_model_path = os.path.join(models_dir, 'first10_model.pkl')
    first10_calibrator_path = os.path.join(models_dir, 'first10_calibrator.pkl')
    
    if not os.path.exists(first5_model_path):
        raise FileNotFoundError(f"Model not found: {first5_model_path}")
    if not os.path.exists(first10_model_path):
        raise FileNotFoundError(f"Model not found: {first10_model_path}")
    
    print("Loading models...")
    first5_model = FirstKillsModel.load(first5_model_path, first5_calibrator_path)
    first10_model = FirstKillsModel.load(first10_model_path, first10_calibrator_path)
    
    # Add rolling stats (this requires historical data)
    # For now, we'll assume upcoming_games_df already has rolling stats
    # In production, you'd need to merge with historical data first
    has_rolling_stats = any(col.startswith('roll') for col in upcoming_games_df.columns)
    if not has_rolling_stats:
        print("Warning: Rolling stats not found. Adding them...")
        if team_df is None:
            raise ValueError(
                "Rolling stats are missing and team_df was not provided. "
                "Use predict_from_raw_data(...) or pass a team-level DataFrame."
            )
        upcoming_games_df = add_team_rolling_stats(
            upcoming_games_df, team_df, window=rolling_window
        )
    
    # Get feature columns
    first5_features = get_first5_feature_cols(upcoming_games_df)
    first10_features = get_first10_feature_cols(upcoming_games_df)
    
    # Ensure we have all required features
    missing_first5 = set(first5_model.feature_cols or []) - set(upcoming_games_df.columns)
    missing_first10 = set(first10_model.feature_cols or []) - set(upcoming_games_df.columns)
    
    if missing_first5:
        print(f"Warning: Missing features for first5: {missing_first5}")
        # Fill with 0
        for col in missing_first5:
            upcoming_games_df[col] = 0
    
    if missing_first10:
        print(f"Warning: Missing features for first10: {missing_first10}")
        # Fill with 0
        for col in missing_first10:
            upcoming_games_df[col] = 0
    
    # Make predictions
    print("Making predictions...")
    p_first5_blue = first5_model.predict_proba(upcoming_games_df)[:, 1]
    p_first10_blue = first10_model.predict_proba(upcoming_games_df)[:, 1]
    
    # Create results DataFrame
    results = pd.DataFrame({
        'gameid': upcoming_games_df['gameid'].values,
        'date': upcoming_games_df['date'].values if 'date' in upcoming_games_df.columns else None,
        'blue_team': upcoming_games_df['blue_teamname'].values if 'blue_teamname' in upcoming_games_df.columns else None,
        'red_team': upcoming_games_df['red_teamname'].values if 'red_teamname' in upcoming_games_df.columns else None,
        'p_first5_blue': p_first5_blue,
        'p_first5_red': 1 - p_first5_blue,
        'p_first10_blue': p_first10_blue,
        'p_first10_red': 1 - p_first10_blue,
    })
    
    # Remove None columns
    results = results.dropna(axis=1, how='all')
    
    return results


def predict_from_raw_data(
    raw_data_path: str,
    models_dir: str = 'output_first_kills',
    rolling_window: int = 10,
    historical_data_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Load raw data, process it, and make predictions.
    
    This is a convenience function that handles the full pipeline.
    
    Args:
        raw_data_path: Path to raw Oracle's Elixir CSV
        models_dir: Directory containing saved models
        rolling_window: Window size for rolling stats
        historical_data_path: Optional path to historical data for rolling stats
        
    Returns:
        DataFrame with predictions
    """
    # Load and process data
    print("Loading raw data...")
    df_raw = load_raw_data(raw_data_path)
    
    print("Building team-level data...")
    team_df = build_team_level(df_raw)
    
    print("Building game-level data...")
    games_df = build_game_level(team_df)
    
    # If historical data provided, merge for rolling stats
    if historical_data_path:
        print("Loading historical data for rolling stats...")
        hist_raw = load_raw_data(historical_data_path)
        hist_team = build_team_level(hist_raw)
        hist_games = build_game_level(hist_team)
        
        # Combine historical and new games
        all_games = pd.concat([hist_games, games_df], ignore_index=True)
        all_games = all_games.sort_values('date').reset_index(drop=True)
        all_team = pd.concat([hist_team, team_df], ignore_index=True)
        all_team = all_team.sort_values(['teamid', 'date']).reset_index(drop=True)
        
        # Add rolling stats to all games
        all_games = add_team_rolling_stats(all_games, all_team, window=rolling_window)
        
        # Extract only the new games
        new_gameids = set(games_df['gameid'])
        games_df = all_games[all_games['gameid'].isin(new_gameids)].copy()
    else:
        # Add rolling stats (may be incomplete without history)
        games_df = add_team_rolling_stats(games_df, team_df, window=rolling_window)
    
    # Make predictions
    predictions = predict_first_kills_for_games(
        games_df, models_dir, rolling_window
    )
    
    return predictions

