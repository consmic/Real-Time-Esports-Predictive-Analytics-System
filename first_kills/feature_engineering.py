"""
Feature engineering for first kills prediction.

Builds pre-game features using rolling team statistics and champion/draft features.
"""

import pandas as pd
import numpy as np
from typing import List

try:
    from .champion_tags import compute_team_comp_features
except ImportError:
    # Fallback if import fails
    from first_kills.champion_tags import compute_team_comp_features


def add_champion_features(games_df: pd.DataFrame, team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add champion/draft-based features to game-level DataFrame.
    
    Args:
        games_df: Game-level DataFrame
        team_df: Team-level DataFrame with pick1-5 columns
        
    Returns:
        Game-level DataFrame with champion features added
    """
    games_df = games_df.copy()
    
    # Initialize composition features for all games (default to 0 if no picks)
    for tag in ['early', 'scaling', 'engage', 'skirmish', 'poke']:
        if f'blue_comp_sum_{tag}' not in games_df.columns:
            games_df[f'blue_comp_sum_{tag}'] = 0.0
        if f'blue_comp_mean_{tag}' not in games_df.columns:
            games_df[f'blue_comp_mean_{tag}'] = 0.0
        if f'red_comp_sum_{tag}' not in games_df.columns:
            games_df[f'red_comp_sum_{tag}'] = 0.0
        if f'red_comp_mean_{tag}' not in games_df.columns:
            games_df[f'red_comp_mean_{tag}'] = 0.0
    
    # Get picks for blue and red teams
    pick_cols = ['pick1', 'pick2', 'pick3', 'pick4', 'pick5']
    available_pick_cols = [col for col in pick_cols if col in team_df.columns]
    
    if len(available_pick_cols) > 0 and len(team_df) > 0:
        blue_picks = team_df[team_df['side'] == 'Blue'][['gameid'] + available_pick_cols].copy()
        red_picks = team_df[team_df['side'] == 'Red'][['gameid'] + available_pick_cols].copy()
        
        # Compute champion features for each team
        blue_comp_features = {}
        red_comp_features = {}
        
        for tag in ['early', 'scaling', 'engage', 'skirmish', 'poke']:
            blue_comp_features[f'blue_comp_sum_{tag}'] = []
            blue_comp_features[f'blue_comp_mean_{tag}'] = []
            red_comp_features[f'red_comp_sum_{tag}'] = []
            red_comp_features[f'red_comp_mean_{tag}'] = []
        
        # Process blue team picks
        for idx, row in blue_picks.iterrows():
            picks = [row.get(f'pick{i}', None) for i in range(1, 6) if f'pick{i}' in available_pick_cols]
            comp_feat = compute_team_comp_features(picks)
            for tag in ['early', 'scaling', 'engage', 'skirmish', 'poke']:
                blue_comp_features[f'blue_comp_sum_{tag}'].append(comp_feat[f'sum_{tag}'])
                blue_comp_features[f'blue_comp_mean_{tag}'].append(comp_feat[f'mean_{tag}'])
        
        # Process red team picks
        for idx, row in red_picks.iterrows():
            picks = [row.get(f'pick{i}', None) for i in range(1, 6) if f'pick{i}' in available_pick_cols]
            comp_feat = compute_team_comp_features(picks)
            for tag in ['early', 'scaling', 'engage', 'skirmish', 'poke']:
                red_comp_features[f'red_comp_sum_{tag}'].append(comp_feat[f'sum_{tag}'])
                red_comp_features[f'red_comp_mean_{tag}'].append(comp_feat[f'mean_{tag}'])
        
        # Create DataFrames and merge (only if we have picks)
        if len(blue_picks) > 0:
            blue_comp_df = pd.DataFrame({'gameid': blue_picks['gameid'].values, **blue_comp_features})
            games_df = games_df.merge(blue_comp_df, on='gameid', how='left', suffixes=('', '_new'))
            # Update only where we have new values
            for tag in ['early', 'scaling', 'engage', 'skirmish', 'poke']:
                for suffix in ['sum', 'mean']:
                    col = f'blue_comp_{suffix}_{tag}'
                    col_new = f'{col}_new'
                    if col_new in games_df.columns:
                        games_df[col] = games_df[col_new].fillna(games_df[col])
                        games_df = games_df.drop(columns=[col_new])
        
        if len(red_picks) > 0:
            red_comp_df = pd.DataFrame({'gameid': red_picks['gameid'].values, **red_comp_features})
            games_df = games_df.merge(red_comp_df, on='gameid', how='left', suffixes=('', '_new'))
            # Update only where we have new values
            for tag in ['early', 'scaling', 'engage', 'skirmish', 'poke']:
                for suffix in ['sum', 'mean']:
                    col = f'red_comp_{suffix}_{tag}'
                    col_new = f'{col}_new'
                    if col_new in games_df.columns:
                        games_df[col] = games_df[col_new].fillna(games_df[col])
                        games_df = games_df.drop(columns=[col_new])
    
    # Compute difference features (always compute, even if values are 0)
    for tag in ['early', 'scaling', 'engage', 'skirmish', 'poke']:
        blue_mean_col = f'blue_comp_mean_{tag}'
        red_mean_col = f'red_comp_mean_{tag}'
        games_df[f'comp_{tag}_diff'] = games_df[blue_mean_col] - games_df[red_mean_col]
    
    return games_df


def add_advanced_rolling_stats(team_df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add advanced rolling team statistics with multiple windows (3, 5, 10).
    
    Computes rolling stats per team from team-level data, then merges to game-level.
    
    Args:
        team_df: Team-level DataFrame (sorted by date)
        games_df: Game-level DataFrame
        
    Returns:
        Game-level DataFrame with advanced rolling stats added
    """
    team_df = team_df.copy()
    games_df = games_df.copy()
    
    # Ensure team_df is sorted by teamid and date
    if 'date' in team_df.columns:
        team_df = team_df.sort_values(['teamid', 'date']).reset_index(drop=True)
    else:
        team_df = team_df.sort_values(['teamid', 'gameid']).reset_index(drop=True)
    
    # Stats to compute rolling averages for
    rolling_stats = [
        'killsat10',
        'golddiffat10',
        'xpdiffat10',
        'firstblood',
        'firstherald',
        'firstdragon'
    ]
    
    # Filter to existing columns
    rolling_stats = [s for s in rolling_stats if s in team_df.columns]
    
    # Windows to use
    windows = [3, 5, 10]
    
    # Compute rolling stats per team
    grouped = team_df.groupby('teamid')
    
    for window in windows:
        for stat in rolling_stats:
            if stat in team_df.columns:
                # Compute rolling mean with shift to avoid leakage
                roll_col = f'roll{window}_{stat}'
                team_df[roll_col] = grouped[stat].shift(1).rolling(window=window, min_periods=1).mean()
    
    # Merge rolling stats to game-level
    # Get blue and red team rolling stats
    blue_team_df = team_df[team_df['side'] == 'Blue'][['gameid', 'teamid'] + 
                                                      [f'roll{w}_{s}' for w in windows for s in rolling_stats]].copy()
    red_team_df = team_df[team_df['side'] == 'Red'][['gameid', 'teamid'] + 
                                                    [f'roll{w}_{s}' for w in windows for s in rolling_stats]].copy()
    
    # Rename columns with blue_/red_ prefix
    blue_rename = {col: f'blue_{col}' for col in blue_team_df.columns if col not in ['gameid', 'teamid']}
    red_rename = {col: f'red_{col}' for col in red_team_df.columns if col not in ['gameid', 'teamid']}
    
    blue_team_df.rename(columns=blue_rename, inplace=True)
    red_team_df.rename(columns=red_rename, inplace=True)
    
    # Merge to games_df
    games_df = games_df.merge(blue_team_df[['gameid'] + list(blue_rename.values())], on='gameid', how='left')
    games_df = games_df.merge(red_team_df[['gameid'] + list(red_rename.values())], on='gameid', how='left')
    
    # Compute difference features
    for window in windows:
        for stat in rolling_stats:
            blue_col = f'blue_roll{window}_{stat}'
            red_col = f'red_roll{window}_{stat}'
            if blue_col in games_df.columns and red_col in games_df.columns:
                diff_col = f'roll{window}_{stat}_diff'
                games_df[diff_col] = games_df[blue_col] - games_df[red_col]
    
    # Fill missing values (for teams with < window games)
    roll_cols = [col for col in games_df.columns if 'roll' in col and ('blue_' in col or 'red_' in col)]
    for col in roll_cols:
        games_df[col] = games_df[col].fillna(games_df[col].median() if games_df[col].notna().any() else 0)
    
    # Recompute diff features after filling
    for window in windows:
        for stat in rolling_stats:
            blue_col = f'blue_roll{window}_{stat}'
            red_col = f'red_roll{window}_{stat}'
            diff_col = f'roll{window}_{stat}_diff'
            if blue_col in games_df.columns and red_col in games_df.columns:
                games_df[diff_col] = games_df[blue_col] - games_df[red_col]
    
    return games_df


def add_team_rolling_stats(games_df: pd.DataFrame, team_df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Main function to add all rolling stats and champion features.
    
    This is a convenience wrapper that calls the advanced rolling stats function.
    
    Args:
        games_df: Game-level DataFrame
        team_df: Team-level DataFrame
        window: Default window (kept for backward compatibility, but we use 3/5/10)
        
    Returns:
        Game-level DataFrame with all features added
    """
    # Add advanced rolling stats with multiple windows
    games_df = add_advanced_rolling_stats(team_df, games_df)
    
    return games_df


def get_first5_feature_cols(games_df: pd.DataFrame) -> List[str]:
    """
    Get list of feature columns for first5 model.
    
    Args:
        games_df: DataFrame with all features
        
    Returns:
        List of feature column names
    """
    feature_cols = [
        # Advanced rolling differences (multiple windows)
        'roll3_killsat10_diff',
        'roll5_killsat10_diff',
        'roll10_killsat10_diff',
        'roll3_golddiffat10_diff',
        'roll5_golddiffat10_diff',
        'roll10_golddiffat10_diff',
        'roll3_xpdiffat10_diff',
        'roll5_xpdiffat10_diff',
        'roll10_xpdiffat10_diff',
        'roll3_firstblood_diff',
        'roll5_firstblood_diff',
        'roll10_firstblood_diff',
        'roll3_firstherald_diff',
        'roll5_firstherald_diff',
        'roll10_firstherald_diff',
        'roll3_firstdragon_diff',
        'roll5_firstdragon_diff',
        'roll10_firstdragon_diff',
        
        # Draft features (comp differences)
        'comp_early_diff',
        'comp_engage_diff',
        'comp_skirmish_diff',
        'comp_poke_diff',
        'comp_scaling_diff',
        
        # Role-based lane matchup features
        'top_early_diff',
        'jungle_early_diff',
        'mid_early_diff',
        'bot_early_diff',
        'support_early_diff',
        'top_killpressure_diff',
        'mid_prio_diff',
        'jungle_skirmish_diff',
        'bot_2v2_killlane_diff',
        'support_engage_diff',
        'lane_pressure_sum_diff',
        'lane_volatility_sum_diff',
        
        # Player-level lane rolling stats (lane matchup features)
        # Note: position names are normalized to lowercase (jng, not jungle)
        'top_roll5_xpdiff10_diff',
        'mid_roll5_xpdiff10_diff',
        'bot_roll5_xpdiff10_diff',
        'jng_roll5_xpdiff10_diff',
        'sup_roll5_xpdiff10_diff',
        'top_roll5_killsat10_diff',
        'mid_roll5_killsat10_diff',
        'bot_roll5_killsat10_diff',
        'jng_roll5_killsat10_diff',
        'sup_roll5_killsat10_diff',
        'top_roll5_golddiffat10_diff',
        'mid_roll5_golddiffat10_diff',
        'bot_roll5_golddiffat10_diff',
        'jng_roll5_golddiffat10_diff',
        'sup_roll5_golddiffat10_diff',
        'top_roll5_csdiffat10_diff',
        'mid_roll5_csdiffat10_diff',
        'bot_roll5_csdiffat10_diff',
        'jng_roll5_csdiffat10_diff',
        'sup_roll5_csdiffat10_diff',
        # Also include roll10 for more stable signals
        'top_roll10_xpdiff10_diff',
        'mid_roll10_xpdiff10_diff',
        'bot_roll10_xpdiff10_diff',
        'jng_roll10_xpdiff10_diff',
        'sup_roll10_xpdiff10_diff',
        'top_roll10_killsat10_diff',
        'mid_roll10_killsat10_diff',
        'bot_roll10_killsat10_diff',
        'jng_roll10_killsat10_diff',
        'sup_roll10_killsat10_diff',
    ]
    
    # Filter to columns that exist
    feature_cols = [col for col in feature_cols if col in games_df.columns]
    
    return feature_cols


def get_lane_champions(game_row: pd.Series) -> dict:
    """
    Extract lane champions from a game row.
    
    Uses role-based champion columns (top_champ, jng_champ, etc.) which are
    extracted from player rows based on actual position assignments.
    Falls back to draft-order picks if role-based columns not available.
    
    Args:
        game_row: Series representing one game
        
    Returns:
        Dictionary with lane assignments
    """
    # Try role-based columns first (these are correctly mapped to positions)
    result = {
        "blue_top": game_row.get("blue_top_champ", None),
        "blue_jungle": game_row.get("blue_jng_champ", None),
        "blue_mid": game_row.get("blue_mid_champ", None),
        "blue_adc": game_row.get("blue_bot_champ", None),
        "blue_support": game_row.get("blue_sup_champ", None),
        "red_top": game_row.get("red_top_champ", None),
        "red_jungle": game_row.get("red_jng_champ", None),
        "red_mid": game_row.get("red_mid_champ", None),
        "red_adc": game_row.get("red_bot_champ", None),
        "red_support": game_row.get("red_sup_champ", None),
    }
    
    # Check if we got role-based picks
    has_role_picks = any(v is not None and pd.notna(v) for v in result.values())
    
    # Fall back to draft-order picks if role-based not available
    # (This maintains backward compatibility but may not be accurate)
    if not has_role_picks:
        result = {
            "blue_top": game_row.get("blue_pick1", None),
            "blue_jungle": game_row.get("blue_pick2", None),
            "blue_mid": game_row.get("blue_pick3", None),
            "blue_adc": game_row.get("blue_pick4", None),
            "blue_support": game_row.get("blue_pick5", None),
            "red_top": game_row.get("red_pick1", None),
            "red_jungle": game_row.get("red_pick2", None),
            "red_mid": game_row.get("red_pick3", None),
            "red_adc": game_row.get("red_pick4", None),
            "red_support": game_row.get("red_pick5", None),
        }
    
    return result


def add_role_based_features(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add role-based lane matchup features.
    
    Computes lane-level early strength, kill pressure, and skirmish differences
    based on champion matchups in each lane.
    
    Uses role-based champion columns (blue_top_champ, etc.) if available,
    otherwise falls back to draft-order picks (blue_pick1, etc.).
    
    Args:
        games_df: Game-level DataFrame with pick columns
        
    Returns:
        Game-level DataFrame with role-based features added
    """
    games_df = games_df.copy()
    
    # Check for role-based columns first (preferred - correctly mapped to positions)
    role_based_cols = ['blue_top_champ', 'blue_jng_champ', 'blue_mid_champ', 'blue_bot_champ', 'blue_sup_champ',
                       'red_top_champ', 'red_jng_champ', 'red_mid_champ', 'red_bot_champ', 'red_sup_champ']
    has_role_based = all(col in games_df.columns for col in role_based_cols)
    
    # Fall back to draft-order picks
    draft_order_cols = ['blue_pick1', 'blue_pick2', 'blue_pick3', 'blue_pick4', 'blue_pick5',
                        'red_pick1', 'red_pick2', 'red_pick3', 'red_pick4', 'red_pick5']
    has_draft_order = all(col in games_df.columns for col in draft_order_cols)
    
    if has_role_based:
        print("  Using role-based champion columns for lane matchup features")
    elif has_draft_order:
        print("  Warning: Role-based columns not found, using draft-order picks (less accurate)")
    else:
        print("Warning: No pick columns found. Skipping role-based features.")
        # Initialize with zeros
        for col in ["top_early_diff", "jungle_early_diff", "mid_early_diff", "bot_early_diff",
                   "support_early_diff", "top_killpressure_diff", "mid_prio_diff",
                   "jungle_skirmish_diff", "bot_2v2_killlane_diff", "support_engage_diff",
                   "lane_pressure_sum_diff", "lane_volatility_sum_diff"]:
            games_df[col] = 0.0
        return games_df
    
    try:
        from .champion_tags import get_role_tag
    except ImportError:
        from first_kills.champion_tags import get_role_tag
    
    # Initialize feature lists
    diffs = {
        "top_early_diff": [],
        "jungle_early_diff": [],
        "mid_early_diff": [],
        "bot_early_diff": [],
        "support_early_diff": [],
        "top_killpressure_diff": [],
        "mid_prio_diff": [],
        "jungle_skirmish_diff": [],
        "bot_2v2_killlane_diff": [],
        "support_engage_diff": [],
        "lane_pressure_sum_diff": [],
        "lane_volatility_sum_diff": [],
    }
    
    for idx, row in games_df.iterrows():
        # Extract lane champions
        lanes = get_lane_champions(row)
        
        def tag(champ, attr, default=5.0):
            """Helper to get tag value with default."""
            return get_role_tag(champ, attr, default=default)
        
        # TOP LANE
        top_early = tag(lanes["blue_top"], "early_strength", 5.0) - tag(lanes["red_top"], "early_strength", 5.0)
        top_kp = tag(lanes["blue_top"], "kill_pressure", 5.0) - tag(lanes["red_top"], "kill_pressure", 5.0)
        
        # JUNGLE
        jng_early = tag(lanes["blue_jungle"], "early_strength", 5.0) - tag(lanes["red_jungle"], "early_strength", 5.0)
        jng_skirm = tag(lanes["blue_jungle"], "skirmish", 5.0) - tag(lanes["red_jungle"], "skirmish", 5.0)
        
        # MID LANE
        mid_early = tag(lanes["blue_mid"], "early_strength", 5.0) - tag(lanes["red_mid"], "early_strength", 5.0)
        mid_prio = tag(lanes["blue_mid"], "lane_prio", 5.0) - tag(lanes["red_mid"], "lane_prio", 5.0)
        
        # BOT LANE (ADC)
        bot_early = tag(lanes["blue_adc"], "early_strength", 5.0) - tag(lanes["red_adc"], "early_strength", 5.0)
        bot_kill2 = tag(lanes["blue_adc"], "2v2_killlane", 5.0) - tag(lanes["red_adc"], "2v2_killlane", 5.0)
        
        # SUPPORT
        sup_early = tag(lanes["blue_support"], "early_strength", 5.0) - tag(lanes["red_support"], "early_strength", 5.0)
        sup_eng = tag(lanes["blue_support"], "engage", 5.0) - tag(lanes["red_support"], "engage", 5.0)
        
        # Append to lists
        diffs["top_early_diff"].append(top_early)
        diffs["top_killpressure_diff"].append(top_kp)
        diffs["jungle_early_diff"].append(jng_early)
        diffs["jungle_skirmish_diff"].append(jng_skirm)
        diffs["mid_early_diff"].append(mid_early)
        diffs["mid_prio_diff"].append(mid_prio)
        diffs["bot_early_diff"].append(bot_early)
        diffs["bot_2v2_killlane_diff"].append(bot_kill2)
        diffs["support_early_diff"].append(sup_early)
        diffs["support_engage_diff"].append(sup_eng)
        
        # Aggregate signals
        lane_pressure = top_early + mid_early + bot_early + sup_early
        lane_volatility = top_kp + mid_prio + jng_skirm + bot_kill2 + sup_eng
        
        diffs["lane_pressure_sum_diff"].append(lane_pressure)
        diffs["lane_volatility_sum_diff"].append(lane_volatility)
    
    # Attach to DataFrame
    for col, values in diffs.items():
        games_df[col] = values
    
    return games_df


def add_player_lane_rolling_stats(player_df: pd.DataFrame, windows: List[int] = [3, 5, 10]) -> pd.DataFrame:
    """
    Compute rolling early-game stats at the player + position level.
    
    For each player at each position, computes rolling averages of early-game stats
    from past N games (no leakage).
    
    Args:
        player_df: Raw per-player DataFrame with columns:
            - gameid, date, teamid, side, position, playerid
            - killsat10, deathsat10, xpat10, goldat10
            - golddiffat10, xpdiffat10, csdiffat10, etc.
        windows: List of window sizes (e.g., [3, 5, 10])
        
    Returns:
        player_df with extra columns:
            - roll{w}_killsat10, roll{w}_deathsat10
            - roll{w}_xpdiffat10, roll{w}_golddiffat10, roll{w}_csdiffat10
        for each window w in windows
    """
    player_df = player_df.copy()
    
    # Ensure date is datetime
    if 'date' in player_df.columns:
        player_df['date'] = pd.to_datetime(player_df['date'], errors='coerce')
    
    # Filter to only starting players (exclude 'team' position)
    if 'position' in player_df.columns:
        valid_positions = ['top', 'jng', 'mid', 'bot', 'sup', 'TOP', 'JUNGLE', 'MID', 'ADC', 'SUPPORT']
        starters_df = player_df[player_df['position'].isin(valid_positions)].copy()
    else:
        print("Warning: No 'position' column found. Using all rows.")
        starters_df = player_df.copy()
    
    if len(starters_df) == 0:
        print("Warning: No starter players found. Returning original DataFrame.")
        return player_df
    
    # Normalize position names to lowercase
    if 'position' in starters_df.columns:
        position_map = {
            'TOP': 'top', 'JUNGLE': 'jng', 'MID': 'mid',
            'ADC': 'bot', 'SUPPORT': 'sup'
        }
        starters_df['position'] = starters_df['position'].map(lambda x: position_map.get(x, x.lower() if isinstance(x, str) else x))
    
    # Sort by playerid, position, and date
    sort_cols = ['playerid', 'position']
    if 'date' in starters_df.columns:
        sort_cols.append('date')
    else:
        sort_cols.append('gameid')
    
    starters_df = starters_df.sort_values(sort_cols).reset_index(drop=True)
    
    # Stats to compute rolling averages for
    rolling_stats = [
        'killsat10', 'deathsat10',
        'xpdiffat10', 'golddiffat10', 'csdiffat10'
    ]
    
    # Filter to existing columns
    rolling_stats = [s for s in rolling_stats if s in starters_df.columns]
    
    if len(rolling_stats) == 0:
        print("Warning: No rolling stat columns found. Available columns:")
        print(f"  {list(starters_df.columns)[:20]}...")
        return player_df
    
    # Group by playerid and position
    grouped = starters_df.groupby(['playerid', 'position'])
    
    # Compute rolling stats for each window
    for window in windows:
        for stat in rolling_stats:
            if stat in starters_df.columns:
                roll_col = f'roll{window}_{stat}'
                # Use shift(1) to prevent leakage, then rolling mean
                starters_df[roll_col] = (
                    grouped[stat]
                    .shift(1)
                    .rolling(window=window, min_periods=1)
                    .mean()
                )
    
    # Merge back to original player_df
    roll_cols = [col for col in starters_df.columns if col.startswith('roll')]
    merge_cols = ['gameid', 'playerid', 'position'] + roll_cols
    
    # Merge rolling stats back
    player_df = player_df.merge(
        starters_df[merge_cols],
        on=['gameid', 'playerid', 'position'],
        how='left',
        suffixes=('', '_roll')
    )
    
    # Fill missing rolling stats (for players with < window games)
    for col in roll_cols:
        if col in player_df.columns:
            player_df[col] = player_df[col].fillna(player_df[col].median() if player_df[col].notna().any() else 0)
    
    return player_df


def build_lane_stats_from_players(player_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build lane-level stats DataFrame from player-level data with rolling stats.
    
    Creates one row per game/team/position with player rolling stats.
    
    Args:
        player_df: Player-level DataFrame with rolling stats columns
        
    Returns:
        DataFrame with columns: gameid, side, position, playerid, and rolling stat columns
    """
    # Filter to starting players
    if 'position' in player_df.columns:
        valid_positions = ['top', 'jng', 'mid', 'bot', 'sup', 'TOP', 'JUNGLE', 'MID', 'ADC', 'SUPPORT']
        starters_df = player_df[player_df['position'].isin(valid_positions)].copy()
    else:
        starters_df = player_df.copy()
    
    if len(starters_df) == 0:
        return pd.DataFrame()
    
    # Normalize position names
    if 'position' in starters_df.columns:
        position_map = {
            'TOP': 'top', 'JUNGLE': 'jng', 'MID': 'mid',
            'ADC': 'bot', 'SUPPORT': 'sup'
        }
        starters_df['position'] = starters_df['position'].map(lambda x: position_map.get(x, x.lower() if isinstance(x, str) else x))
    
    # Get rolling stat columns
    roll_cols = [col for col in starters_df.columns if col.startswith('roll')]
    
    # Required columns
    required_cols = ['gameid', 'side', 'position', 'playerid']
    available_cols = [col for col in required_cols if col in starters_df.columns]
    
    if len(available_cols) < len(required_cols):
        print(f"Warning: Missing required columns. Available: {available_cols}")
        return pd.DataFrame()
    
    # Select columns to keep
    keep_cols = available_cols + roll_cols
    lane_stats_df = starters_df[keep_cols].copy()
    
    # Handle duplicates: if multiple rows per game/team/position, take first
    lane_stats_df = lane_stats_df.drop_duplicates(subset=['gameid', 'side', 'position'], keep='first')
    
    return lane_stats_df


def add_lane_player_rolling_features(games_df: pd.DataFrame, lane_stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge player-level lane rolling stats into game-level frame and compute lane matchup diff features.
    
    Args:
        games_df: Game-level DataFrame
        lane_stats_df: Lane-level stats DataFrame from build_lane_stats_from_players
        
    Returns:
        Game-level DataFrame with lane player rolling diff features added
    """
    games_df = games_df.copy()
    
    if len(lane_stats_df) == 0:
        print("Warning: Empty lane_stats_df. Skipping lane player rolling features.")
        return games_df
    
    # Get rolling stat columns
    roll_cols = [col for col in lane_stats_df.columns if col.startswith('roll')]
    
    if len(roll_cols) == 0:
        print("Warning: No rolling stat columns in lane_stats_df.")
        return games_df
    
    # Pivot lane stats to wide format (one row per game)
    # Create blue and red versions
    blue_lane_stats = lane_stats_df[lane_stats_df['side'] == 'Blue'].copy()
    red_lane_stats = lane_stats_df[lane_stats_df['side'] == 'Red'].copy()
    
    # Pivot blue stats - handle each rolling stat separately to avoid MultiIndex issues
    blue_pivot_list = []
    for roll_col in roll_cols:
        blue_pivot_col = blue_lane_stats.pivot_table(
            index='gameid',
            columns='position',
            values=roll_col,
            aggfunc='first'
        )
        # Rename columns
        blue_pivot_col.columns = [f'blue_{pos}_{roll_col}' for pos in blue_pivot_col.columns]
        blue_pivot_list.append(blue_pivot_col)
    
    # Combine all blue pivots
    if blue_pivot_list:
        blue_pivot = pd.concat(blue_pivot_list, axis=1).reset_index()
    else:
        blue_pivot = pd.DataFrame(columns=['gameid'])
    
    # Pivot red stats - handle each rolling stat separately
    red_pivot_list = []
    for roll_col in roll_cols:
        red_pivot_col = red_lane_stats.pivot_table(
            index='gameid',
            columns='position',
            values=roll_col,
            aggfunc='first'
        )
        # Rename columns
        red_pivot_col.columns = [f'red_{pos}_{roll_col}' for pos in red_pivot_col.columns]
        red_pivot_list.append(red_pivot_col)
    
    # Combine all red pivots
    if red_pivot_list:
        red_pivot = pd.concat(red_pivot_list, axis=1).reset_index()
    else:
        red_pivot = pd.DataFrame(columns=['gameid'])
    
    # Merge into games_df
    games_df = games_df.merge(blue_pivot, on='gameid', how='left')
    games_df = games_df.merge(red_pivot, on='gameid', how='left')
    
    # Compute lane matchup differences
    # For each position and each rolling stat, create diff feature
    positions = ['top', 'jng', 'mid', 'bot', 'sup']
    
    for pos in positions:
        for roll_col in roll_cols:
            blue_col = f'blue_{pos}_{roll_col}'
            red_col = f'red_{pos}_{roll_col}'
            
            if blue_col in games_df.columns and red_col in games_df.columns:
                # Create diff feature name
                # e.g., roll5_xpdiffat10 -> pos_roll5_xpdiff10_diff
                # Handle different roll column formats
                if roll_col.startswith('roll'):
                    # Extract window and stat name
                    parts = roll_col.split('_', 1)
                    if len(parts) == 2:
                        roll_window = parts[0]  # e.g., roll5
                        stat_name = parts[1].replace('at10', '10').replace('at15', '15')  # e.g., xpdiffat10 -> xpdiff10
                        diff_col = f'{pos}_{roll_window}_{stat_name}_diff'
                    else:
                        # Fallback
                        stat_name = roll_col.replace('roll', '').replace('at10', '10').replace('at15', '15')
                        diff_col = f'{pos}_{stat_name}_diff'
                else:
                    stat_name = roll_col.replace('at10', '10').replace('at15', '15')
                    diff_col = f'{pos}_{stat_name}_diff'
                
                # Only compute if both columns are numeric
                if pd.api.types.is_numeric_dtype(games_df[blue_col]) and pd.api.types.is_numeric_dtype(games_df[red_col]):
                    games_df[diff_col] = games_df[blue_col] - games_df[red_col]
    
    # Fill missing values
    diff_cols = [col for col in games_df.columns if col.endswith('_diff') and any(pos in col for pos in positions)]
    for col in diff_cols:
        games_df[col] = games_df[col].fillna(0)
    
    return games_df


def add_team_egi_features(team_df: pd.DataFrame, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
    """
    Compute team-level Early Game Identity (EGI) features over rolling windows.
    
    Features are computed per team (teamid) in chronological order (by date),
    and only use past games via .shift(1).rolling().
    
    Args:
        team_df: Team-level DataFrame
        windows: List of window sizes (e.g., [5, 10, 20])
        
    Returns:
        Team-level DataFrame with EGI features added
    """
    team_df = team_df.copy()
    
    # Ensure sorted by teamid and date
    if 'date' in team_df.columns:
        team_df = team_df.sort_values(['teamid', 'date']).reset_index(drop=True)
    else:
        team_df = team_df.sort_values(['teamid', 'gameid']).reset_index(drop=True)
    
    # Stats to compute rolling averages for
    numeric_stats = ['killsat10', 'deathsat10', 'golddiffat10', 'xpdiffat10', 'csdiffat10']
    binary_stats = ['firstblood', 'firstherald', 'firstdragon']
    optional_stats = ['team kpm', 'ckpm']
    
    # Filter to existing columns
    numeric_stats = [s for s in numeric_stats if s in team_df.columns]
    binary_stats = [s for s in binary_stats if s in team_df.columns]
    optional_stats = [s for s in optional_stats if s in team_df.columns]
    
    # Group by teamid
    grouped = team_df.groupby('teamid')
    
    # Compute rolling stats for each window
    for window in windows:
        # Numeric stats
        for col in numeric_stats:
            egi_col = f'egi_roll{window}_{col}'
            team_df[egi_col] = grouped[col].shift(1).rolling(window=window, min_periods=1).mean()
        
        # Binary stats (as rates)
        for col in binary_stats:
            egi_col = f'egi_roll{window}_{col}_rate'
            team_df[egi_col] = grouped[col].shift(1).rolling(window=window, min_periods=1).mean()
        
        # Optional stats
        for col in optional_stats:
            egi_col = f'egi_roll{window}_{col}'
            team_df[egi_col] = grouped[col].shift(1).rolling(window=window, min_periods=1).mean()
    
    return team_df


def add_egi_to_games(games_df: pd.DataFrame, team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge EGI features into game-level DataFrame and create diff features.
    
    Args:
        games_df: Game-level DataFrame
        team_df: Team-level DataFrame with EGI features
        
    Returns:
        Game-level DataFrame with EGI features added
    """
    games_df = games_df.copy()
    
    # Get EGI columns
    egi_cols = [col for col in team_df.columns if col.startswith('egi_roll')]
    
    if len(egi_cols) == 0:
        print("Warning: No EGI columns found in team_df.")
        return games_df
    
    # Get blue and red EGI stats
    blue_team_df = team_df[team_df['side'] == 'Blue'][['gameid', 'teamid'] + egi_cols].copy()
    red_team_df = team_df[team_df['side'] == 'Red'][['gameid', 'teamid'] + egi_cols].copy()
    
    # Rename with blue_/red_ prefix
    blue_rename = {col: f'blue_{col}' for col in egi_cols}
    red_rename = {col: f'red_{col}' for col in egi_cols}
    
    blue_team_df.rename(columns=blue_rename, inplace=True)
    red_team_df.rename(columns=red_rename, inplace=True)
    
    # Merge to games_df
    games_df = games_df.merge(blue_team_df[['gameid'] + list(blue_rename.values())], on='gameid', how='left')
    games_df = games_df.merge(red_team_df[['gameid'] + list(red_rename.values())], on='gameid', how='left')
    
    # Compute diff features
    for egi_col in egi_cols:
        blue_col = f'blue_{egi_col}'
        red_col = f'red_{egi_col}'
        
        if blue_col in games_df.columns and red_col in games_df.columns:
            # Create diff feature name
            diff_col = f'{egi_col}_diff'
            if pd.api.types.is_numeric_dtype(games_df[blue_col]) and pd.api.types.is_numeric_dtype(games_df[red_col]):
                games_df[diff_col] = games_df[blue_col] - games_df[red_col]
    
    # Fill missing values
    egi_diff_cols = [col for col in games_df.columns if col.startswith('egi_') and col.endswith('_diff')]
    for col in egi_diff_cols:
        games_df[col] = games_df[col].fillna(0)
    
    return games_df


def add_region_patch_features(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add region, league, and patch features to game-level DataFrame.
    
    Args:
        games_df: Game-level DataFrame with league and patch columns
        
    Returns:
        Game-level DataFrame with region/patch features added
    """
    games_df = games_df.copy()
    
    # Region mapping: league -> (region, region_strength)
    REGION_MAP = {
        "LCK": ("KR", 8),
        "LPL": ("CN", 9),
        "LEC": ("EU", 7),
        "LCS": ("NA", 5),
        "PCS": ("PCS", 6),
        "VCS": ("VN", 6),
        "CBLOL": ("BR", 4),
        "LLA": ("LATAM", 4),
        "TCL": ("TR", 5),
        "LJL": ("JP", 5),
        "LCO": ("OCE", 4),
    }
    
    # Map leagues to regions
    if 'league' in games_df.columns:
        # Get blue and red leagues
        blue_league = games_df.get('blue_league', None)
        red_league = games_df.get('red_league', None)
        
        # If league is at game level, use it for both
        if blue_league is None and 'league' in games_df.columns:
            blue_league = games_df['league']
            red_league = games_df['league']
        
        if blue_league is not None:
            # Map to region strength
            def get_region_strength(league):
                if pd.isna(league) or league is None:
                    return 5.0  # Neutral default
                league_str = str(league).upper()
                for key, (region, strength) in REGION_MAP.items():
                    if key in league_str:
                        return float(strength)
                return 5.0  # Default for unknown leagues
            
            games_df['blue_region_strength'] = blue_league.apply(get_region_strength)
            games_df['red_region_strength'] = red_league.apply(get_region_strength) if red_league is not None else games_df['blue_region_strength']
            games_df['region_strength_diff'] = games_df['blue_region_strength'] - games_df['red_region_strength']
    
    # Patch features
    if 'patch' in games_df.columns:
        def extract_patch_info(patch_str):
            if pd.isna(patch_str) or patch_str is None:
                return 14, 0, "unknown"
            
            patch_str = str(patch_str)
            try:
                # Try to parse "14.2" format
                parts = patch_str.split('.')
                if len(parts) >= 2:
                    major = int(float(parts[0]))
                    minor = int(float(parts[1]))
                    
                    # Map to meta era
                    if major == 14:
                        if minor <= 3:
                            meta = "early_2024"
                        elif minor <= 7:
                            meta = "mid_2024"
                        else:
                            meta = "late_2024"
                    else:
                        meta = f"patch_{major}_{minor}"
                    
                    return major, minor, meta
            except:
                pass
            
            return 14, 0, "unknown"
        
        patch_info = games_df['patch'].apply(extract_patch_info)
        games_df['patch_major'] = patch_info.apply(lambda x: x[0])
        games_df['patch_minor'] = patch_info.apply(lambda x: x[1])
        games_df['patch_meta'] = patch_info.apply(lambda x: x[2])
    
    return games_df


def add_synergy_duo_features(games_df: pd.DataFrame, player_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add synergy/duo features for jungle-mid and bot lane pairs.
    
    Args:
        games_df: Game-level DataFrame
        player_df: Player-level DataFrame with playerid and position
        
    Returns:
        Game-level DataFrame with synergy features added
    """
    games_df = games_df.copy()
    
    # Filter to starting players
    if 'position' not in player_df.columns:
        print("Warning: No position column in player_df. Skipping synergy features.")
        return games_df
    
    # Check required columns
    required_cols = ['gameid', 'teamid', 'side', 'playerid', 'position']
    missing_cols = [c for c in required_cols if c not in player_df.columns]
    if missing_cols:
        print(f"Warning: Missing columns for synergy features: {missing_cols}. Skipping.")
        return games_df
    
    # Check for stat columns
    stat_cols = ['killsat10', 'deathsat10', 'firstblood']
    available_stat_cols = [c for c in stat_cols if c in player_df.columns]
    if len(available_stat_cols) == 0:
        print("Warning: No stat columns found for synergy features. Skipping.")
        return games_df
    
    valid_positions = ['top', 'jng', 'mid', 'bot', 'sup', 'TOP', 'JUNGLE', 'MID', 'ADC', 'SUPPORT']
    starters_df = player_df[player_df['position'].isin(valid_positions)].copy()
    
    if len(starters_df) == 0:
        print("Warning: No starter players found. Skipping synergy features.")
        return games_df
    
    # Normalize position names
    position_map = {
        'TOP': 'top', 'JUNGLE': 'jng', 'MID': 'mid',
        'ADC': 'bot', 'SUPPORT': 'sup'
    }
    starters_df['position'] = starters_df['position'].map(lambda x: position_map.get(x, x.lower() if isinstance(x, str) else x))
    
    # Ensure date is datetime
    if 'date' in starters_df.columns:
        starters_df['date'] = pd.to_datetime(starters_df['date'], errors='coerce')
        starters_df = starters_df.sort_values(['teamid', 'date']).reset_index(drop=True)
    
    # Build duo tables
    # Jungle-Mid duos
    jg_mid_df = starters_df[starters_df['position'].isin(['jng', 'mid'])].copy()
    jg_mid_pivot = jg_mid_df.pivot_table(
        index=['gameid', 'teamid', 'side'],
        columns='position',
        values='playerid',
        aggfunc='first'
    ).reset_index()
    
    jg_mid_pivot.columns = ['gameid', 'teamid', 'side', 'jng_playerid', 'mid_playerid']
    jg_mid_pivot = jg_mid_pivot.dropna(subset=['jng_playerid', 'mid_playerid'])
    
    # Bot duos (ADC + Support)
    bot_df = starters_df[starters_df['position'].isin(['bot', 'sup'])].copy()
    bot_pivot = bot_df.pivot_table(
        index=['gameid', 'teamid', 'side'],
        columns='position',
        values='playerid',
        aggfunc='first'
    ).reset_index()
    
    bot_pivot.columns = ['gameid', 'teamid', 'side', 'adc_playerid', 'sup_playerid']
    bot_pivot = bot_pivot.dropna(subset=['adc_playerid', 'sup_playerid'])
    
    # Merge game stats to duos
    # Get stats for each position (use available stat columns)
    stat_cols_to_use = ['gameid', 'teamid', 'playerid'] + available_stat_cols
    jg_stats = starters_df[starters_df['position'] == 'jng'][stat_cols_to_use].copy()
    mid_stats = starters_df[starters_df['position'] == 'mid'][stat_cols_to_use].copy()
    bot_stats = starters_df[starters_df['position'] == 'bot'][stat_cols_to_use].copy()
    sup_stats = starters_df[starters_df['position'] == 'sup'][stat_cols_to_use].copy()
    
    # Fill missing stat columns with 0
    for col in stat_cols:
        if col not in jg_stats.columns:
            jg_stats[col] = 0
            mid_stats[col] = 0
            bot_stats[col] = 0
            sup_stats[col] = 0
    
    # For jungle-mid: aggregate stats for the duo
    jg_mid_with_stats = jg_mid_pivot.merge(
        jg_stats,
        left_on=['gameid', 'teamid', 'jng_playerid'],
        right_on=['gameid', 'teamid', 'playerid'],
        how='left',
        suffixes=('', '_jng')
    ).merge(
        mid_stats,
        left_on=['gameid', 'teamid', 'mid_playerid'],
        right_on=['gameid', 'teamid', 'playerid'],
        how='left',
        suffixes=('', '_mid')
    )
    
    # Compute duo-level stats (handle missing columns)
    if 'killsat10' in available_stat_cols:
        if 'killsat10' in jg_mid_with_stats.columns and 'killsat10_mid' in jg_mid_with_stats.columns:
            jg_mid_with_stats['duo_killsat10'] = (
                jg_mid_with_stats['killsat10'].fillna(0) + jg_mid_with_stats['killsat10_mid'].fillna(0)
            )
        else:
            jg_mid_with_stats['duo_killsat10'] = 0
    else:
        jg_mid_with_stats['duo_killsat10'] = 0
    
    if 'deathsat10' in available_stat_cols:
        if 'deathsat10' in jg_mid_with_stats.columns and 'deathsat10_mid' in jg_mid_with_stats.columns:
            jg_mid_with_stats['duo_deathsat10'] = (
                jg_mid_with_stats['deathsat10'].fillna(0) + jg_mid_with_stats['deathsat10_mid'].fillna(0)
            )
        else:
            jg_mid_with_stats['duo_deathsat10'] = 0
    else:
        jg_mid_with_stats['duo_deathsat10'] = 0
    
    if 'firstblood' in available_stat_cols:
        if 'firstblood' in jg_mid_with_stats.columns and 'firstblood_mid' in jg_mid_with_stats.columns:
            jg_mid_with_stats['duo_firstblood'] = (
                (jg_mid_with_stats['firstblood'].fillna(0) + jg_mid_with_stats['firstblood_mid'].fillna(0)) > 0
            ).astype(int)
        else:
            jg_mid_with_stats['duo_firstblood'] = 0
    else:
        jg_mid_with_stats['duo_firstblood'] = 0
    
    # Compute rolling stats for jg-mid duos
    jg_mid_with_stats = jg_mid_with_stats.sort_values(['teamid', 'jng_playerid', 'mid_playerid', 'gameid'])
    grouped_jg_mid = jg_mid_with_stats.groupby(['teamid', 'jng_playerid', 'mid_playerid'])
    
    jg_mid_with_stats['duo_roll5_killsat10'] = grouped_jg_mid['duo_killsat10'].shift(1).rolling(5, min_periods=1).mean()
    jg_mid_with_stats['duo_roll5_deathsat10'] = grouped_jg_mid['duo_deathsat10'].shift(1).rolling(5, min_periods=1).mean()
    jg_mid_with_stats['duo_roll5_firstblood_rate'] = grouped_jg_mid['duo_firstblood'].shift(1).rolling(5, min_periods=1).mean()
    
    # Similar for bot duos
    bot_with_stats = bot_pivot.merge(
        bot_stats,
        left_on=['gameid', 'teamid', 'adc_playerid'],
        right_on=['gameid', 'teamid', 'playerid'],
        how='left',
        suffixes=('', '_adc')
    ).merge(
        sup_stats,
        left_on=['gameid', 'teamid', 'sup_playerid'],
        right_on=['gameid', 'teamid', 'playerid'],
        how='left',
        suffixes=('', '_sup')
    )
    
    # Compute bot duo stats (handle missing columns)
    if 'killsat10' in available_stat_cols:
        if 'killsat10' in bot_with_stats.columns and 'killsat10_sup' in bot_with_stats.columns:
            bot_with_stats['duo_killsat10'] = (
                bot_with_stats['killsat10'].fillna(0) + bot_with_stats['killsat10_sup'].fillna(0)
            )
        else:
            bot_with_stats['duo_killsat10'] = 0
    else:
        bot_with_stats['duo_killsat10'] = 0
    
    if 'deathsat10' in available_stat_cols:
        if 'deathsat10' in bot_with_stats.columns and 'deathsat10_sup' in bot_with_stats.columns:
            bot_with_stats['duo_deathsat10'] = (
                bot_with_stats['deathsat10'].fillna(0) + bot_with_stats['deathsat10_sup'].fillna(0)
            )
        else:
            bot_with_stats['duo_deathsat10'] = 0
    else:
        bot_with_stats['duo_deathsat10'] = 0
    
    if 'firstblood' in available_stat_cols:
        if 'firstblood' in bot_with_stats.columns and 'firstblood_sup' in bot_with_stats.columns:
            bot_with_stats['duo_firstblood'] = (
                (bot_with_stats['firstblood'].fillna(0) + bot_with_stats['firstblood_sup'].fillna(0)) > 0
            ).astype(int)
        else:
            bot_with_stats['duo_firstblood'] = 0
    else:
        bot_with_stats['duo_firstblood'] = 0
    
    bot_with_stats = bot_with_stats.sort_values(['teamid', 'adc_playerid', 'sup_playerid', 'gameid'])
    grouped_bot = bot_with_stats.groupby(['teamid', 'adc_playerid', 'sup_playerid'])
    
    bot_with_stats['duo_roll5_killsat10'] = grouped_bot['duo_killsat10'].shift(1).rolling(5, min_periods=1).mean()
    bot_with_stats['duo_roll5_deathsat10'] = grouped_bot['duo_deathsat10'].shift(1).rolling(5, min_periods=1).mean()
    bot_with_stats['duo_roll5_firstblood_rate'] = grouped_bot['duo_firstblood'].shift(1).rolling(5, min_periods=1).mean()
    
    # Merge to games_df
    jg_mid_blue = jg_mid_with_stats[jg_mid_with_stats['side'] == 'Blue'][['gameid', 'duo_roll5_killsat10', 'duo_roll5_deathsat10', 'duo_roll5_firstblood_rate']].copy()
    jg_mid_blue.columns = ['gameid', 'blue_jg_mid_duo_roll5_killsat10', 'blue_jg_mid_duo_roll5_deathsat10', 'blue_jg_mid_duo_roll5_firstblood_rate']
    
    jg_mid_red = jg_mid_with_stats[jg_mid_with_stats['side'] == 'Red'][['gameid', 'duo_roll5_killsat10', 'duo_roll5_deathsat10', 'duo_roll5_firstblood_rate']].copy()
    jg_mid_red.columns = ['gameid', 'red_jg_mid_duo_roll5_killsat10', 'red_jg_mid_duo_roll5_deathsat10', 'red_jg_mid_duo_roll5_firstblood_rate']
    
    bot_blue = bot_with_stats[bot_with_stats['side'] == 'Blue'][['gameid', 'duo_roll5_killsat10', 'duo_roll5_deathsat10', 'duo_roll5_firstblood_rate']].copy()
    bot_blue.columns = ['gameid', 'blue_bot_duo_roll5_killsat10', 'blue_bot_duo_roll5_deathsat10', 'blue_bot_duo_roll5_firstblood_rate']
    
    bot_red = bot_with_stats[bot_with_stats['side'] == 'Red'][['gameid', 'duo_roll5_killsat10', 'duo_roll5_deathsat10', 'duo_roll5_firstblood_rate']].copy()
    bot_red.columns = ['gameid', 'red_bot_duo_roll5_killsat10', 'red_bot_duo_roll5_deathsat10', 'red_bot_duo_roll5_firstblood_rate']
    
    games_df = games_df.merge(jg_mid_blue, on='gameid', how='left')
    games_df = games_df.merge(jg_mid_red, on='gameid', how='left')
    games_df = games_df.merge(bot_blue, on='gameid', how='left')
    games_df = games_df.merge(bot_red, on='gameid', how='left')
    
    # Compute diff features
    games_df['jg_mid_duo_roll5_kills10_diff'] = (
        games_df['blue_jg_mid_duo_roll5_killsat10'].fillna(0) - games_df['red_jg_mid_duo_roll5_killsat10'].fillna(0)
    )
    games_df['jg_mid_duo_roll5_deaths10_diff'] = (
        games_df['blue_jg_mid_duo_roll5_deathsat10'].fillna(0) - games_df['red_jg_mid_duo_roll5_deathsat10'].fillna(0)
    )
    games_df['jg_mid_duo_roll5_firstblood_rate_diff'] = (
        games_df['blue_jg_mid_duo_roll5_firstblood_rate'].fillna(0) - games_df['red_jg_mid_duo_roll5_firstblood_rate'].fillna(0)
    )
    
    games_df['bot_duo_roll5_kills10_diff'] = (
        games_df['blue_bot_duo_roll5_killsat10'].fillna(0) - games_df['red_bot_duo_roll5_killsat10'].fillna(0)
    )
    games_df['bot_duo_roll5_deaths10_diff'] = (
        games_df['blue_bot_duo_roll5_deathsat10'].fillna(0) - games_df['red_bot_duo_roll5_deathsat10'].fillna(0)
    )
    games_df['bot_duo_roll5_firstblood_rate_diff'] = (
        games_df['blue_bot_duo_roll5_firstblood_rate'].fillna(0) - games_df['red_bot_duo_roll5_firstblood_rate'].fillna(0)
    )
    
    return games_df


def get_first10_feature_cols(games_df: pd.DataFrame) -> List[str]:
    """
    Get list of feature columns for first10 model (dedicated, richer feature set).
    
    Args:
        games_df: DataFrame with all features
        
    Returns:
        List of feature column names optimized for first10 prediction
    """
    feature_cols = [
        # Team rolling diffs (prioritize 10/20 windows for first10)
        'roll10_killsat10_diff',
        'roll10_golddiffat10_diff',
        'roll10_xpdiffat10_diff',
        'roll10_firstblood_diff',
        'roll10_firstherald_diff',
        'roll10_firstdragon_diff',
        'roll5_killsat10_diff',
        'roll5_golddiffat10_diff',
        'roll5_xpdiffat10_diff',
        
        # EGI features (Early Game Identity)
        'egi_roll10_killsat10_diff',
        'egi_roll10_golddiffat10_diff',
        'egi_roll10_xpdiffat10_diff',
        'egi_roll10_firstblood_rate_diff',
        'egi_roll10_firstherald_rate_diff',
        'egi_roll10_firstdragon_rate_diff',
        'egi_roll20_killsat10_diff',
        'egi_roll20_golddiffat10_diff',
        'egi_roll20_firstblood_rate_diff',
        'egi_roll10_ckpm_diff',
        
        # Champion comp features
        'comp_early_diff',
        'comp_engage_diff',
        'comp_skirmish_diff',
        
        # Role-based lane matchup features
        'top_early_diff',
        'jungle_early_diff',
        'mid_early_diff',
        'bot_early_diff',
        'support_early_diff',
        'mid_prio_diff',
        'jungle_skirmish_diff',
        'bot_2v2_killlane_diff',
        'support_engage_diff',
        
        # Player-level lane rolling stats (prioritize xpdiff and golddiff)
        'top_roll5_xpdiff10_diff',
        'mid_roll5_xpdiff10_diff',
        'bot_roll5_xpdiff10_diff',
        'jng_roll5_xpdiff10_diff',
        'top_roll5_golddiffat10_diff',
        'mid_roll5_golddiffat10_diff',
        'bot_roll5_golddiffat10_diff',
        'jng_roll5_golddiffat10_diff',
        'top_roll10_xpdiff10_diff',
        'mid_roll10_xpdiff10_diff',
        'bot_roll10_xpdiff10_diff',
        'jng_roll10_xpdiff10_diff',
        
        # Region and patch features
        'region_strength_diff',
        'patch_major',
        'patch_minor',
        
        # Synergy/duo features
        'jg_mid_duo_roll5_kills10_diff',
        'jg_mid_duo_roll5_firstblood_rate_diff',
        'bot_duo_roll5_kills10_diff',
        'bot_duo_roll5_firstblood_rate_diff',
    ]
    
    # Filter to columns that exist
    feature_cols = [col for col in feature_cols if col in games_df.columns]
    
    return feature_cols
