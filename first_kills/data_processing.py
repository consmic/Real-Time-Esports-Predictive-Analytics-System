"""
Data processing module for first kills prediction.

Handles loading raw Oracle's Elixir data, aggregating to team-level,
and building game-level features.
"""

import pandas as pd
import numpy as np
from typing import Optional
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load raw Oracle's Elixir dataset from CSV.
    
    Args:
        path: Path to CSV file (can be relative or absolute)
        
    Returns:
        DataFrame with all columns, date parsed as datetime
    """
    # Try to find the file in multiple locations
    path_obj = Path(path)
    script_dir = Path(__file__).parent.parent
    cwd = Path.cwd()
    
    # List of paths to try
    paths_to_try = []
    
    # If absolute path, try it directly
    if path_obj.is_absolute():
        paths_to_try.append(path_obj)
    else:
        # Try relative to current working directory
        paths_to_try.append(cwd / path)
        # Try relative to script directory
        paths_to_try.append(script_dir / path)
    
    # If filename contains " (1)", also try without it
    if ' (1)' in path_obj.name:
        alt_name = path_obj.name.replace(' (1)', '')
        if path_obj.is_absolute():
            paths_to_try.append(path_obj.parent / alt_name)
        else:
            paths_to_try.append(cwd / alt_name)
            paths_to_try.append(script_dir / alt_name)
    
    # Try each path until we find one that exists
    found_path = None
    for try_path in paths_to_try:
        if try_path.exists():
            found_path = try_path
            break
    
    # If still not found, provide helpful error
    if found_path is None:
        # List similar files in current directory to help user
        similar_files = [f.name for f in cwd.glob('*.csv') 
                        if 'OraclesElixir' in f.name or 'oracle' in f.name.lower()]
        error_msg = f"File not found: {path}\n"
        error_msg += f"  Checked locations:\n"
        for try_path in paths_to_try:
            error_msg += f"    - {try_path}\n"
        if similar_files:
            error_msg += f"\n  Similar CSV files found in current directory:\n"
            for f in similar_files[:5]:  # Show up to 5 similar files
                error_msg += f"    - {f}\n"
        raise FileNotFoundError(error_msg)
    
    # Convert to string for pandas
    path_str = str(found_path.resolve())
    df = pd.read_csv(path_str, low_memory=False)
    
    # Parse date column
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Ensure key columns are present and correctly typed
    if 'gameid' in df.columns:
        df['gameid'] = df['gameid'].astype(str)
    if 'teamid' in df.columns:
        df['teamid'] = df['teamid'].astype(str)
    if 'side' in df.columns:
        df['side'] = df['side'].astype(str)
    if 'teamname' in df.columns:
        df['teamname'] = df['teamname'].astype(str)
    if 'league' in df.columns:
        df['league'] = df['league'].astype(str)
    if 'patch' in df.columns:
        df['patch'] = df['patch'].astype(str)
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
    
    return df


def build_team_level(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw per-player data to team-level.
    
    For each gameid + teamid combination, aggregates:
    - Team stats (kills, deaths, assists, etc.)
    - Early game stats (killsat10, goldat10, etc.)
    - First objectives (firstblood, firstdragon, etc.)
    
    Args:
        df_raw: Raw per-player DataFrame
        
    Returns:
        DataFrame with one row per team per game
    """
    # Check if there's a 'team' position row (aggregated team data)
    if 'position' in df_raw.columns:
        team_rows = df_raw[df_raw['position'] == 'team'].copy()
        if len(team_rows) > 0:
            # Use team rows directly, but ensure we have all needed columns
            team_df = team_rows.copy()
        else:
            # Aggregate from player rows
            team_df = _aggregate_from_players(df_raw)
    else:
        # No position column, aggregate from players
        team_df = _aggregate_from_players(df_raw)
    
    # Ensure we have key identifiers
    required_cols = ['gameid', 'teamid', 'teamname', 'side', 'league', 'year', 
                     'split', 'playoffs', 'patch', 'gamelength', 'result', 'date']
    for col in required_cols:
        if col not in team_df.columns:
            if col == 'date' and 'date' in df_raw.columns:
                # Merge date from raw data
                date_map = df_raw.groupby('gameid')['date'].first()
                team_df['date'] = team_df['gameid'].map(date_map)
            else:
                # Try to get from first row per gameid+teamid
                if col in df_raw.columns:
                    col_map = df_raw.groupby(['gameid', 'teamid'])[col].first()
                    team_df[col] = team_df.set_index(['gameid', 'teamid']).index.map(
                        lambda x: col_map.get(x, None)
                    )
    
    # Sort by date and gameid for chronological ordering
    if 'date' in team_df.columns:
        team_df = team_df.sort_values(['date', 'gameid', 'side']).reset_index(drop=True)
    else:
        team_df = team_df.sort_values(['gameid', 'side']).reset_index(drop=True)
    
    # Add role-based champion picks from player rows
    team_df = _add_role_based_picks(team_df, df_raw)
    
    return team_df


def _add_role_based_picks(team_df: pd.DataFrame, df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Extract champion picks by role from player rows and add to team DataFrame.
    
    Creates columns: top_champ, jng_champ, mid_champ, bot_champ, sup_champ
    These are based on actual player positions, not draft order.
    
    Args:
        team_df: Team-level DataFrame
        df_raw: Raw per-player DataFrame with position and champion columns
        
    Returns:
        Team DataFrame with role-based champion columns added
    """
    if 'position' not in df_raw.columns or 'champion' not in df_raw.columns:
        print("Warning: position or champion column not found. Skipping role-based picks.")
        return team_df
    
    # Filter to player rows only (exclude 'team' position)
    player_rows = df_raw[df_raw['position'] != 'team'].copy()
    
    if len(player_rows) == 0:
        print("Warning: No player rows found. Skipping role-based picks.")
        return team_df
    
    # Normalize position names
    position_map = {
        'top': 'top', 'TOP': 'top',
        'jng': 'jng', 'jungle': 'jng', 'JUNGLE': 'jng', 'jg': 'jng',
        'mid': 'mid', 'MID': 'mid', 'middle': 'mid',
        'bot': 'bot', 'adc': 'bot', 'ADC': 'bot', 'BOT': 'bot',
        'sup': 'sup', 'support': 'sup', 'SUPPORT': 'sup', 'SUP': 'sup'
    }
    
    player_rows['position_norm'] = player_rows['position'].map(
        lambda x: position_map.get(str(x).lower(), str(x).lower()) if pd.notna(x) else None
    )
    
    # Valid positions
    valid_positions = ['top', 'jng', 'mid', 'bot', 'sup']
    player_rows = player_rows[player_rows['position_norm'].isin(valid_positions)]
    
    if len(player_rows) == 0:
        print("Warning: No valid player positions found. Skipping role-based picks.")
        return team_df
    
    # Create pivot table: gameid + teamid -> position -> champion
    # Using first() in case of duplicates
    picks_pivot = player_rows.pivot_table(
        index=['gameid', 'teamid'],
        columns='position_norm',
        values='champion',
        aggfunc='first'
    ).reset_index()
    
    # Rename columns to role_champ format
    rename_map = {}
    for pos in valid_positions:
        if pos in picks_pivot.columns:
            rename_map[pos] = f'{pos}_champ'
    picks_pivot.rename(columns=rename_map, inplace=True)
    
    # Merge with team_df
    role_champ_cols = [f'{pos}_champ' for pos in valid_positions if f'{pos}_champ' in picks_pivot.columns]
    if role_champ_cols:
        merge_cols = ['gameid', 'teamid'] + role_champ_cols
        # Only keep columns that exist
        merge_cols = [c for c in merge_cols if c in picks_pivot.columns]
        team_df = team_df.merge(picks_pivot[merge_cols], on=['gameid', 'teamid'], how='left')
        
        # Count how many games got role picks
        games_with_picks = team_df[role_champ_cols[0]].notna().sum() if role_champ_cols else 0
        print(f"  Added role-based champion picks for {games_with_picks}/{len(team_df)} team rows")
    
    return team_df


def _aggregate_from_players(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate team-level stats from player rows."""
    
    # Columns to sum
    sum_cols = [
        'kills', 'deaths', 'assists', 'teamkills', 'teamdeaths',
        'doublekills', 'triplekills', 'quadrakills', 'pentakills',
        'dragons', 'opp_dragons', 'elementaldrakes', 'opp_elementaldrakes',
        'infernals', 'mountains', 'clouds', 'oceans', 'chemtechs', 'hextechs',
        'elders', 'opp_elders', 'heralds', 'opp_heralds', 'void_grubs', 'opp_void_grubs',
        'barons', 'opp_barons', 'atakhans', 'opp_atakhans',
        'towers', 'opp_towers', 'turretplates', 'opp_turretplates',
        'inhibitors', 'opp_inhibitors', 'wardsplaced', 'wardskilled',
        'controlwardsbought', 'totalgold', 'earnedgold', 'goldspent',
        'total cs', 'minionkills', 'monsterkills', 'monsterkillsownjungle',
        'monsterkillsenemyjungle'
    ]
    
    # Columns to take first (should be same for all players on team)
    first_cols = [
        'gameid', 'teamid', 'teamname', 'side', 'league', 'year', 'split',
        'playoffs', 'patch', 'gamelength', 'result', 'date',
        'team kpm', 'ckpm', 'firstblood', 'firstdragon', 'firstherald',
        'firsttower', 'firstmidtower', 'firsttothreetowers',
        'goldat10', 'xpat10', 'csat10', 'opp_goldat10', 'opp_xpat10', 'opp_csat10',
        'golddiffat10', 'xpdiffat10', 'csdiffat10',
        'killsat10', 'assistsat10', 'deathsat10',
        'opp_killsat10', 'opp_assistsat10', 'opp_deathsat10',
        'goldat15', 'xpat15', 'csat15', 'opp_goldat15', 'opp_xpat15', 'opp_csat15',
        'golddiffat15', 'xpdiffat15', 'csdiffat15',
        'killsat15', 'assistsat15', 'deathsat15',
        'opp_killsat15', 'opp_assistsat15', 'opp_deathsat15',
        'goldat20', 'xpat20', 'csat20', 'opp_goldat20', 'opp_xpat20', 'opp_csat20',
        'golddiffat20', 'xpdiffat20', 'csdiffat20',
        'killsat20', 'assistsat20', 'deathsat20',
        'opp_killsat20', 'opp_assistsat20', 'opp_deathsat20',
        'goldat25', 'xpat25', 'csat25', 'opp_goldat25', 'opp_xpat25', 'opp_csat25',
        'golddiffat25', 'xpdiffat25', 'csdiffat25',
        'killsat25', 'assistsat25', 'deathsat25',
        'opp_killsat25', 'opp_assistsat25', 'opp_deathsat25',
        'pick1', 'pick2', 'pick3', 'pick4', 'pick5', 'champion'  # Include picks
    ]
    
    # Filter to only columns that exist
    sum_cols = [c for c in sum_cols if c in df_raw.columns]
    first_cols = [c for c in first_cols if c in df_raw.columns]
    
    # Group by gameid and teamid
    agg_dict = {}
    
    # Sum columns
    for col in sum_cols:
        agg_dict[col] = 'sum'
    
    # First columns
    for col in first_cols:
        if col not in ['gameid', 'teamid']:
            agg_dict[col] = 'first'
    
    team_df = df_raw.groupby(['gameid', 'teamid'], as_index=False).agg(agg_dict)
    
    return team_df


def build_game_level(team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build game-level DataFrame with blue vs red features.
    
    Each row represents one game with:
    - Game identifiers (gameid, date, league, patch, etc.)
    - Blue and red team names/IDs
    - Blue_* and red_* prefixed features
    - Difference features (blue - red)
    
    Args:
        team_df: Team-level DataFrame
        
    Returns:
        Game-level DataFrame
    """
    # Pivot to get blue and red side data
    blue_df = team_df[team_df['side'] == 'Blue'].copy()
    red_df = team_df[team_df['side'] == 'Red'].copy()
    
    # Rename columns with blue_/red_ prefix
    feature_cols = [
        'teamkills', 'teamdeaths', 'kills', 'deaths', 'assists',
        'team kpm', 'ckpm',
        'goldat10', 'xpat10', 'csat10', 'opp_goldat10', 'opp_xpat10', 'opp_csat10',
        'golddiffat10', 'xpdiffat10', 'csdiffat10',
        'killsat10', 'assistsat10', 'deathsat10',
        'opp_killsat10', 'opp_assistsat10', 'opp_deathsat10',
        'goldat15', 'xpat15', 'csat15', 'opp_goldat15', 'opp_xpat15', 'opp_csat15',
        'golddiffat15', 'xpdiffat15', 'csdiffat15',
        'killsat15', 'assistsat15', 'deathsat15',
        'opp_killsat15', 'opp_assistsat15', 'opp_deathsat15',
        'goldat20', 'xpat20', 'csat20', 'opp_goldat20', 'opp_xpat20', 'opp_csat20',
        'golddiffat20', 'xpdiffat20', 'csdiffat20',
        'killsat20', 'assistsat20', 'deathsat20',
        'opp_killsat20', 'opp_assistsat20', 'opp_deathsat20',
        'goldat25', 'xpat25', 'csat25', 'opp_goldat25', 'opp_xpat25', 'opp_csat25',
        'golddiffat25', 'xpdiffat25', 'csdiffat25',
        'killsat25', 'assistsat25', 'deathsat25',
        'opp_killsat25', 'opp_assistsat25', 'opp_deathsat25',
        'firstblood', 'firstdragon', 'firstherald', 'firsttower',
        'gamelength', 'result',
        'pick1', 'pick2', 'pick3', 'pick4', 'pick5',  # Original picks (draft order)
        'top_champ', 'jng_champ', 'mid_champ', 'bot_champ', 'sup_champ'  # Role-based picks
    ]
    
    # Filter to existing columns
    feature_cols = [c for c in feature_cols if c in team_df.columns]
    
    # Merge blue and red
    blue_renamed = {col: f'blue_{col}' for col in feature_cols + ['teamid', 'teamname']}
    red_renamed = {col: f'red_{col}' for col in feature_cols + ['teamid', 'teamname']}
    
    blue_subset = blue_df[['gameid'] + feature_cols + ['teamid', 'teamname']].copy()
    red_subset = red_df[['gameid'] + feature_cols + ['teamid', 'teamname']].copy()
    
    blue_subset.rename(columns=blue_renamed, inplace=True)
    red_subset.rename(columns=red_renamed, inplace=True)
    
    # Merge on gameid (inner join ensures we only keep games with both teams)
    games_df = blue_subset.merge(red_subset, on='gameid', how='inner')
    
    # Add game metadata from blue side (should be same for both)
    meta_cols = ['date', 'league', 'year', 'split', 'playoffs', 'patch']
    meta_cols = [c for c in meta_cols if c in blue_df.columns]
    if meta_cols:
        meta_df = blue_df[['gameid'] + meta_cols].drop_duplicates('gameid')
        games_df = games_df.merge(meta_df, on='gameid', how='left')
    
    # Also add blue/red league if available
    if 'league' in blue_df.columns:
        blue_league_df = blue_df[['gameid', 'league']].rename(columns={'league': 'blue_league'})
        games_df = games_df.merge(blue_league_df, on='gameid', how='left')
    if 'league' in red_df.columns:
        red_league_df = red_df[['gameid', 'league']].rename(columns={'league': 'red_league'})
        games_df = games_df.merge(red_league_df, on='gameid', how='left')
    
    # Calculate difference features (only for numeric columns)
    diff_features = []
    # Columns to skip (non-numeric or shouldn't be differenced)
    skip_cols = ['gamelength', 'result', 'pick1', 'pick2', 'pick3', 'pick4', 'pick5', 'champion']
    
    for col in feature_cols:
        if col not in skip_cols:
            blue_col = f'blue_{col}'
            red_col = f'red_{col}'
            if blue_col in games_df.columns and red_col in games_df.columns:
                # Check if columns are numeric before computing difference
                if pd.api.types.is_numeric_dtype(games_df[blue_col]) and pd.api.types.is_numeric_dtype(games_df[red_col]):
                    diff_col = f'{col}_diff'
                    games_df[diff_col] = games_df[blue_col] - games_df[red_col]
                    diff_features.append(diff_col)
    
    # Sort by date
    if 'date' in games_df.columns:
        games_df = games_df.sort_values('date').reset_index(drop=True)
    
    return games_df


def build_first_kills_labels(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add labels for first to 5 kills and first to 10 kills.
    
    Uses killsat10, killsat15, killsat20, killsat25 to approximate
    which team reached the threshold first.
    
    Args:
        games_df: Game-level DataFrame with blue_*/red_* kill columns
        
    Returns:
        DataFrame with first5_blue and first10_blue labels added
    """
    games_df = games_df.copy()
    
    # Check which kill columns exist
    time_buckets = [10, 15, 20, 25]
    available_cols = []
    for t in time_buckets:
        blue_col = f'blue_killsat{t}'
        red_col = f'red_killsat{t}'
        if blue_col in games_df.columns and red_col in games_df.columns:
            available_cols.append(t)
    
    if len(available_cols) == 0:
        print("Warning: No killsat* columns found. Available columns:")
        kill_cols = [c for c in games_df.columns if 'kill' in c.lower()]
        print(f"  Kill-related columns: {kill_cols[:10]}...")
        # Try alternative: use teamkills if available
        if 'blue_teamkills' in games_df.columns and 'red_teamkills' in games_df.columns:
            print("  Using teamkills as fallback...")
            # Use teamkills at game end as approximation (less ideal)
            games_df['first5_blue'] = np.where(
                (games_df['blue_teamkills'] >= 5) & (games_df['red_teamkills'] < 5), 1,
                np.where(
                    (games_df['red_teamkills'] >= 5) & (games_df['blue_teamkills'] < 5), 0,
                    np.nan
                )
            )
            games_df['first10_blue'] = np.where(
                (games_df['blue_teamkills'] >= 10) & (games_df['red_teamkills'] < 10), 1,
                np.where(
                    (games_df['red_teamkills'] >= 10) & (games_df['blue_teamkills'] < 10), 0,
                    np.nan
                )
            )
            return games_df
    
    print(f"Using time buckets: {available_cols}")
    
    # Initialize labels
    games_df['first5_blue'] = np.nan
    games_df['first10_blue'] = np.nan
    
    # Use vectorized operations for better performance
    for t in available_cols:
        blue_kills_col = f'blue_killsat{t}'
        red_kills_col = f'red_killsat{t}'
        
        if blue_kills_col not in games_df.columns or red_kills_col not in games_df.columns:
            continue
        
        blue_kills = pd.to_numeric(games_df[blue_kills_col], errors='coerce')
        red_kills = pd.to_numeric(games_df[red_kills_col], errors='coerce')
        
        # First to 5 kills (only set if not already set)
        mask_first5 = games_df['first5_blue'].isna()
        mask_blue_wins_5 = mask_first5 & (blue_kills >= 5) & (red_kills < 5)
        mask_red_wins_5 = mask_first5 & (red_kills >= 5) & (blue_kills < 5)
        
        games_df.loc[mask_blue_wins_5, 'first5_blue'] = 1
        games_df.loc[mask_red_wins_5, 'first5_blue'] = 0
        
        # First to 10 kills (only set if not already set)
        mask_first10 = games_df['first10_blue'].isna()
        mask_blue_wins_10 = mask_first10 & (blue_kills >= 10) & (red_kills < 10)
        mask_red_wins_10 = mask_first10 & (red_kills >= 10) & (blue_kills < 10)
        
        games_df.loc[mask_blue_wins_10, 'first10_blue'] = 1
        games_df.loc[mask_red_wins_10, 'first10_blue'] = 0
    
    # Report label statistics
    n_first5 = games_df['first5_blue'].notna().sum()
    n_first10 = games_df['first10_blue'].notna().sum()
    print(f"Label statistics:")
    print(f"  First5 labels: {n_first5} valid out of {len(games_df)} games ({100*n_first5/len(games_df):.1f}%)")
    print(f"  First10 labels: {n_first10} valid out of {len(games_df)} games ({100*n_first10/len(games_df):.1f}%)")
    
    if n_first5 > 0:
        print(f"  First5 distribution: Blue={games_df['first5_blue'].sum():.0f}, Red={n_first5 - games_df['first5_blue'].sum():.0f}")
    if n_first10 > 0:
        print(f"  First10 distribution: Blue={games_df['first10_blue'].sum():.0f}, Red={n_first10 - games_df['first10_blue'].sum():.0f}")
    
    return games_df

