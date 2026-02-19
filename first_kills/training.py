"""
Training script for first kills models.
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict
from sklearn.metrics import (
    brier_score_loss, log_loss, roc_auc_score, accuracy_score,
    classification_report, confusion_matrix
)


def find_best_accuracy_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple:
    """
    Find the probability threshold that maximizes accuracy.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class
        
    Returns:
        Tuple of (best_threshold, best_accuracy)
    """
    thresholds = np.linspace(0.3, 0.7, 41)  # step of 0.01
    best_acc = 0.0
    best_thr = 0.5
    
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_thr = t
    
    return best_thr, best_acc

from .models import FirstKillsModel
from .feature_engineering import (
    add_team_rolling_stats,
    add_champion_features,
    add_advanced_rolling_stats,
    get_first5_feature_cols,
    get_first10_feature_cols
)


def time_based_split(games_df: pd.DataFrame, split_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split games into train and test sets based on date.
    
    Args:
        games_df: Game-level DataFrame
        split_date: Date string (YYYY-MM-DD) - test set starts on this date
        
    Returns:
        Tuple of (train_df, test_df)
    """
    games_df = games_df.copy()
    
    if 'date' not in games_df.columns:
        raise ValueError("DataFrame must have 'date' column for time-based split")
    
    games_df['date'] = pd.to_datetime(games_df['date'])
    split_date = pd.to_datetime(split_date)
    
    train_df = games_df[games_df['date'] < split_date].copy()
    test_df = games_df[games_df['date'] >= split_date].copy()
    
    return train_df, test_df


def create_validation_set(train_df: pd.DataFrame, val_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create validation set from training data (last N% chronologically).
    
    Args:
        train_df: Training DataFrame
        val_ratio: Proportion of training data to use for validation
        
    Returns:
        Tuple of (train_core_df, val_df)
    """
    train_df = train_df.sort_values('date').reset_index(drop=True)
    
    n_val = int(len(train_df) * val_ratio)
    train_core_df = train_df.iloc[:-n_val].copy()
    val_df = train_df.iloc[-n_val:].copy()
    
    return train_core_df, val_df


def train_first_kills_models(
    games_df: pd.DataFrame,
    split_date: str,
    output_dir: str = 'output_first_kills',
    rolling_window: int = 10,
    team_df: pd.DataFrame = None,
    player_df: pd.DataFrame = None
) -> Dict:
    """
    Train and evaluate first5 and first10 models.
    
    Args:
        games_df: Game-level DataFrame with labels
        split_date: Date to split train/test
        output_dir: Directory to save models and outputs
        rolling_window: Window size for rolling stats (kept for compatibility, uses 3/5/10)
        team_df: Team-level DataFrame (needed for rolling stats and champion features)
        player_df: Player-level DataFrame (needed for lane player rolling stats)
        
    Returns:
        Dictionary with metrics and predictions
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Add player-level lane rolling stats
    if player_df is not None:
        print("Adding player-level lane rolling statistics...")
        from .feature_engineering import (
            add_player_lane_rolling_stats,
            build_lane_stats_from_players,
            add_lane_player_rolling_features
        )
        player_df_with_rolls = add_player_lane_rolling_stats(player_df, windows=[3, 5, 10])
        lane_stats_df = build_lane_stats_from_players(player_df_with_rolls)
        games_df = add_lane_player_rolling_features(games_df, lane_stats_df)
        print(f"  Added lane player rolling features. Lane stats: {len(lane_stats_df)} rows")
    else:
        print("Warning: player_df not provided, skipping player-level lane rolling stats")
    
    # Add EGI (Early Game Identity) features
    if team_df is not None:
        print("Adding Early Game Identity (EGI) features...")
        from .feature_engineering import add_team_egi_features, add_egi_to_games
        team_df_with_egi = add_team_egi_features(team_df, windows=[5, 10, 20])
        games_df = add_egi_to_games(games_df, team_df_with_egi)
        print(f"  Added EGI features")
    
    # Add region and patch features
    print("Adding region and patch features...")
    from .feature_engineering import add_region_patch_features
    games_df = add_region_patch_features(games_df)
    print(f"  Added region/patch features")
    
    # Add synergy/duo features
    if player_df is not None:
        print("Adding synergy/duo features (jungle-mid, bot lane)...")
        from .feature_engineering import add_synergy_duo_features
        games_df = add_synergy_duo_features(games_df, player_df)
        print(f"  Added synergy features")
    else:
        print("Warning: player_df not provided, skipping synergy features")
    
    # Add champion features
    if team_df is not None:
        print("Adding champion/draft features...")
        games_df = add_champion_features(games_df, team_df)
    
    # Add role-based lane matchup features
    print("Adding role-based lane matchup features...")
    from .feature_engineering import add_role_based_features
    games_df = add_role_based_features(games_df)
    
    # Add rolling stats
    print("Adding advanced rolling team statistics...")
    if team_df is not None:
        games_df = add_team_rolling_stats(games_df, team_df, window=rolling_window)
    else:
        print("Warning: team_df not provided, skipping advanced rolling stats")
    
    # Split into train and test
    print(f"Splitting data at {split_date}...")
    
    # Check date range
    if 'date' in games_df.columns:
        min_date = games_df['date'].min()
        max_date = games_df['date'].max()
        print(f"Data date range: {min_date} to {max_date}")
    
    train_df, test_df = time_based_split(games_df, split_date)
    
    print(f"Train set: {len(train_df)} games")
    print(f"Test set: {len(test_df)} games")
    
    if len(train_df) == 0:
        raise ValueError(f"No training data found! All data is after {split_date}. Try an earlier split date.")
    if len(test_df) == 0:
        print(f"Warning: No test data found! All data is before {split_date}.")
    
    # Create validation set
    train_core_df, val_df = create_validation_set(train_df, val_ratio=0.2)
    print(f"Train core: {len(train_core_df)} games")
    print(f"Validation: {len(val_df)} games")
    
    results = {}
    
    # Train first5 model
    print("\n" + "="*50)
    print("Training first5 model...")
    print("="*50)
    first5_results = _train_single_model(
        train_core_df, val_df, test_df,
        target='first5_blue',
        get_feature_cols=get_first5_feature_cols,
        output_dir=output_dir,
        model_name='first5'
    )
    results['first5'] = first5_results
    
    # Train first10 model
    print("\n" + "="*50)
    print("Training first10 model...")
    print("="*50)
    first10_results = _train_single_model(
        train_core_df, val_df, test_df,
        target='first10_blue',
        get_feature_cols=get_first10_feature_cols,
        output_dir=output_dir,
        model_name='first10'
    )
    results['first10'] = first10_results
    
    # Create combined predictions CSV
    print("\nCreating predictions CSV...")
    predictions_df = _create_predictions_csv(
        test_df, first5_results, first10_results, output_dir
    )
    
    results['predictions_df'] = predictions_df
    
    print(f"\nAll outputs saved to {output_dir}/")
    
    return results


def _train_single_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    get_feature_cols,
    output_dir: str,
    model_name: str
) -> Dict:
    """Train a single model and return results."""
    
    # Drop rows with missing labels
    train_clean = train_df.dropna(subset=[target]).copy()
    val_clean = val_df.dropna(subset=[target]).copy()
    test_clean = test_df.dropna(subset=[target]).copy()
    
    print(f"Training samples: {len(train_clean)} (out of {len(train_df)} total)")
    print(f"Validation samples: {len(val_clean)} (out of {len(val_df)} total)")
    print(f"Test samples: {len(test_clean)} (out of {len(test_df)} total)")
    
    if len(train_clean) == 0:
        # Diagnostic information
        print(f"\nDiagnostics for {target}:")
        print(f"  Total train games: {len(train_df)}")
        print(f"  Games with NaN label: {train_df[target].isna().sum()}")
        print(f"  Games with valid label: {train_df[target].notna().sum()}")
        if train_df[target].notna().sum() > 0:
            print(f"  Label distribution: {train_df[target].value_counts().to_dict()}")
        
        # Check if columns exist
        required_cols = [f'blue_killsat{t}' for t in [10, 15, 20, 25]]
        existing_cols = [c for c in required_cols if c in train_df.columns]
        print(f"  Available kill columns: {existing_cols}")
        
        raise ValueError(f"No training samples with valid {target} labels. Check data processing and split date.")
    
    # Get feature columns
    feature_cols = get_feature_cols(train_clean)
    print(f"Using {len(feature_cols)} features")
    
    # Prepare data
    X_train = train_clean[feature_cols]
    y_train = train_clean[target].astype(int)
    
    X_val = val_clean[feature_cols]
    y_val = val_clean[target].astype(int)
    
    X_test = test_clean[feature_cols]
    y_test = test_clean[target].astype(int)
    
    # Compute class weights for imbalanced data
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    # Convert to list format [weight_class0, weight_class1]
    class_weights_list = [class_weight_dict.get(0, 1.0), class_weight_dict.get(1, 1.0)]
    
    # Train model
    model = FirstKillsModel(target=target, class_weights=class_weights_list)
    model.fit(X_train, y_train, X_val, y_val, feature_cols=feature_cols)
    
    # Evaluate on validation set for threshold optimization (especially for first10)
    y_val_proba = model.predict_proba(X_val)[:, 1]
    
    # Handle empty test set (when all data is before split date)
    has_test_data = len(test_clean) > 0
    
    if has_test_data:
        y_test_proba = model.predict_proba(X_test)[:, 1]
    else:
        # Use validation set for evaluation if no test set
        print("\nWarning: No test data available. Using validation set for evaluation.")
        y_test_proba = y_val_proba.copy()
        y_test = y_val.copy()
        X_test = X_val.copy()
    
    # Find best threshold on validation set (especially important for first10)
    if target == 'first10_blue':
        best_threshold, best_val_acc = find_best_accuracy_threshold(y_val, y_val_proba)
        model.set_threshold(best_threshold)
        print(f"\nThreshold Optimization (First10):")
        print(f"  Best threshold on validation: {best_threshold:.3f}")
        print(f"  Validation accuracy @ {best_threshold:.3f}: {best_val_acc:.4f}")
        val_acc_default = accuracy_score(y_val, (y_val_proba >= 0.5).astype(int))
        print(f"  Validation accuracy @ 0.5: {val_acc_default:.4f}")
    else:
        best_threshold = 0.5
    
    # Predictions with default threshold
    y_pred_default = (y_test_proba >= 0.5).astype(int)
    
    # Predictions with optimized threshold (for first10)
    y_pred_optimized = model.predict(X_test, threshold=best_threshold)
    
    # Calculate metrics
    if has_test_data:
        metrics = {
            'brier_score': brier_score_loss(y_test, y_test_proba),
            'log_loss': log_loss(y_test, y_test_proba),
            'roc_auc': roc_auc_score(y_test, y_test_proba),
            'accuracy_default': accuracy_score(y_test, y_pred_default),
            'accuracy_optimized': accuracy_score(y_test, y_pred_optimized),
            'best_threshold': best_threshold,
        }
        eval_label = "Test Set"
    else:
        metrics = {
            'brier_score': brier_score_loss(y_val, y_val_proba),
            'log_loss': log_loss(y_val, y_val_proba),
            'roc_auc': roc_auc_score(y_val, y_val_proba),
            'accuracy_default': accuracy_score(y_val, (y_val_proba >= 0.5).astype(int)),
            'accuracy_optimized': accuracy_score(y_val, model.predict(X_val, threshold=best_threshold)),
            'best_threshold': best_threshold,
        }
        eval_label = "Validation Set (no test data)"
    
    print(f"\n{eval_label} Metrics:")
    print(f"  Brier Score: {metrics['brier_score']:.4f}")
    print(f"  Log Loss: {metrics['log_loss']:.4f}")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"  Accuracy @ 0.5: {metrics['accuracy_default']:.4f}")
    if target == 'first10_blue':
        print(f"  Accuracy @ {best_threshold:.3f}: {metrics['accuracy_optimized']:.4f}")
    
    # Use optimized predictions for first10, default for first5
    y_pred = y_pred_optimized if target == 'first10_blue' else y_pred_default
    
    if has_test_data:
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Red', 'Blue']))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
    else:
        print("\nClassification Report (Validation Set):")
        print(classification_report(y_val, model.predict(X_val, threshold=best_threshold), target_names=['Red', 'Blue']))
        
        print("\nConfusion Matrix (Validation Set):")
        print(confusion_matrix(y_val, model.predict(X_val, threshold=best_threshold)))
    
    # Save model
    model_path = os.path.join(output_dir, f'{model_name}_model.pkl')
    calibrator_path = os.path.join(output_dir, f'{model_name}_calibrator.pkl')
    model.save(model_path, calibrator_path)
    print(f"\nModel saved to {model_path}")
    
    # Store predictions
    # Use appropriate arrays based on whether test data exists
    if has_test_data:
        results = {
            'model': model,
            'metrics': metrics,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_test_proba,  # Store probabilities
            'test_indices': test_clean.index,
            'feature_cols': feature_cols,
            'best_threshold': best_threshold
        }
    else:
        # Use validation set indices and data when no test set
        results = {
            'model': model,
            'metrics': metrics,
            'y_test': y_val,  # Use validation labels
            'y_pred': y_pred,
            'y_pred_proba': y_val_proba,  # Use validation probabilities
            'test_indices': val_clean.index,  # Use validation indices
            'feature_cols': feature_cols,
            'best_threshold': best_threshold
        }
    
    return results


def _create_predictions_csv(
    test_df: pd.DataFrame,
    first5_results: Dict,
    first10_results: Dict,
    output_dir: str
) -> pd.DataFrame:
    """Create combined predictions CSV."""
    
    # Get test indices used for each model
    first5_indices = first5_results['test_indices']
    first10_indices = first10_results['test_indices']
    
    # Handle empty test set
    if len(test_df) == 0:
        print("\nWarning: No test data available. Skipping predictions CSV creation.")
        return pd.DataFrame()
    
    # Start with all test games
    predictions_df = test_df[['gameid']].copy()
    
    # Add metadata columns
    if 'date' in test_df.columns:
        predictions_df['date'] = test_df['date'].values
    if 'blue_teamname' in test_df.columns:
        predictions_df['blue_team'] = test_df['blue_teamname'].values
    if 'red_teamname' in test_df.columns:
        predictions_df['red_team'] = test_df['red_teamname'].values
    
    # Initialize prediction columns with NaN
    predictions_df['p_first5_blue'] = np.nan
    predictions_df['true_first5_blue'] = np.nan
    predictions_df['pred_first5_blue'] = np.nan
    predictions_df['p_first10_blue'] = np.nan
    predictions_df['true_first10_blue'] = np.nan
    predictions_df['pred_first10_blue'] = np.nan
    
    # Map first5 predictions to gameids
    # Get gameids for the test samples that had valid first5 labels
    first5_gameids = test_df.loc[first5_indices, 'gameid'].values if len(first5_indices) > 0 else []
    
    if len(first5_gameids) > 0:
        first5_map = dict(zip(first5_gameids, first5_results['y_pred_proba']))
        first5_map_true = dict(zip(first5_gameids, first5_results['y_test'].values))
        first5_map_pred = dict(zip(first5_gameids, first5_results['y_pred']))
        
        # Update predictions for games with first5 labels
        mask_first5 = predictions_df['gameid'].isin(first5_gameids)
        predictions_df.loc[mask_first5, 'p_first5_blue'] = predictions_df.loc[mask_first5, 'gameid'].map(first5_map)
        predictions_df.loc[mask_first5, 'true_first5_blue'] = predictions_df.loc[mask_first5, 'gameid'].map(first5_map_true)
        predictions_df.loc[mask_first5, 'pred_first5_blue'] = predictions_df.loc[mask_first5, 'gameid'].map(first5_map_pred)
    
    # Map first10 predictions to gameids
    # Get gameids for the test samples that had valid first10 labels
    first10_gameids = test_df.loc[first10_indices, 'gameid'].values if len(first10_indices) > 0 else []
    
    if len(first10_gameids) > 0:
        first10_map = dict(zip(first10_gameids, first10_results['y_pred_proba']))
        first10_map_true = dict(zip(first10_gameids, first10_results['y_test'].values))
        first10_map_pred = dict(zip(first10_gameids, first10_results['y_pred']))
        
        # Update predictions for games with first10 labels
        mask_first10 = predictions_df['gameid'].isin(first10_gameids)
        predictions_df.loc[mask_first10, 'p_first10_blue'] = predictions_df.loc[mask_first10, 'gameid'].map(first10_map)
        predictions_df.loc[mask_first10, 'true_first10_blue'] = predictions_df.loc[mask_first10, 'gameid'].map(first10_map_true)
        predictions_df.loc[mask_first10, 'pred_first10_blue'] = predictions_df.loc[mask_first10, 'gameid'].map(first10_map_pred)
    
    # Add derived columns
    predictions_df['p_first5_red'] = 1 - predictions_df['p_first5_blue']
    predictions_df['p_first10_red'] = 1 - predictions_df['p_first10_blue']
    
    # Save to CSV
    output_path = os.path.join(output_dir, 'first_kills_predictions.csv')
    predictions_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    print(f"  Total games: {len(predictions_df)}")
    print(f"  Games with first5 predictions: {predictions_df['p_first5_blue'].notna().sum()}")
    print(f"  Games with first10 predictions: {predictions_df['p_first10_blue'].notna().sum()}")
    
    return predictions_df

