"""
Main CLI for first kills prediction pipeline.
"""

import argparse
import pandas as pd
import os
from pathlib import Path

from first_kills.data_processing import (
    load_raw_data,
    build_team_level,
    build_game_level,
    build_first_kills_labels
)
from first_kills.training import train_first_kills_models
from first_kills.predict import predict_first_kills_for_games, predict_from_raw_data


def main():
    parser = argparse.ArgumentParser(
        description='First Kills Prediction Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['first_kills_train', 'first_kills_predict', 'first_kills_backtest'],
        help='Pipeline mode: train, predict, or backtest'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        help='Path to raw Oracle\'s Elixir CSV file'
    )
    
    parser.add_argument(
        '--split-date',
        type=str,
        default='2024-01-01',
        help='Date to split train/test (YYYY-MM-DD). Default: 2024-01-01'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output_first_kills',
        help='Output directory for models and predictions. Default: output_first_kills'
    )
    
    parser.add_argument(
        '--rolling-window',
        type=int,
        default=10,
        help='Window size for rolling team stats. Default: 10'
    )
    
    parser.add_argument(
        '--predictions-file',
        type=str,
        help='Path to save predictions CSV (for predict mode)'
    )
    
    parser.add_argument(
        '--historical-data',
        type=str,
        help='Path to historical data CSV for computing rolling stats (for predict mode)'
    )
    
    parser.add_argument(
        '--predictions',
        type=str,
        help='Path to predictions CSV for backtesting (for backtest mode)'
    )
    
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.65,
        help='Confidence threshold for betting (default: 0.65)'
    )
    
    parser.add_argument(
        '--min-edge',
        type=float,
        default=0.01,
        help='Minimum EV edge to place bet (default: 0.01)'
    )
    
    parser.add_argument(
        '--stake-type',
        type=str,
        default='fractional_kelly',
        choices=['flat', 'kelly', 'fractional_kelly'],
        help='Staking strategy (default: fractional_kelly)'
    )
    
    parser.add_argument(
        '--flat-stake',
        type=float,
        default=1.0,
        help='Flat stake amount (for flat staking, default: 1.0)'
    )
    
    parser.add_argument(
        '--kelly-fraction',
        type=float,
        default=0.25,
        help='Fractional Kelly multiplier (default: 0.25)'
    )
    
    parser.add_argument(
        '--initial-bankroll',
        type=float,
        default=100.0,
        help='Initial bankroll for backtesting (default: 100.0)'
    )
    
    parser.add_argument(
        '--odds-file',
        type=str,
        help='Path to CSV file with odds data (optional, if odds not in predictions CSV)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'first_kills_train':
        # Training mode
        if not args.data:
            parser.error("--data is required for training mode")
        
        print("="*60)
        print("First Kills Prediction - Training Mode")
        print("="*60)
        
        # Load and process data
        print(f"\nLoading data from {args.data}...")
        # Handle comma-separated file paths
        if ',' in args.data:
            data_paths = [p.strip() for p in args.data.split(',')]
            print(f"Loading {len(data_paths)} file(s)...")
            all_dfs = []
            for path in data_paths:
                print(f"  Loading {path}...")
                df = load_raw_data(path)
                all_dfs.append(df)
            df_raw = pd.concat(all_dfs, ignore_index=True)
            print(f"Loaded {len(df_raw)} rows from {len(data_paths)} file(s)")
        else:
            df_raw = load_raw_data(args.data)
            print(f"Loaded {len(df_raw)} rows")
        
        print("\nBuilding team-level data...")
        team_df = build_team_level(df_raw)
        print(f"Team-level: {len(team_df)} team-game rows")
        
        print("\nBuilding game-level data...")
        games_df = build_game_level(team_df)
        print(f"Game-level: {len(games_df)} games")
        
        print("\nBuilding labels...")
        games_df = build_first_kills_labels(games_df)
        
        # Count valid labels
        n_first5 = games_df['first5_blue'].notna().sum()
        n_first10 = games_df['first10_blue'].notna().sum()
        print(f"\nValid first5 labels: {n_first5} out of {len(games_df)} games")
        print(f"Valid first10 labels: {n_first10} out of {len(games_df)} games")
        
        # Check date distribution
        if 'date' in games_df.columns:
            print(f"\nDate distribution:")
            print(f"  Min date: {games_df['date'].min()}")
            print(f"  Max date: {games_df['date'].max()}")
            print(f"  Split date: {args.split_date}")
            n_before_split = (games_df['date'] < pd.to_datetime(args.split_date)).sum()
            n_after_split = (games_df['date'] >= pd.to_datetime(args.split_date)).sum()
            print(f"  Games before split: {n_before_split}")
            print(f"  Games after split: {n_after_split}")
        
        # Train models
        print(f"\nTraining models (split date: {args.split_date})...")
        results = train_first_kills_models(
            games_df,
            split_date=args.split_date,
            output_dir=args.output_dir,
            rolling_window=args.rolling_window,
            team_df=team_df,  # Pass team_df for feature engineering
            player_df=df_raw  # Pass raw player data for lane rolling stats
        )
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(f"\nModels saved to: {args.output_dir}/")
        print(f"  - first5_model.pkl")
        print(f"  - first5_calibrator.pkl")
        print(f"  - first10_model.pkl")
        print(f"  - first10_calibrator.pkl")
        print(f"  - first_kills_predictions.csv")
        
        # Print summary metrics
        print("\nFirst5 Model Metrics:")
        for metric, value in results['first5']['metrics'].items():
            print(f"  {metric}: {value:.4f}")
        
        print("\nFirst10 Model Metrics:")
        for metric, value in results['first10']['metrics'].items():
            print(f"  {metric}: {value:.4f}")
    
    elif args.mode == 'first_kills_predict':
        # Prediction mode
        if not args.data:
            parser.error("--data is required for predict mode")
        
        print("="*60)
        print("First Kills Prediction - Prediction Mode")
        print("="*60)
        
        # Check if models exist
        models_dir = args.output_dir
        first5_model_path = os.path.join(models_dir, 'first5_model.pkl')
        first10_model_path = os.path.join(models_dir, 'first10_model.pkl')
        
        if not os.path.exists(first5_model_path) or not os.path.exists(first10_model_path):
            print(f"\nError: Models not found in {models_dir}/")
            print("Please train models first using --mode first_kills_train")
            return
        
        # Load and process data
        if args.historical_data:
            # Use convenience function that handles historical data
            print(f"\nLoading data from {args.data}...")
            print(f"Using historical data from {args.historical_data} for rolling stats...")
            predictions = predict_from_raw_data(
                raw_data_path=args.data,
                models_dir=models_dir,
                rolling_window=args.rolling_window,
                historical_data_path=args.historical_data
            )
        else:
            # Process data manually
            print(f"\nLoading data from {args.data}...")
            df_raw = load_raw_data(args.data)
            
            print("Building team-level data...")
            team_df = build_team_level(df_raw)
            
            print("Building game-level data...")
            games_df = build_game_level(team_df)
            
            # Add rolling stats (may be incomplete without history)
            print("Adding rolling stats...")
            from first_kills.feature_engineering import add_team_rolling_stats
            games_df = add_team_rolling_stats(games_df, team_df, window=args.rolling_window)
            
            # Make predictions
            print("Making predictions...")
            predictions = predict_first_kills_for_games(
                games_df,
                models_dir=models_dir,
                rolling_window=args.rolling_window
            )
        
        # Display predictions
        print("\n" + "="*60)
        print("Predictions")
        print("="*60)
        print(predictions.to_string(index=False))
        
        # Save predictions
        output_file = args.predictions_file or os.path.join(models_dir, 'upcoming_predictions.csv')
        predictions.to_csv(output_file, index=False)
        print(f"\nPredictions saved to: {output_file}")
    
    elif args.mode == 'first_kills_backtest':
        # Backtesting mode
        from first_kills.betting_first10 import backtest_first10_high_confidence, BetConfig
        
        predictions_path = args.predictions or os.path.join(args.output_dir, 'first_kills_predictions.csv')
        
        if not os.path.exists(predictions_path):
            print(f"\nError: Predictions file not found: {predictions_path}")
            print("Please train models first using --mode first_kills_train")
            return
        
        print("="*60)
        print("First 10 Kills - High-Confidence Betting Backtest")
        print("="*60)
        
        config = BetConfig(
            confidence_threshold=args.confidence_threshold,
            min_edge=args.min_edge,
            stake_type=args.stake_type,
            flat_stake=args.flat_stake,
            kelly_fraction=args.kelly_fraction,
        )
        
        print(f"\nBetting Configuration:")
        print(f"  Confidence threshold: {config.confidence_threshold}")
        print(f"  Minimum edge: {config.min_edge}")
        print(f"  Stake type: {config.stake_type}")
        if config.stake_type == 'flat':
            print(f"  Flat stake: {config.flat_stake}")
        elif config.stake_type == 'fractional_kelly':
            print(f"  Kelly fraction: {config.kelly_fraction}")
        
        print(f"\nRunning backtest on: {predictions_path}")
        
        # Check if odds file is provided
        odds_path = getattr(args, 'odds_file', None)
        
        results = backtest_first10_high_confidence(
            predictions_path=predictions_path,
            config=config,
            initial_bankroll=args.initial_bankroll,
            odds_path=odds_path,
        )
        
        print("\n" + "="*60)
        print("Backtest Results")
        print("="*60)
        print(f"  Initial Bankroll: ${results['initial_bankroll']:.2f}")
        print(f"  Final Bankroll:   ${results['final_bankroll']:.2f}")
        print(f"  Number of bets:   {results['n_bets']}")
        print(f"  Hit rate:         {results['hit_rate']:.3f} ({results['hit_rate']*100:.1f}%)")
        print(f"  Total PnL:        ${results['total_pnl']:.2f}")
        print(f"  ROI:              {results['roi']:.3f} ({results['roi']*100:.1f}%)")
        print(f"  Max Drawdown:     {results['max_drawdown']:.3f} ({results['max_drawdown']*100:.1f}%)")
        
        if results['n_bets'] > 0:
            print("\n" + "="*60)
            print("Performance by Confidence Band")
            print("="*60)
            print(results['conf_summary'])
            
            # Save detailed bets to CSV
            bets_output = os.path.join(args.output_dir, 'first10_bets.csv')
            results['bets_df'].to_csv(bets_output, index=False)
            print(f"\nDetailed bets saved to: {bets_output}")
            
            # Save equity curve
            equity_df = pd.DataFrame({
                'game_index': range(len(results['equity_curve'])),
                'bankroll': results['equity_curve']
            })
            equity_output = os.path.join(args.output_dir, 'first10_equity_curve.csv')
            equity_df.to_csv(equity_output, index=False)
            print(f"Equity curve saved to: {equity_output}")
        else:
            print("\nWarning: No bets were placed. Check:")
            print("  - Confidence threshold may be too high")
            print("  - Minimum edge may be too high")
            print("  - Odds columns may be missing or invalid")


if __name__ == '__main__':
    main()
