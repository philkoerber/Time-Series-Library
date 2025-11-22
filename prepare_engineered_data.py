#!/usr/bin/env python3
"""
Prepare engineered multi-asset features for training.
Combines features from multiple crypto symbols using feature_engineering.py
"""
import os
import pandas as pd
from data_provider.feature_engineering import engineer_features_from_symbol_data

def prepare_training_data(
    symbols: list[str] = ["BTCUSD", "ETHUSD", "LTCUSD"],
    target_symbol: str = "BTCUSD",
    data_dir: str = "dataset/trading",
    output_path: str = "dataset/trading/engineered_features.csv",
):
    """
    Load OHLCV data, engineer features, and save for training.
    
    Args:
        symbols: List of symbols to include (use filename format, e.g., "BTCUSD" not "BTC/USD")
        target_symbol: Symbol to predict (future log return)
        data_dir: Directory containing CSV files
        output_path: Where to save engineered features
    """
    print(f"Loading data for symbols: {symbols}")
    
    # Load all symbol data
    symbol_data = {}
    for symbol in symbols:
        csv_path = os.path.join(data_dir, f"{symbol}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found: {csv_path}")
        
        print(f"  Loading {symbol}...")
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        
        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"{symbol} missing columns: {missing}")
        
        symbol_data[symbol] = df[required]
        print(f"  {symbol}: {len(df)} rows, {df.index[0]} to {df.index[-1]}")
    
    # Engineer features
    print(f"\nEngineering features (target: {target_symbol})...")
    feature_df, target_series = engineer_features_from_symbol_data(
        symbol_data=symbol_data,
        target_symbol=target_symbol,
        symbols=symbols,
    )
    
    # Combine features and target
    # Convert target (future_log_return) to a column for training
    training_df = feature_df.copy()
    training_df['target'] = target_series
    
    # Reset index to make date a column (required by data loader)
    training_df = training_df.reset_index()
    training_df = training_df.rename(columns={'index': 'date'})
    
    # Ensure 'date' is first column, 'target' is last
    cols = [c for c in training_df.columns if c not in ['date', 'target']]
    training_df = training_df[['date'] + cols + ['target']]
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    training_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Saved engineered features:")
    print(f"  File: {output_path}")
    print(f"  Shape: {training_df.shape}")
    print(f"  Features: {len(cols)}")
    print(f"  Date range: {training_df['date'].min()} to {training_df['date'].max()}")
    print(f"  Sample features: {list(cols[:10])}...")
    
    return training_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare engineered features for training")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSD", "ETHUSD", "LTCUSD"],
                       help="Symbols to include (use filename format, e.g., BTCUSD)")
    parser.add_argument("--target", type=str, default="BTCUSD",
                       help="Target symbol to predict")
    parser.add_argument("--output", type=str, default="dataset/trading/engineered_features.csv",
                       help="Output CSV path")
    
    args = parser.parse_args()
    
    prepare_training_data(
        symbols=args.symbols,
        target_symbol=args.target,
        output_path=args.output,
    )

