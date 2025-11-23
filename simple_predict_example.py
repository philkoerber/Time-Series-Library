#!/usr/bin/env python3
"""
Simple example: Sequence In → Sequence Out
Shows exactly how the model works - give it a sequence, get a predicted sequence.
"""

import torch
import numpy as np
import pandas as pd
import os
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from trading_bot_inference import load_predictor_from_checkpoint
except ImportError:
    print("Note: trading_bot_inference module not found. Showing explanation only.")
    load_predictor_from_checkpoint = None

print("="*70)
print("SIMPLE SEQUENCE IN → SEQUENCE OUT EXAMPLE")
print("="*70)

# ============================================================================
# WHERE ARE YOUR MODELS?
# ============================================================================
print("\n📍 YOUR MODELS ARE HERE:")
checkpoint_path = "./checkpoints/long_term_forecast_engineered_features_1440_60_iTransformer_custom_ftM_sl1440_ll720_pl60_dm64_nh8_el2_dl1_df128_expand2_dc4_fc1_ebtimeF_dtTrue_Trading_Transformer_0/checkpoint.pth"
print(f"   {checkpoint_path}")
print(f"   Size: {os.path.getsize(checkpoint_path) / 1024:.1f} KB")

# List all checkpoints
print("\n📦 ALL SAVED MODELS:")
checkpoint_dir = "./checkpoints"
if os.path.exists(checkpoint_dir):
    for item in os.listdir(checkpoint_dir):
        checkpoint_file = os.path.join(checkpoint_dir, item, "checkpoint.pth")
        if os.path.exists(checkpoint_file):
            size = os.path.getsize(checkpoint_file) / 1024
            print(f"   ✓ {item[:60]}... ({size:.1f} KB)")

# ============================================================================
# HOW IT WORKS: SEQUENCE IN → SEQUENCE OUT
# ============================================================================
print("\n" + "="*70)
print("HOW IT WORKS:")
print("="*70)

print("""
┌─────────────────────────────────────────────────────────────────────┐
│                        YOUR MODEL                                    │
│                                                                      │
│  INPUT SEQUENCE          →      OUTPUT SEQUENCE                      │
│  ────────────────────             ─────────────────                  │
│                                                                      │
│  Last 1440 time steps    →      Next 60 time steps                  │
│  (24 hours of data)              (1 hour predictions)                │
│                                                                      │
│  Shape: (1440, 96)       →      Shape: (60, 96)                     │
│  1440 steps × 96 features         60 steps × 96 features             │
│                                                                      │
│  Example:                        Example:                            │
│  [t-1439, t-1438, ..., t-1, t]  →  [t+1, t+2, ..., t+60]           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

The model learns patterns from the past 1440 steps and predicts 
the next 60 steps for all 96 features (prices, volumes, indicators, etc.)
""")

# ============================================================================
# LOAD MODEL
# ============================================================================
print("\n" + "="*70)
print("LOADING MODEL...")
print("="*70)

if load_predictor_from_checkpoint is None:
    print("⚠️  Cannot load predictor module. Showing explanation only.")
    predictor = None
else:
    try:
        predictor = load_predictor_from_checkpoint()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\nMake sure you've trained a model first!")
        predictor = None

# ============================================================================
# DEMONSTRATION
# ============================================================================
print("\n" + "="*70)
print("DEMONSTRATION: Sequence In → Sequence Out")
print("="*70)

# Load some data
data_path = "./dataset/trading/engineered_features.csv"
if os.path.exists(data_path):
    print(f"\n📊 Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"   Total rows in file: {len(df)}")
    
    # Get the last 1440 rows (this is our INPUT SEQUENCE)
    input_data = df.tail(1440).copy()
    print(f"   ✅ Input sequence: {len(input_data)} rows (last 1440 steps)")
    print(f"   Features: {len([c for c in df.columns if c not in ['date', 'target']])} features")
    
    # Show what we're feeding in
    print("\n" + "-"*70)
    print("INPUT SEQUENCE (Last 1440 time steps):")
    print("-"*70)
    print(f"   First timestamp:  {input_data['date'].iloc[0] if 'date' in input_data.columns else 'N/A'}")
    print(f"   Last timestamp:   {input_data['date'].iloc[-1] if 'date' in input_data.columns else 'N/A'}")
    print(f"   Shape: ({len(input_data)}, {len(input_data.columns) - 2})")  # minus date and target
    
    # Make prediction (this is where the magic happens)
    if predictor is None:
        print("\n⚠️  Skipping prediction (model not loaded)")
        predictions, future_dates, features = None, None, None
    else:
        print("\n" + "-"*70)
        print("🤖 MODEL PROCESSING...")
        print("-"*70)
        
        predictions, future_dates, features = predictor.predict(input_data)
    
    if predictions is not None:
        print("\n" + "-"*70)
        print("OUTPUT SEQUENCE (Next 60 time steps):")
        print("-"*70)
        print(f"   ✅ Predicted sequence: {len(predictions)} steps")
        print(f"   Shape: {predictions.shape} = (60 time steps, 96 features)")
        print(f"   First prediction time:  {future_dates[0]}")
        print(f"   Last prediction time:   {future_dates[-1]}")
        
        # Show what we got back
        print("\n" + "-"*70)
        print("WHAT YOU GET:")
        print("-"*70)
        print(f"   predictions.shape = {predictions.shape}")
        print(f"   predictions[0] = First time step prediction (96 values)")
        print(f"   predictions[-1] = Last time step prediction (96 values)")
        
        # Show actual sample rows from predictions
        print("\n" + "-"*70)
        print("SAMPLE PREDICTION ROWS (showing first 5 time steps):")
        print("-"*70)
        print(f"\n   Time Step 0 (t+1):  {future_dates[0]}")
        print(f"   Values: {predictions[0][:10].tolist()}... (showing first 10 of 96 features)")
        print(f"\n   Time Step 1 (t+2):  {future_dates[1]}")
        print(f"   Values: {predictions[1][:10].tolist()}... (showing first 10 of 96 features)")
        print(f"\n   Time Step 2 (t+3):  {future_dates[2]}")
        print(f"   Values: {predictions[2][:10].tolist()}... (showing first 10 of 96 features)")
        
        # Show feature names if available
        if features and len(features) > 0:
            print("\n" + "-"*70)
            print(f"FEATURE NAMES (showing first 10 of {len(features)}):")
            print("-"*70)
            for i, feat in enumerate(features[:10]):
                print(f"   [{i}] {feat}")
            if len(features) > 10:
                print(f"   ... and {len(features) - 10} more features")
        
        # Create a sample DataFrame for better visualization
        print("\n" + "-"*70)
        print("PREDICTIONS AS TABLE (first 5 time steps, first 10 features):")
        print("-"*70)
        sample_features = features[:10] if features and len(features) >= 10 else [f"Feature_{i}" for i in range(10)]
        sample_df = pd.DataFrame(
            predictions[:5, :10],
            columns=sample_features,
            index=[f"t+{i+1} ({future_dates[i]})" for i in range(5)]
        )
        print(sample_df.to_string())
        
        # Show statistics
        print("\n" + "-"*70)
        print("PREDICTION STATISTICS (all 60 steps, all 96 features):")
        print("-"*70)
        print(f"   Mean: {predictions.mean():.4f}")
        print(f"   Std:  {predictions.std():.4f}")
        print(f"   Min:  {predictions.min():.4f}")
        print(f"   Max:  {predictions.max():.4f}")
        
        # Show OHLCV predictions
        if features and len(features) > 0 and predictor is not None:
            # Find OHLCV columns (assuming BTCUSD as primary asset)
            ohlcv_cols = {}
            for asset in ['BTCUSD', 'ETHUSD', 'LTCUSD']:
                asset_cols = {}
                for col_type in ['open', 'high', 'low', 'close', 'volume']:
                    col_name = f"{asset}_{col_type}"
                    if col_name in features:
                        asset_cols[col_type] = features.index(col_name)
                if asset_cols:
                    ohlcv_cols[asset] = asset_cols
            
            # Show OHLCV predictions for primary asset (BTCUSD if available, else first available)
            primary_asset = None
            if 'BTCUSD' in ohlcv_cols:
                primary_asset = 'BTCUSD'
            elif ohlcv_cols:
                primary_asset = list(ohlcv_cols.keys())[0]
            
            if primary_asset:
                asset_cols = ohlcv_cols[primary_asset]
                print("\n" + "="*70)
                print(f"{primary_asset} OHLCV PREDICTIONS (next 10 time steps):")
                print("="*70)
                
                # Extract OHLCV predictions
                ohlcv_indices = list(asset_cols.values())
                ohlcv_names = [f"{primary_asset}_{col}" for col in ['open', 'high', 'low', 'close', 'volume']]
                
                # Create DataFrame with OHLCV predictions
                ohlcv_data = {}
                for col_type, idx in asset_cols.items():
                    ohlcv_data[col_type.upper()] = predictions[:10, idx]
                
                ohlcv_df = pd.DataFrame(
                    ohlcv_data,
                    index=[f"t+{i+1} ({future_dates[i]})" for i in range(10)]
                )
                
                # Format the display
                print(ohlcv_df.to_string())
                
                # Show some key statistics
                print(f"\n   Current {primary_asset} Close (last known): {input_data[f'{primary_asset}_close'].iloc[-1]:.2f}")
                print(f"   Predicted Close in 1 step (t+1): {predictions[0, asset_cols['close']]:.2f}")
                print(f"   Predicted Close in 60 steps (t+60): {predictions[-1, asset_cols['close']]:.2f}")
                
                # Show price change
                current_close = input_data[f'{primary_asset}_close'].iloc[-1]
                pred_close_1 = predictions[0, asset_cols['close']]
                pred_close_60 = predictions[-1, asset_cols['close']]
                
                change_1 = ((pred_close_1 - current_close) / current_close) * 100
                change_60 = ((pred_close_60 - current_close) / current_close) * 100
                
                print(f"\n   Predicted Change:")
                print(f"   +1 step:  {change_1:+.2f}% ({pred_close_1:.2f})")
                print(f"   +60 steps: {change_60:+.2f}% ({pred_close_60:.2f})")
                
                # Show volume predictions
                print(f"\n   Volume Predictions:")
                print(f"   Current Volume: {input_data[f'{primary_asset}_volume'].iloc[-1]:.2f}")
                print(f"   Predicted Volume (t+1): {predictions[0, asset_cols['volume']]:.2f}")
                print(f"   Predicted Volume (t+60): {predictions[-1, asset_cols['volume']]:.2f}")
                
                # Show high-low range
                print(f"\n   Price Range Predictions (High-Low):")
                for i in range(min(5, len(predictions))):
                    high = predictions[i, asset_cols['high']]
                    low = predictions[i, asset_cols['low']]
                    range_val = high - low
                    print(f"   t+{i+1}: {low:.2f} - {high:.2f} (range: {range_val:.2f})")
                
                if len(ohlcv_cols) > 1:
                    print(f"\n   Note: Also available for {', '.join([k for k in ohlcv_cols.keys() if k != primary_asset])}")
            else:
                print("\n" + "-"*70)
                print("OHLCV COLUMNS NOT FOUND:")
                print("-"*70)
                print("   Could not find OHLCV columns in features.")
                print("   Available feature prefixes:", set([f.split('_')[0] for f in features[:10]]))
    else:
        print("\n⚠️  Skipping prediction output (model not loaded)")
    
    # Visual summary
    print("\n" + "="*70)
    print("SUMMARY:")
    print("="*70)
    print("""
    INPUT:  1440 time steps of historical data
            ↓
    MODEL:  Analyzes patterns and learns relationships
            ↓
    OUTPUT: 60 time steps of future predictions
    
    It's that simple! Give it history, get future predictions.
    """)
    
    # Practical example
    print("\n" + "="*70)
    print("PRACTICAL EXAMPLE:")
    print("="*70)
    print("""
    # Simple usage:
    
    # 1. Get last 1440 rows of your data
    input_data = df.tail(1440)
    
    # 2. Get predictions
    predictions, dates, features = predictor.predict(input_data)
    
    # 3. Use predictions (shape: 60 steps, 96 features)
    next_hour_predictions = predictions  # All 60 steps
    price_in_1_hour = predictions[-1][0]  # Last step, first feature
    
    # That's it! Sequence in → Sequence out.
    """)
    
else:
    print(f"❌ Data file not found: {data_path}")
    print("\nBut here's how it would work:")
    print("""
    # Simple usage:
    input_sequence = df.tail(1440)  # Last 1440 steps
    predictions, dates, features = predictor.predict(input_sequence)
    # predictions shape: (60, 96) - 60 future steps, 96 features each
    """)

print("\n" + "="*70)
print("DONE!")
print("="*70)

