#!/usr/bin/env python3
"""
Inference script for using a trained model in a trading bot.
This loads a saved model and makes predictions on new data.
"""

import torch
import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime
from utils.timefeatures import time_features

# Model and args imports
from models.iTransformer import Model
from utils.tools import dotdict


class TradingBotPredictor:
    """
    A class to load and use a trained model for trading predictions.
    """
    
    def __init__(
        self,
        checkpoint_path,
        seq_len=1440,
        pred_len=60,
        num_features=96,
        d_model=64,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=128,
        freq='t',  # 't' for minutely data
        use_gpu=True,
        gpu_type='mps'
    ):
        """
        Initialize the predictor with model configuration.
        
        Args:
            checkpoint_path: Path to the saved checkpoint.pth file
            seq_len: Input sequence length (history window)
            pred_len: Prediction length (how far ahead to predict)
            num_features: Number of features in the data
            d_model, n_heads, e_layers, d_layers, d_ff: Model architecture parameters
            freq: Time frequency ('t' for minutes, 'h' for hours, etc.)
            use_gpu: Whether to use GPU
            gpu_type: 'mps' for Mac, 'cuda' for NVIDIA, 'cpu' for CPU
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_features = num_features
        self.freq = freq
        
        # Setup device
        if use_gpu:
            if gpu_type == 'mps' and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif gpu_type == 'cuda' and torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            else:
                self.device = torch.device("cpu")
                use_gpu = False
        else:
            self.device = torch.device("cpu")
        
        print(f"Using device: {self.device}")
        
        # Create model args
        args = dotdict({
            'task_name': 'long_term_forecast',
            'seq_len': seq_len,
            'label_len': seq_len // 2,  # Usually half of seq_len
            'pred_len': pred_len,
            'enc_in': num_features,
            'dec_in': num_features,
            'c_out': num_features,
            'd_model': d_model,
            'n_heads': n_heads,
            'e_layers': e_layers,
            'd_layers': d_layers,
            'd_ff': d_ff,
            'factor': 1,
            'dropout': 0.1,
            'embed': 'timeF',
            'freq': freq,
            'activation': 'gelu',
            'output_attention': False,
            'features': 'M',  # Multivariate
            'expand': 2,
            'd_conv': 4,
        })
        
        # Create and load model
        self.model = Model(args).float().to(self.device)
        
        # Load checkpoint
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"✓ Loaded model from {checkpoint_path}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.model.eval()
        print("Model ready for inference!")
    
    def prepare_data(
        self,
        data_df,
        date_column='date',
        feature_columns=None,
        target_column='target'
    ):
        """
        Prepare data for prediction. Assumes data_df has the same features as training data.
        
        Args:
            data_df: DataFrame with historical data (must have at least seq_len rows)
            date_column: Name of the date column
            feature_columns: List of feature column names (if None, auto-detect)
            target_column: Name of target column (for reference, not used in prediction)
        
        Returns:
            batch_x: Tensor of shape (1, seq_len, num_features) ready for model
            batch_x_mark: Time features tensor
            batch_y_mark: Time features for prediction window
        """
        # Auto-detect feature columns if not provided
        if feature_columns is None:
            feature_columns = [c for c in data_df.columns 
                             if c not in [date_column, target_column]]
        
        # Ensure we have enough data
        if len(data_df) < self.seq_len:
            raise ValueError(f"Need at least {self.seq_len} rows, got {len(data_df)}")
        
        # Take the most recent seq_len rows
        recent_data = data_df.tail(self.seq_len).copy()
        
        # Extract features
        features = recent_data[feature_columns].values.astype(np.float32)
        
        # Handle NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Extract dates
        if date_column in recent_data.columns:
            dates_raw = recent_data[date_column].values  # Get numpy array
            # Convert to DatetimeIndex (time_features expects DatetimeIndex, not Series)
            dates = pd.DatetimeIndex(pd.to_datetime(dates_raw))
        else:
            # Create dummy dates if no date column
            dates = pd.date_range(end=datetime.now(), periods=self.seq_len, freq='1min')
        
        # Create time features (expects DatetimeIndex)
        time_feat = time_features(dates, freq=self.freq)
        time_feat = time_feat.transpose(1, 0)  # (seq_len, num_time_features)
        
        # Create prediction window time features
        last_date = dates[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(minutes=1),
            periods=self.pred_len,
            freq='1min'
        )
        future_time_feat = time_features(future_dates, freq=self.freq)
        future_time_feat = future_time_feat.transpose(1, 0)
        
        # Convert to tensors
        batch_x = torch.from_numpy(features).float().unsqueeze(0).to(self.device)  # (1, seq_len, features)
        batch_x_mark = torch.from_numpy(time_feat).float().unsqueeze(0).to(self.device)  # (1, seq_len, time_feat)
        
        # For decoder (not used by iTransformer but required for interface)
        batch_y_mark = torch.from_numpy(future_time_feat).float().unsqueeze(0).to(self.device)  # (1, pred_len, time_feat)
        
        return batch_x, batch_x_mark, batch_y_mark, feature_columns
    
    def predict(
        self,
        data_df,
        date_column='date',
        feature_columns=None,
        target_column='target',
        inverse_transform=None
    ):
        """
        Make predictions on new data.
        
        Args:
            data_df: DataFrame with historical data
            date_column: Name of the date column
            feature_columns: List of feature column names
            target_column: Name of target column
            inverse_transform: Optional scaler for inverse transformation
        
        Returns:
            predictions: numpy array of shape (pred_len, num_features)
            future_dates: DatetimeIndex for prediction timestamps
            feature_columns: List of feature column names used
        """
        # Prepare data
        batch_x, batch_x_mark, batch_y_mark, feature_cols = self.prepare_data(
            data_df, date_column, feature_columns, target_column
        )
        
        # Create decoder input (zeros, not used by iTransformer)
        batch_y = torch.zeros(1, self.seq_len // 2 + self.pred_len, self.num_features).float().to(self.device)
        dec_inp = torch.zeros(1, self.pred_len, self.num_features).float().to(self.device)
        dec_inp = torch.cat([batch_y[:, :self.seq_len // 2, :], dec_inp], dim=1).float().to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            # iTransformer returns predictions directly
            predictions = outputs[:, -self.pred_len:, :].cpu().numpy()  # (1, pred_len, features)
        
        predictions = predictions[0]  # Remove batch dimension: (pred_len, features)
        
        # Inverse transform if scaler provided
        if inverse_transform is not None:
            predictions = inverse_transform(predictions)
        
        # Create future dates
        if date_column in data_df.columns:
            last_date = pd.to_datetime(data_df[date_column].iloc[-1])
            if isinstance(last_date, pd.Series):
                last_date = last_date.iloc[0]
        else:
            last_date = datetime.now()
        
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(minutes=1),
            periods=self.pred_len,
            freq='1min'
        )
        
        return predictions, future_dates, feature_cols
    
    def predict_target_only(
        self,
        data_df,
        date_column='date',
        target_column='target',
        feature_columns=None
    ):
        """
        Simplified method that only returns predictions for the target column.
        Useful when you only care about predicting one value (e.g., price).
        
        Args:
            data_df: DataFrame with historical data
            date_column: Name of the date column
            target_column: Name of target column to predict
            feature_columns: List of feature column names (if None, auto-detect)
        
        Returns:
            target_predictions: numpy array of shape (pred_len,) with target predictions
            future_dates: DatetimeIndex for prediction timestamps
        """
        predictions, future_dates, feature_cols = self.predict(
            data_df, date_column, feature_columns, target_column
        )
        
        # Find target column index
        if feature_columns is None:
            feature_columns = [c for c in data_df.columns 
                             if c not in [date_column, target_column]]
        
        # If target is in features, find its index
        if target_column in feature_columns:
            target_idx = feature_columns.index(target_column)
            target_predictions = predictions[:, target_idx]
        else:
            # Otherwise, return the last feature (common pattern)
            target_predictions = predictions[:, -1]
        
        return target_predictions, future_dates


def load_predictor_from_checkpoint(
    checkpoint_dir_name="long_term_forecast_engineered_features_1440_60_iTransformer_custom_ftM_sl1440_ll720_pl60_dm64_nh8_el2_dl1_df128_expand2_dc4_fc1_ebtimeF_dtTrue_Trading_Transformer_0",
    data_path="engineered_features.csv",
    base_path="./checkpoints"
):
    """
    Convenience function to load predictor from checkpoint directory.
    Auto-detects parameters from checkpoint directory name.
    
    Args:
        checkpoint_dir_name: Name of checkpoint directory
        data_path: Path to training data CSV (to get feature count)
        base_path: Base path for checkpoints
    
    Returns:
        TradingBotPredictor instance
    """
    checkpoint_path = os.path.join(base_path, checkpoint_dir_name, "checkpoint.pth")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Parse parameters from checkpoint directory name
    parts = checkpoint_dir_name.split('_')
    
    # Extract parameters (matching the setting format)
    seq_len = int([p for p in parts if p.startswith('sl')][0][2:]) if any(p.startswith('sl') for p in parts) else 1440
    pred_len = int([p for p in parts if p.startswith('pl')][0][2:]) if any(p.startswith('pl') for p in parts) else 60
    d_model = int([p for p in parts if p.startswith('dm')][0][2:]) if any(p.startswith('dm') for p in parts) else 64
    n_heads = int([p for p in parts if p.startswith('nh')][0][2:]) if any(p.startswith('nh') for p in parts) else 8
    e_layers = int([p for p in parts if p.startswith('el')][0][2:]) if any(p.startswith('el') for p in parts) else 2
    d_layers = int([p for p in parts if p.startswith('dl')][0][2:]) if any(p.startswith('dl') for p in parts) else 1
    d_ff = int([p for p in parts if p.startswith('df')][0][2:]) if any(p.startswith('df') for p in parts) else 128
    
    # Count features from data
    full_data_path = data_path
    if not os.path.isabs(data_path):
        # Try different common paths
        possible_paths = [
            os.path.join(".", "dataset", "trading", data_path),
            os.path.join(".", data_path),
            data_path
        ]
        for path in possible_paths:
            if os.path.exists(path):
                full_data_path = path
                break
    
    if os.path.exists(full_data_path):
        df = pd.read_csv(full_data_path, nrows=1)
        num_features = len([c for c in df.columns if c not in ['date', 'target']])
    else:
        print(f"Warning: Data file not found at {full_data_path}, using default num_features=96")
        num_features = 96  # Default fallback
    
    # Detect GPU
    use_gpu = True
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        gpu_type = 'mps'
    elif torch.cuda.is_available():
        gpu_type = 'cuda'
    else:
        gpu_type = 'cpu'
        use_gpu = False
    
    return TradingBotPredictor(
        checkpoint_path=checkpoint_path,
        seq_len=seq_len,
        pred_len=pred_len,
        num_features=num_features,
        d_model=d_model,
        n_heads=n_heads,
        e_layers=e_layers,
        d_layers=d_layers,
        d_ff=d_ff,
        use_gpu=use_gpu,
        gpu_type=gpu_type
    )


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("Trading Bot Model Inference Example")
    print("="*60)
    
    # Load predictor
    try:
        predictor = load_predictor_from_checkpoint(
            checkpoint_dir_name="long_term_forecast_engineered_features_1440_60_iTransformer_custom_ftM_sl1440_ll720_pl60_dm64_nh8_el2_dl1_df128_expand2_dc4_fc1_ebtimeF_dtTrue_Trading_Transformer_0",
            data_path="./dataset/trading/engineered_features.csv"
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Load recent data (example: load last 2000 rows)
    data_path = "./dataset/trading/engineered_features.csv"
    if os.path.exists(data_path):
        print(f"\nLoading data from {data_path}...")
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} rows")
        
        # Make prediction
        print("\nMaking prediction...")
        predictions, future_dates, features = predictor.predict(df)
        
        print(f"\n✓ Prediction complete!")
        print(f"  Shape: {predictions.shape}")
        print(f"  Features: {len(features)}")
        print(f"\nNext {len(future_dates)} predictions:")
        print(f"  First prediction date: {future_dates[0]}")
        print(f"  Last prediction date: {future_dates[-1]}")
        
        # Example: Get target predictions only
        if 'target' in df.columns:
            target_predictions, _ = predictor.predict_target_only(
                df,
                target_column='target'  # Adjust based on your target column
            )
            
            print(f"\nTarget predictions (next {len(target_predictions)} steps):")
            print(f"  Mean: {target_predictions.mean():.4f}")
            print(f"  Min: {target_predictions.min():.4f}")
            print(f"  Max: {target_predictions.max():.4f}")
            print(f"  Latest: {target_predictions[-1]:.4f}")
        
    else:
        print(f"Data file not found: {data_path}")
        print("\nExample usage in your trading bot:")
        print("""
        from trading_bot_inference import load_predictor_from_checkpoint
        
        # Load model once at startup
        predictor = load_predictor_from_checkpoint()
        
        # In your trading loop:
        # 1. Get latest data (last seq_len rows)
        recent_data = get_latest_market_data(seq_len=1440)
        
        # 2. Make prediction
        predictions, future_dates, features = predictor.predict(recent_data)
        
        # 3. Get target price predictions
        target_predictions, dates = predictor.predict_target_only(
            recent_data,
            target_column='BTCUSD_close'  # or your target
        )
        
        # 4. Make trading decision based on predictions
        next_price = target_predictions[-1]  # Price 60 minutes ahead
        current_price = recent_data['BTCUSD_close'].iloc[-1]
        
        if next_price > current_price * 1.01:  # 1% higher
            execute_buy()
        elif next_price < current_price * 0.99:  # 1% lower
            execute_sell()
        """)
