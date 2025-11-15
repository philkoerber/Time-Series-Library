#!/usr/bin/env python3
"""
Training script for transformer models on trading data.
"""
import subprocess
import sys
import os
import platform
import torch

def _detect_gpu_type():
    """
    Detect the appropriate GPU type based on the system.
    Returns 'mps' for Mac with MPS, 'cuda' for systems with CUDA, or None for CPU.
    """
    # Check for MPS (Mac)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return 'mps'
    # Check for CUDA
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return None

def train_transformer(
    data_path="engineered_features.csv",
    model="iTransformer",  # Options: iTransformer, Transformer, PatchTST
    seq_len=1440,  # 1 day of minutely data (24*60)
    pred_len=60,   # Predict 1 hour ahead (60 minutes)
    epochs=20,
    batch_size=32,
    learning_rate=0.0001,
    use_gpu=True,
):
    """
    Train a transformer model on trading data.
    
    Args:
        data_path: CSV filename in dataset/trading/
        model: Model name (iTransformer, Transformer, PatchTST, etc.)
        seq_len: Input sequence length (history to look at)
        pred_len: Prediction length (how far to predict)
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        use_gpu: Whether to use GPU
    """
    # Count features (excluding 'date')
    import pandas as pd
    full_path = os.path.join("dataset", "trading", data_path)
    if not os.path.exists(full_path):
        print(f"Error: {full_path} not found!")
        return
    
    df = pd.read_csv(full_path, nrows=1)
    # Exclude 'date' and 'target' columns
    feature_columns = [c for c in df.columns if c not in ['date', 'target']]
    num_features = len(feature_columns)
    
    # Target column (should be 'target' for engineered data)
    target_column = 'target' if 'target' in df.columns else df.columns[-1]
    
    print(f"Detected {num_features} engineered features")
    print(f"Target column: {target_column}")
    if num_features > 20:
        print(f"  (showing first 10: {feature_columns[:10]}...)")
    else:
        print(f"  Features: {feature_columns}")
    
    # Create model_id from data path
    model_id = f"{data_path.replace('.csv', '')}_{seq_len}_{pred_len}"
    
    # Base command
    cmd = [
        "python", "run.py",
        "--task_name", "long_term_forecast",
        "--is_training", "1",
        "--root_path", "./dataset/trading/",
        "--data_path", data_path,
        "--model_id", model_id,
        "--model", model,
        "--data", "custom",
        "--features", "M",  # Multivariate forecasting
        "--target", target_column,  # Set target column (required by data loader)
        "--seq_len", str(seq_len),
        "--label_len", str(seq_len // 2),  # Half of seq_len
        "--pred_len", str(pred_len),
        "--e_layers", "2",  # Encoder layers
        "--d_layers", "1",  # Decoder layers
        "--enc_in", str(num_features),
        "--dec_in", str(num_features),
        "--c_out", str(num_features),
        "--d_model", "64",  # Model dimension
        "--d_ff", "128",    # Feed-forward dimension
        "--n_heads", "8",   # Number of attention heads
        "--des", "Trading_Transformer",
        "--itr", "1",       # Number of experiments
        "--train_epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--learning_rate", str(learning_rate),
        "--patience", "5",  # Early stopping patience
        "--freq", "t",      # 't' for minutely data
    ]
    
    # Detect GPU type and configure accordingly
    gpu_type = _detect_gpu_type() if use_gpu else None
    
    if use_gpu and gpu_type:
        cmd.extend(["--use_gpu", "True", "--gpu", "0", "--gpu_type", gpu_type])
        print(f"Using {gpu_type.upper()} for acceleration")
    else:
        cmd.extend(["--use_gpu", "False"])
        print("Using CPU (GPU not available or disabled)")
    
    # Model-specific parameters
    if model == "TimesNet":
        cmd.extend(["--top_k", "5"])
    elif model == "PatchTST":
        cmd.extend(["--patch_len", "16", "--stride", "8"])
    
    print(f"\n{'='*60}")
    print(f"Training {model} on {data_path}")
    print(f"Sequence length: {seq_len} ({seq_len/60:.1f} hours)")
    print(f"Prediction length: {pred_len} ({pred_len} minutes)")
    print(f"Features: {num_features}")
    print(f"{'='*60}\n")
    
    # Run training
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print(f"\n✅ Training completed successfully!")
        print(f"Check results in: ./results/")
        print(f"Check checkpoints in: ./checkpoints/")
    else:
        print(f"\n❌ Training failed with exit code {result.returncode}")
    
    return result.returncode


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train transformer on trading data")
    parser.add_argument("--data", type=str, default="engineered_features.csv", 
                       help="CSV filename in dataset/trading/ (use engineered_features.csv for multi-asset)")
    parser.add_argument("--model", type=str, default="iTransformer",
                       choices=["iTransformer", "Transformer", "PatchTST", "TimesNet"],
                       help="Model to train")
    parser.add_argument("--seq_len", type=int, default=1440,
                       help="Input sequence length (default: 1440 = 1 day)")
    parser.add_argument("--pred_len", type=int, default=60,
                       help="Prediction length (default: 60 = 1 hour)")
    parser.add_argument("--epochs", type=int, default=20,
                       help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0001,
                       help="Learning rate")
    parser.add_argument("--no_gpu", action="store_true",
                       help="Disable GPU")
    
    args = parser.parse_args()
    
    train_transformer(
        data_path=args.data,
        model=args.model,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_gpu=not args.no_gpu,
    )

