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

def _get_training_config(use_gpu=True):
    """
    Get training configuration based on the detected environment.
    Returns different parameters for Mac (MPS) vs CUDA VM.
    
    Returns:
        dict: Configuration with epochs, batch_size, learning_rate, d_model, d_ff, n_heads, e_layers, d_layers
    """
    gpu_type = _detect_gpu_type() if use_gpu else None
    
    if gpu_type == 'cuda':
        # Stronger parameters for CUDA VM
        return {
            'epochs': 50,
            'batch_size': 64,
            'learning_rate': 0.0001,
            'd_model': 128,
            'd_ff': 256,
            'n_heads': 16,
            'e_layers': 4,
            'd_layers': 2,
        }
    elif gpu_type == 'mps':
        # Faster parameters for Mac (MPS)
        return {
            'epochs': 10,
            'batch_size': 16,
            'learning_rate': 0.0001,
            'd_model': 64,
            'd_ff': 128,
            'n_heads': 8,
            'e_layers': 2,
            'd_layers': 1,
        }
    else:
        # CPU fallback - use lighter parameters
        return {
            'epochs': 5,
            'batch_size': 8,
            'learning_rate': 0.0001,
            'd_model': 32,
            'd_ff': 64,
            'n_heads': 4,
            'e_layers': 1,
            'd_layers': 1,
        }

def train_transformer(
    data_path="engineered_features.csv",
    model="iTransformer",  # Options: iTransformer, Transformer, PatchTST
    seq_len=1440,  # 1 day of minutely data (24*60)
    pred_len=60,   # Predict 1 hour ahead (60 minutes)
    epochs=None,  # Auto-detect based on environment if None
    batch_size=None,  # Auto-detect based on environment if None
    learning_rate=None,  # Auto-detect based on environment if None
    use_gpu=True,
    d_model=None,  # Auto-detect based on environment if None
    d_ff=None,  # Auto-detect based on environment if None
    n_heads=None,  # Auto-detect based on environment if None
    e_layers=None,  # Auto-detect based on environment if None
    d_layers=None,  # Auto-detect based on environment if None
):
    """
    Train a transformer model on trading data.
    
    Args:
        data_path: CSV filename in dataset/trading/
        model: Model name (iTransformer, Transformer, PatchTST, etc.)
        seq_len: Input sequence length (history to look at)
        pred_len: Prediction length (how far to predict)
        epochs: Number of training epochs (None = auto-detect based on environment)
        batch_size: Batch size (None = auto-detect based on environment)
        learning_rate: Learning rate (None = auto-detect based on environment)
        use_gpu: Whether to use GPU
        d_model: Model dimension (None = auto-detect based on environment)
        d_ff: Feed-forward dimension (None = auto-detect based on environment)
        n_heads: Number of attention heads (None = auto-detect based on environment)
        e_layers: Encoder layers (None = auto-detect based on environment)
        d_layers: Decoder layers (None = auto-detect based on environment)
    """
    # Get environment-based configuration
    config = _get_training_config(use_gpu)
    
    # Use provided values or fall back to environment-based defaults
    epochs = epochs if epochs is not None else config['epochs']
    batch_size = batch_size if batch_size is not None else config['batch_size']
    learning_rate = learning_rate if learning_rate is not None else config['learning_rate']
    d_model = d_model if d_model is not None else config['d_model']
    d_ff = d_ff if d_ff is not None else config['d_ff']
    n_heads = n_heads if n_heads is not None else config['n_heads']
    e_layers = e_layers if e_layers is not None else config['e_layers']
    d_layers = d_layers if d_layers is not None else config['d_layers']
    
    # Detect GPU type for logging
    gpu_type = _detect_gpu_type() if use_gpu else None
    env_name = "CUDA VM" if gpu_type == 'cuda' else ("Mac (MPS)" if gpu_type == 'mps' else "CPU")
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
    
    print(f"\n{'='*60}")
    print(f"Environment Detection: {env_name}")
    print(f"{'='*60}")
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
        "--e_layers", str(e_layers),  # Encoder layers (auto-detected)
        "--d_layers", str(d_layers),  # Decoder layers (auto-detected)
        "--enc_in", str(num_features),
        "--dec_in", str(num_features),
        "--c_out", str(num_features),
        "--d_model", str(d_model),  # Model dimension (auto-detected)
        "--d_ff", str(d_ff),    # Feed-forward dimension (auto-detected)
        "--n_heads", str(n_heads),   # Number of attention heads (auto-detected)
        "--des", "Trading_Transformer",
        "--itr", "1",       # Number of experiments
        "--train_epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--learning_rate", str(learning_rate),
        "--patience", "5",  # Early stopping patience
        "--freq", "t",      # 't' for minutely data
    ]
    
    # Detect GPU type and configure accordingly
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
    print(f"Training Configuration ({env_name}):")
    print(f"  Model: {model}")
    print(f"  Data: {data_path}")
    print(f"  Sequence length: {seq_len} ({seq_len/60:.1f} hours)")
    print(f"  Prediction length: {pred_len} ({pred_len} minutes)")
    print(f"  Features: {num_features}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Model dim (d_model): {d_model}")
    print(f"  FF dim (d_ff): {d_ff}")
    print(f"  Attention heads: {n_heads}")
    print(f"  Encoder layers: {e_layers}")
    print(f"  Decoder layers: {d_layers}")
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
    parser.add_argument("--epochs", type=int, default=None,
                       help="Training epochs (default: auto-detect based on environment - 10 for Mac, 50 for CUDA)")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size (default: auto-detect based on environment - 16 for Mac, 64 for CUDA)")
    parser.add_argument("--lr", type=float, default=None,
                       help="Learning rate (default: auto-detect based on environment)")
    parser.add_argument("--d_model", type=int, default=None,
                       help="Model dimension (default: auto-detect based on environment - 64 for Mac, 128 for CUDA)")
    parser.add_argument("--d_ff", type=int, default=None,
                       help="Feed-forward dimension (default: auto-detect based on environment - 128 for Mac, 256 for CUDA)")
    parser.add_argument("--n_heads", type=int, default=None,
                       help="Number of attention heads (default: auto-detect based on environment - 8 for Mac, 16 for CUDA)")
    parser.add_argument("--e_layers", type=int, default=None,
                       help="Encoder layers (default: auto-detect based on environment - 2 for Mac, 4 for CUDA)")
    parser.add_argument("--d_layers", type=int, default=None,
                       help="Decoder layers (default: auto-detect based on environment - 1 for Mac, 2 for CUDA)")
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
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_layers=args.d_layers,
    )

