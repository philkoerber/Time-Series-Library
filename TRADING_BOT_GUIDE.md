# Complete Guide: Using Time-Series-Library for Trading Bot Models

## Table of Contents
1. [Overview](#overview)
2. [Installation & Setup](#installation--setup)
3. [Understanding the Library Structure](#understanding-the-library-structure)
4. [Data Preparation for Trading](#data-preparation-for-trading)
5. [Running Your First Model](#running-your-first-model)
6. [Model Selection for Trading](#model-selection-for-trading)
7. [Custom Trading Data Integration](#custom-trading-data-integration)
8. [Training Trading Models](#training-trading-models)
9. [Making Predictions](#making-predictions)
10. [Best Practices for Trading](#best-practices-for-trading)
11. [Advanced Usage](#advanced-usage)

---

## Overview

This Time-Series-Library (TSLib) is a comprehensive deep learning framework for time series analysis. It supports:
- **Long-term forecasting**: Predict future values over extended horizons
- **Short-term forecasting**: Predict immediate future values
- **Anomaly detection**: Identify unusual patterns
- **Imputation**: Fill missing data
- **Classification**: Classify time series patterns

For trading bots, you'll primarily use **forecasting** tasks to predict:
- Stock prices
- Cryptocurrency prices
- Trading volumes
- Technical indicators
- Market trends

---

## Installation & Setup

### 1. Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM

### 2. Install Dependencies

```bash
# Navigate to the repository directory
cd /Users/philipp/Documents/Coding\ Projects/Trading/Time-Series-Library

# Install required packages
pip install -r requirements.txt

# For Mamba model (optional, if you want to use it)
pip install mamba_ssm
```

### 3. Verify Installation

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

---

## Understanding the Library Structure

### Key Components

```
Time-Series-Library/
├── models/          # All available models (TimesNet, iTransformer, etc.)
├── exp/             # Experiment classes for different tasks
├── data_provider/   # Data loading and preprocessing
├── layers/          # Neural network building blocks
├── utils/           # Utilities (metrics, losses, tools)
├── scripts/         # Pre-configured training scripts
└── run.py           # Main entry point
```

### Key Concepts

1. **Task Types**:
   - `long_term_forecast`: Predict far into the future (e.g., 96-720 steps ahead)
   - `short_term_forecast`: Predict near future (e.g., 1-24 steps ahead)

2. **Sequence Parameters**:
   - `seq_len`: Input sequence length (historical data to look at)
   - `label_len`: Start token length (overlap between input and output)
   - `pred_len`: Prediction length (how far to predict)

3. **Feature Modes**:
   - `M`: Multivariate predict multivariate (use all features to predict all)
   - `S`: Univariate predict univariate (single feature predicts itself)
   - `MS`: Multivariate predict univariate (all features predict one target)

---

## Data Preparation for Trading

### 1. Data Format Requirements

Your trading data CSV should follow this format:

```csv
date,feature1,feature2,feature3,...,target
2023-01-01 00:00:00,100.5,50.2,75.3,...,50000
2023-01-01 01:00:00,101.2,50.5,75.8,...,51000
...
```

**Requirements**:
- First column must be named `date` (datetime format)
- Last column should be your target variable (e.g., price, volume)
- All other columns are features (e.g., OHLCV, indicators)
- No missing values in the date column

### 2. Example Trading Data Structure

```python
# Example: Cryptocurrency price data
date,open,high,low,close,volume,RSI,MACD,target_price
2023-01-01 00:00:00,50000,51000,49500,50500,1000000,55,0.5,50500
2023-01-01 01:00:00,50500,51500,50000,51000,1200000,60,0.6,51000
...
```

### 3. Data Directory Structure

```
dataset/
└── trading/
    ├── BTC_USD.csv          # Your trading data
    ├── ETH_USD.csv
    └── ...
```

---

## Running Your First Model

### Method 1: Using Python Script

Create a file `train_trading_model.py`:

```python
import subprocess
import sys

# Training command
cmd = [
    "python", "run.py",
    "--task_name", "long_term_forecast",
    "--is_training", "1",
    "--root_path", "./dataset/trading/",
    "--data_path", "BTC_USD.csv",
    "--model_id", "BTC_96_96",
    "--model", "TimesNet",
    "--data", "custom",  # Use 'custom' for your own data
    "--features", "M",   # Multivariate forecasting
    "--seq_len", "96",   # Look at 96 time steps
    "--label_len", "48", # Overlap length
    "--pred_len", "96",  # Predict 96 steps ahead
    "--e_layers", "2",   # Encoder layers
    "--d_layers", "1",   # Decoder layers
    "--enc_in", "7",     # Number of input features (adjust to your data)
    "--dec_in", "7",
    "--c_out", "7",      # Number of output features
    "--d_model", "32",   # Model dimension
    "--d_ff", "64",      # Feed-forward dimension
    "--top_k", "5",      # TimesNet specific
    "--des", "Trading_Exp",
    "--itr", "1",        # Number of experiments
    "--train_epochs", "10",
    "--batch_size", "32",
    "--learning_rate", "0.0001",
    "--patience", "3",
    "--freq", "h"        # Frequency: h=hourly, t=minutely, d=daily
]

subprocess.run(cmd)
```

### Method 2: Using Shell Script

Create `scripts/trading/TimesNet_BTC.sh`:

```bash
#!/bin/bash

model_name=TimesNet

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/trading/ \
  --data_path BTC_USD.csv \
  --model_id BTC_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 32 \
  --d_ff 64 \
  --des 'Trading_Exp' \
  --itr 1 \
  --top_k 5 \
  --train_epochs 10 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --patience 3 \
  --freq h
```

Run it:
```bash
chmod +x scripts/trading/TimesNet_BTC.sh
bash scripts/trading/TimesNet_BTC.sh
```

### Method 3: Direct Command Line

```bash
python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/trading/ \
  --data_path BTC_USD.csv \
  --model_id BTC_96_96 \
  --model TimesNet \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 32 \
  --d_ff 64 \
  --des Trading_Exp \
  --itr 1 \
  --top_k 5 \
  --train_epochs 10 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --patience 3 \
  --freq h
```

---

## Model Selection for Trading

### Top Models for Trading (Based on Performance)

1. **TimesNet** (Recommended for beginners)
   - Excellent for capturing periodic patterns
   - Good for daily/hourly trading data
   - Parameters: `--top_k 5`

2. **iTransformer** (Best for long-term)
   - State-of-the-art for long-term forecasting
   - Great for trend prediction
   - No special parameters needed

3. **TimeMixer** (Best overall)
   - Excellent multi-scale mixing
   - Good for various time horizons
   - No special parameters needed

4. **PatchTST** (Efficient)
   - Fast training and inference
   - Good for high-frequency data
   - No special parameters needed

5. **DLinear** (Simple & Fast)
   - Very fast, simple linear model
   - Good baseline
   - Parameter: `--individual 1` (per-channel linear)

### Model Comparison for Trading

| Model | Speed | Accuracy | Best For |
|-------|-------|----------|----------|
| TimesNet | Medium | High | Periodic patterns |
| iTransformer | Medium | Very High | Long-term trends |
| TimeMixer | Medium | Very High | Multi-scale patterns |
| PatchTST | Fast | High | High-frequency data |
| DLinear | Very Fast | Medium | Quick baselines |

---

## Custom Trading Data Integration

### Step 1: Prepare Your Data

```python
import pandas as pd
import numpy as np

# Load your trading data
df = pd.read_csv('your_trading_data.csv')

# Ensure date column is datetime
df['date'] = pd.to_datetime(df['date'])

# Sort by date
df = df.sort_values('date').reset_index(drop=True)

# Save in the required format
df.to_csv('./dataset/trading/BTC_USD.csv', index=False)
```

### Step 2: Count Your Features

```python
# Count total features (excluding 'date' and target)
# If you have: date, open, high, low, close, volume, RSI, target_price
# Then enc_in = 7 (open, high, low, close, volume, RSI, target_price)
# c_out = 7 (same as enc_in for multivariate)
```

### Step 3: Determine Sequence Lengths

For trading:
- **Hourly data**: `seq_len=96` (4 days), `pred_len=24` (1 day)
- **Daily data**: `seq_len=30` (1 month), `pred_len=7` (1 week)
- **Minute data**: `seq_len=1440` (1 day), `pred_len=60` (1 hour)

---

## Training Trading Models

### Complete Training Example

```python
# train_trading_model.py
import subprocess

def train_model(
    data_path="BTC_USD.csv",
    model="TimesNet",
    seq_len=96,
    pred_len=24,
    epochs=20,
    batch_size=32,
    learning_rate=0.0001
):
    cmd = [
        "python", "run.py",
        "--task_name", "long_term_forecast",
        "--is_training", "1",
        "--root_path", "./dataset/trading/",
        "--data_path", data_path,
        "--model_id", f"{data_path.replace('.csv', '')}_{seq_len}_{pred_len}",
        "--model", model,
        "--data", "custom",
        "--features", "M",
        "--seq_len", str(seq_len),
        "--label_len", str(seq_len // 2),
        "--pred_len", str(pred_len),
        "--e_layers", "2",
        "--d_layers", "1",
        "--enc_in", "7",  # Adjust based on your features
        "--dec_in", "7",
        "--c_out", "7",
        "--d_model", "32",
        "--d_ff", "64",
        "--des", "Trading_Model",
        "--itr", "1",
        "--train_epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--learning_rate", str(learning_rate),
        "--patience", "5",
        "--freq", "h",
        "--use_gpu", "True",
        "--gpu", "0"
    ]
    
    if model == "TimesNet":
        cmd.extend(["--top_k", "5"])
    
    subprocess.run(cmd)

# Train your model
train_model(
    data_path="BTC_USD.csv",
    model="TimesNet",
    seq_len=96,
    pred_len=24,
    epochs=20
)
```

### Training Tips

1. **Start Small**: Begin with `seq_len=96`, `pred_len=24`, `epochs=10`
2. **Monitor Loss**: Check `./checkpoints/` for saved models
3. **Use Early Stopping**: `--patience 5` stops training if no improvement
4. **Adjust Learning Rate**: Start with `0.0001`, increase if training is slow
5. **Batch Size**: Use larger batches (32-64) if you have GPU memory

---

## Making Predictions

### After Training: Test Your Model

```bash
python run.py \
  --task_name long_term_forecast \
  --is_training 0 \  # Set to 0 for testing only
  --root_path ./dataset/trading/ \
  --data_path BTC_USD.csv \
  --model_id BTC_96_96 \
  --model TimesNet \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 32 \
  --d_ff 64 \
  --des Trading_Exp \
  --freq h
```

### Load Predictions

```python
import numpy as np
import matplotlib.pyplot as plt

# Load predictions
setting = "long_term_forecast_BTC_96_96_TimesNet_custom_ftM_sl96_ll48_pl96_dm32_nh8_el2_dl1_df64_expand2_dc4_fc1_ebtimeF_dtTrue_Trading_Exp_0"
preds = np.load(f'./results/{setting}/pred.npy')
trues = np.load(f'./results/{setting}/true.npy')
metrics = np.load(f'./results/{setting}/metrics.npy')

print(f"MAE: {metrics[0]:.4f}")
print(f"MSE: {metrics[1]:.4f}")
print(f"RMSE: {metrics[2]:.4f}")
print(f"MAPE: {metrics[3]:.4f}")

# Visualize
plt.figure(figsize=(12, 6))
plt.plot(trues[0, :, -1], label='True')
plt.plot(preds[0, :, -1], label='Predicted')
plt.legend()
plt.title('Trading Price Prediction')
plt.show()
```

### Real-Time Prediction Function

```python
import torch
import numpy as np
from data_provider.data_factory import data_provider
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
import argparse

def predict_future(model_path, data_path, seq_len=96, pred_len=24):
    """
    Make predictions on new data
    """
    # Load model arguments (you'll need to save these during training)
    args = argparse.Namespace(
        task_name='long_term_forecast',
        model='TimesNet',
        data='custom',
        root_path='./dataset/trading/',
        data_path=data_path,
        features='M',
        seq_len=seq_len,
        label_len=seq_len // 2,
        pred_len=pred_len,
        enc_in=7,
        dec_in=7,
        c_out=7,
        d_model=32,
        d_ff=64,
        e_layers=2,
        d_layers=1,
        top_k=5,
        embed='timeF',
        freq='h',
        use_gpu=True,
        gpu=0,
        use_amp=False,
        inverse=False,
        use_dtw=False
    )
    
    # Create experiment
    exp = Exp_Long_Term_Forecast(args)
    
    # Load trained model
    exp.model.load_state_dict(torch.load(model_path))
    exp.model.eval()
    
    # Get latest data
    _, test_loader = exp._get_data(flag='test')
    
    # Make prediction
    with torch.no_grad():
        batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(test_loader))
        batch_x = batch_x.float().to(exp.device)
        batch_x_mark = batch_x_mark.float().to(exp.device)
        
        dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(exp.device)
        batch_y_mark = batch_y_mark.float().to(exp.device)
        
        outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        predictions = outputs[:, -pred_len:, :].cpu().numpy()
    
    return predictions

# Usage
predictions = predict_future(
    model_path='./checkpoints/your_model/checkpoint.pth',
    data_path='BTC_USD.csv',
    seq_len=96,
    pred_len=24
)
print(f"Next 24 predictions: {predictions[0, :, -1]}")
```

---

## Best Practices for Trading

### 1. Data Preprocessing

```python
# Normalize your data (already done by the library, but good to know)
# The library uses StandardScaler on training data
# Always use the same scaler for inference
```

### 2. Feature Engineering

Include relevant features:
- **Price features**: Open, High, Low, Close
- **Volume**: Trading volume
- **Technical indicators**: RSI, MACD, Bollinger Bands
- **Market features**: Volatility, spreads

### 3. Model Selection Strategy

1. **Start with DLinear** for a quick baseline
2. **Try TimesNet** for periodic patterns
3. **Use iTransformer** for long-term trends
4. **Experiment with TimeMixer** for best overall performance

### 4. Hyperparameter Tuning

Key parameters to tune:
- `seq_len`: How much history to use (try 48, 96, 192)
- `pred_len`: How far to predict (start small: 24, then increase)
- `d_model`: Model capacity (32, 64, 128)
- `learning_rate`: Start with 0.0001, try 0.0005, 0.001
- `batch_size`: 16, 32, 64 (depends on GPU memory)

### 5. Validation Strategy

```python
# The library automatically splits:
# - 70% training
# - 20% validation  
# - 10% testing

# For trading, consider:
# - Use recent data for testing (most important)
# - Avoid data leakage (don't use future data)
```

### 6. Evaluation Metrics

The library provides:
- **MAE** (Mean Absolute Error): Average prediction error
- **MSE** (Mean Squared Error): Penalizes large errors
- **RMSE** (Root Mean Squared Error): Error in same units as target
- **MAPE** (Mean Absolute Percentage Error): Percentage error

For trading, also consider:
- **Directional Accuracy**: % of correct direction predictions
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst peak-to-trough decline

### 7. Model Ensemble

Combine multiple models for better predictions:

```python
# Train multiple models
models = ['TimesNet', 'iTransformer', 'TimeMixer']
predictions = []

for model in models:
    # Train and get predictions
    pred = train_and_predict(model)
    predictions.append(pred)

# Average predictions
ensemble_pred = np.mean(predictions, axis=0)
```

---

## Advanced Usage

### 1. Using Exogenous Variables (TimeXer)

For incorporating external features (news, sentiment, etc.):

```bash
python run.py \
  --task_name long_term_forecast \
  --model TimeXer \
  --data custom \
  --patch_len 16 \
  # ... other parameters
```

### 2. Data Augmentation

Improve model robustness:

```bash
python run.py \
  --augmentation_ratio 2 \
  --jitter \
  --scaling \
  # ... other parameters
```

### 3. Multi-GPU Training

```bash
python run.py \
  --use_multi_gpu \
  --devices 0,1,2,3 \
  # ... other parameters
```

### 4. Custom Loss Functions

Modify `utils/losses.py` to add trading-specific losses:
- Sharpe ratio loss
- Directional loss
- Quantile loss

### 5. Real-Time Trading Integration

```python
# trading_bot.py
import time
from your_exchange_api import get_latest_data, place_order

def trading_loop(model, update_interval=3600):
    while True:
        # Get latest data
        data = get_latest_data()
        
        # Make prediction
        prediction = model.predict(data)
        
        # Trading logic
        if prediction > current_price * 1.02:  # 2% increase predicted
            place_order('buy')
        elif prediction < current_price * 0.98:  # 2% decrease predicted
            place_order('sell')
        
        time.sleep(update_interval)
```

---

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce `batch_size` (try 16 or 8)
   - Reduce `d_model` (try 16 or 32)
   - Reduce `seq_len`

2. **Poor Predictions**
   - Increase `seq_len` to use more history
   - Try different models
   - Check data quality
   - Increase training epochs

3. **Training Too Slow**
   - Use GPU (`--use_gpu True`)
   - Reduce `seq_len` or `pred_len`
   - Use simpler model (DLinear)

4. **Data Format Errors**
   - Ensure date column is named 'date'
   - Check for missing values
   - Verify CSV encoding (UTF-8)

---

## Quick Reference

### Essential Commands

```bash
# Train
python run.py --task_name long_term_forecast --is_training 1 --model TimesNet --data custom ...

# Test
python run.py --task_name long_term_forecast --is_training 0 --model TimesNet --data custom ...

# View results
ls ./results/
ls ./checkpoints/
```

### Key Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `seq_len` | Input history length | 48, 96, 192 |
| `pred_len` | Prediction horizon | 24, 48, 96 |
| `d_model` | Model dimension | 32, 64, 128 |
| `e_layers` | Encoder layers | 2, 3, 4 |
| `batch_size` | Batch size | 16, 32, 64 |
| `learning_rate` | Learning rate | 0.0001, 0.0005 |

---

## Next Steps

1. **Prepare your trading data** in the required CSV format
2. **Start with a simple model** (DLinear or TimesNet)
3. **Train on a small subset** to verify everything works
4. **Evaluate predictions** and adjust hyperparameters
5. **Scale up** to full dataset and longer training
6. **Integrate** with your trading bot

---

## Resources

- **Repository**: https://github.com/thuml/Time-Series-Library
- **Paper**: TimesNet paper for understanding the models
- **Tutorial**: Check `tutorial/TimesNet_tutorial.ipynb`

---

## Support

For issues specific to trading:
1. Check data format matches requirements
2. Verify feature counts match `enc_in`/`c_out`
3. Ensure sufficient training data
4. Review model-specific parameters

Happy trading! 📈

