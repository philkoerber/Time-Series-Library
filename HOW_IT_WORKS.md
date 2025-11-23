# How Your Model Works: Sequence In → Sequence Out

## Simple Explanation

Your model is a **sequence-to-sequence predictor**. It's super simple:

```
INPUT:  Give it 1440 time steps of history
        ↓
MODEL:  Analyzes patterns
        ↓
OUTPUT: Gives you 60 time steps of future predictions
```

## Visual Example

```
┌────────────────────────────────────────────────────────────┐
│                   YOUR TRAINED MODEL                       │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  INPUT SEQUENCE (Last 1440 steps)                         │
│  ───────────────────────────────                          │
│  [t-1439, t-1438, t-1437, ..., t-2, t-1, t]             │
│  │                                                        │
│  │  Each step has 96 features:                           │
│  │  - Prices (open, high, low, close)                    │
│  │  - Volumes                                             │
│  │  - Technical indicators                                │
│  │  - Engineered features                                 │
│  │                                                        │
│  ↓                                                        │
│  [MODEL PROCESSES PATTERNS]                               │
│  ↓                                                        │
│  OUTPUT SEQUENCE (Next 60 steps)                          │
│  ───────────────────────────────                          │
│  [t+1, t+2, t+3, ..., t+59, t+60]                       │
│                                                            │
│  Each step has 96 features (predicted values)             │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

## Shape Details

```python
# INPUT
input_sequence.shape = (1440, 96)
#      ↑        ↑
#      |        └─ 96 features per time step
#      └─ 1440 time steps (1 day of minute data)

# OUTPUT
output_sequence.shape = (60, 96)
#      ↑        ↑
#      |        └─ 96 features per time step
#      └─ 60 time steps (1 hour ahead)
```

## Where Are Your Models?

Your trained models are saved in:
```
./checkpoints/[model_name]/checkpoint.pth
```

### Your Current Model:
```
./checkpoints/long_term_forecast_engineered_features_1440_60_iTransformer_custom_ftM_sl1440_ll720_pl60_dm64_nh8_el2_dl1_df128_expand2_dc4_fc1_ebtimeF_dtTrue_Trading_Transformer_0/checkpoint.pth
```

### Model Parameters (from the directory name):
- **seq_len = 1440**: Input sequence length (24 hours)
- **pred_len = 60**: Output sequence length (1 hour)
- **d_model = 64**: Model dimension
- **n_heads = 8**: Attention heads
- **e_layers = 2**: Encoder layers
- **d_layers = 1**: Decoder layers
- **d_ff = 128**: Feed-forward dimension

## Simple Code Example

```python
from trading_bot_inference import load_predictor_from_checkpoint
import pandas as pd

# Load model (do this once)
predictor = load_predictor_from_checkpoint()

# Get your data (must have at least 1440 rows)
df = pd.read_csv('your_data.csv')

# Take last 1440 rows (INPUT SEQUENCE)
input_data = df.tail(1440)

# Get predictions (OUTPUT SEQUENCE)
predictions, future_dates, features = predictor.predict(input_data)

# predictions shape: (60, 96)
# - 60 future time steps
# - 96 features per step

# Access predictions:
first_step = predictions[0]      # Prediction for t+1
last_step = predictions[-1]      # Prediction for t+60
all_predictions = predictions     # All 60 steps

print(f"Predicted {len(predictions)} steps ahead")
print(f"Shape: {predictions.shape}")
```

## What Does Each Feature Mean?

Your model predicts **all 96 features** at once:

1. **Price features**: BTCUSD_open, BTCUSD_high, BTCUSD_low, BTCUSD_close, etc.
2. **Volume features**: BTCUSD_volume, ETHUSD_volume, etc.
3. **Technical indicators**: RSI, MACD, moving averages, etc.
4. **Engineered features**: Returns, lags, momentum, etc.

To get just your target prediction:
```python
# Get only target predictions (simpler)
target_predictions, dates = predictor.predict_target_only(
    df, 
    target_column='target'  # or 'BTCUSD_close' etc.
)
# target_predictions shape: (60,) - just 60 price predictions
```

## Time Flow

```
Past (History)          Present          Future (Predictions)
───────────────────────────────────────────────────────────────
t-1439  t-1438  ...  t-1  [t]  │  [t+1]  [t+2]  ...  [t+60]
↑                              │  ↑
│                              │  │
└── INPUT: 1440 steps ─────────┼──┴── OUTPUT: 60 steps
                               │
                          MODEL
```

## Real-World Example

```python
# You have data up to "2024-01-15 15:00:00"
last_timestamp = df['date'].iloc[-1]  # "2024-01-15 15:00:00"

# Get last 1440 steps (24 hours of minute data)
input_data = df.tail(1440)

# Predict next 60 steps (next hour)
predictions, future_dates, features = predictor.predict(input_data)

# future_dates will be:
# ["2024-01-15 15:01:00", "2024-01-15 15:02:00", ..., "2024-01-15 16:00:00"]
#       ↑                                                    ↑
#    t+1                                                  t+60
```

## The Magic

The model learned from training data to:
1. **Recognize patterns** in sequences (trends, cycles, relationships)
2. **Extract meaningful features** from the 96-dimensional data
3. **Predict future values** based on historical patterns

It's like having a model that says:
> "I've seen this pattern before in training, so I predict the next 60 steps will look like this..."

## Try It Yourself

Run the simple example:
```bash
python simple_predict_example.py
```

This will:
1. Show where your models are stored
2. Load your model
3. Demonstrate the sequence-in → sequence-out flow
4. Show you the exact shapes and formats

## Key Points

✅ **Input**: Always 1440 time steps × 96 features  
✅ **Output**: Always 60 time steps × 96 features  
✅ **Simple**: Give it history, get future predictions  
✅ **Fast**: Predictions take milliseconds  
✅ **Multivariate**: Predicts all features at once  

That's it! It's really that simple: **sequence in, sequence out** 🚀

