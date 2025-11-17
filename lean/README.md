# GLD Trading Bot - LEAN Backtest

Simple QuantConnect LEAN setup for backtesting GLD (Gold) trading strategies.

## Quick Start

1. **Convert data to LEAN format:**
   ```bash
   python3 convert_data.py
   ```

2. **Run backtest:**
   ```bash
   lean backtest .
   ```

3. **View results:**
   Results are in `backtests/` directory, organized by timestamp.

## Files

- `main.py` - Trading algorithm (moving average crossover strategy)
- `convert_data.py` - Converts GLD.csv to LEAN zip format
- `Data/` - Converted data files (created by convert_data.py)
- `backtests/` - Backtest results (created after running)

## Algorithm

Simple MA crossover: Buys when 10-min MA crosses above 30-min MA, sells when it crosses below.

