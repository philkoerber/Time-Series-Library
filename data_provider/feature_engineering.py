"""
Feature engineering utilities for historical price data.
"""
from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd


BASE_PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index (RSI)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Use min_periods=1 to allow earlier computation, then forward fill initial NaNs
    avg_gain = gain.ewm(alpha=1 / period, min_periods=1, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=1, adjust=False).mean()

    # Handle division by zero
    avg_loss = avg_loss.replace(0, np.nan)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    # Forward fill initial NaN values where we don't have enough data yet
    rsi = rsi.bfill().fillna(50.0)  # Default to neutral RSI if still NaN
    return rsi


def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Average True Range (ATR)."""
    prev_close = close.shift(1)
    high_low = high - low
    high_close = (high - prev_close).abs()
    low_close = (low - prev_close).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    # Use min_periods=1 and forward fill initial values
    atr = true_range.rolling(window=period, min_periods=1).mean()
    # Forward fill initial NaN values
    atr = atr.bfill().fillna(0.0)
    return atr
def _engineer_single_asset_features(
    df: pd.DataFrame,
    prefix: str = "",
    rsi_period: int = 14,
    atr_period: int = 14,
    bollinger_window: int = 20,
    volume_window: int = 20,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Compute engineered features for a single instrument.

    Args:
        df: DataFrame with columns open/high/low/close/volume.
        prefix: Optional column prefix applied to the engineered features.
        rsi_period: RSI lookback.
        atr_period: ATR lookback.
        bollinger_window: Window for Bollinger bands.
        volume_window: Window for relative volume.

    Returns:
        Tuple of (feature DataFrame, target Series).
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty. Cannot engineer features.")

    data = df.copy().sort_index()

    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex for time-based features.")

    close = data["close"]
    volume = data["volume"]

    # Basic price features - fill initial NaN with 0 for returns
    data["log_return"] = np.log(close / close.shift(1))
    data["lag_return_1m"] = np.log(close / close.shift(1))
    data["lag_return_2m"] = np.log(close / close.shift(2))
    data["lag_return_5m"] = np.log(close / close.shift(5))
    data["high_low_range"] = data["high"] - data["low"]
    
    # Rolling features with min_periods=1 and forward fill
    data["sma_5"] = close.rolling(window=5, min_periods=1).mean()
    data["sma_10"] = close.rolling(window=10, min_periods=1).mean()
    data["ema_5"] = close.ewm(span=5, adjust=False, min_periods=1).mean()
    data["ema_10"] = close.ewm(span=10, adjust=False, min_periods=1).mean()
    data["momentum_1m"] = close - close.shift(1)
    data["momentum_5m"] = close - close.shift(5)
    
    # RSI uses forward fill internally
    data["rsi"] = _compute_rsi(close, period=rsi_period)

    # MACD with min_periods=1 and forward fill
    ema_fast = close.ewm(span=12, adjust=False, min_periods=1).mean()
    ema_slow = close.ewm(span=26, adjust=False, min_periods=1).mean()
    data["macd"] = ema_fast - ema_slow
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False, min_periods=1).mean()

    # Bollinger bands with min_periods=1
    rolling_mean = close.rolling(window=bollinger_window, min_periods=1).mean()
    rolling_std = close.rolling(window=bollinger_window, min_periods=1).std()
    upper_band = rolling_mean + (2 * rolling_std)
    lower_band = rolling_mean - (2 * rolling_std)
    band_range = upper_band - lower_band
    band_range = band_range.replace(0, np.nan)

    # Fill NaN for bollinger_percent where band_range is 0 or NaN
    data["bollinger_percent"] = (close - lower_band) / band_range
    data["bollinger_percent"] = data["bollinger_percent"].fillna(0.5)  # Default to middle of band
    data["bollinger_width"] = band_range.fillna(0.0)
    data["atr"] = _compute_atr(data["high"], data["low"], close, period=atr_period)

    # Volume features
    data["log_volume"] = np.log1p(volume)
    vol_mean = volume.rolling(window=volume_window, min_periods=1).mean()
    vol_std = volume.rolling(window=volume_window, min_periods=1).std()
    vol_std = vol_std.replace(0, np.nan)
    data["relative_volume"] = (volume - vol_mean) / vol_std
    # Fill NaN for relative_volume (when std is 0, volume is constant)
    data["relative_volume"] = data["relative_volume"].fillna(0.0)

    minutes_since_start = (
        data.index.hour * 60
        + data.index.minute
        + data.index.second / 60
    )
    total_minutes = 24 * 60
    angle = 2 * np.pi * (minutes_since_start / total_minutes)
    data["time_sin"] = np.sin(angle)
    data["time_cos"] = np.cos(angle)

    # Future return is only NaN for the last row (which we'll drop)
    data["future_log_return"] = data["log_return"].shift(-1)

    feature_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "log_return",
        "lag_return_1m",
        "lag_return_2m",
        "lag_return_5m",
        "high_low_range",
        "sma_5",
        "sma_10",
        "ema_5",
        "ema_10",
        "momentum_1m",
        "momentum_5m",
        "rsi",
        "macd",
        "macd_signal",
        "bollinger_percent",
        "bollinger_width",
        "atr",
        "log_volume",
        "relative_volume",
        "time_sin",
        "time_cos",
    ]

    # Forward fill any remaining NaN in features (should be minimal now)
    for col in feature_columns:
        if col in data.columns:
            data[col] = data[col].ffill().fillna(0.0)

    # Only drop the last row where future_log_return is NaN (we can't predict without future data)
    # This is much more conservative than dropping all rows with any NaN
    data = data[data["future_log_return"].notna()]

    features = data[feature_columns]
    if prefix:
        features = features.add_prefix(prefix)

    targets = data["future_log_return"]
    return features, targets


def _combine_symbol_dataframes(
    data: Mapping[str, pd.DataFrame],
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Merge individual symbol DataFrames into a single wide DataFrame.

    Each column is prefixed with the symbol, e.g. GLD_close, SLV_volume, etc.
    
    Uses outer join to preserve all timestamps, then forward fills missing values.
    """
    if not data:
        raise ValueError("No symbol data provided to combine.")

    renamed_frames = []
    for symbol, df in data.items():
        if df.empty:
            raise ValueError(f"DataFrame for symbol {symbol} is empty.")
        missing_columns = [col for col in BASE_PRICE_COLUMNS if col not in df.columns]
        if missing_columns:
            raise KeyError(
                f"Missing required columns for {symbol}: {', '.join(missing_columns)}"
            )
        renamed = df.copy()
        renamed.columns = [f"{symbol}_{col}" for col in renamed.columns]
        renamed_frames.append(renamed)

    # Use outer join to preserve all timestamps, then forward fill missing values
    # This preserves much more data than inner join
    combined = pd.concat(renamed_frames, axis=1, join="outer").sort_index()
    
    if drop_na:
        # Instead of dropping rows, forward fill missing values
        # Only drop rows where ALL symbols have NaN (should be rare)
        combined = combined.ffill().bfill()
        # Only drop if all columns are NaN (shouldn't happen after ffill/bfill)
        combined = combined.dropna(how='all')

    return combined


def engineer_features_from_symbol_data(
    symbol_data: Mapping[str, pd.DataFrame],
    target_symbol: str,
    symbols: Iterable[str] | None = None,
    rsi_period: int = 14,
    atr_period: int = 14,
    bollinger_window: int = 20,
    volume_window: int = 20,
    drop_na: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Convenience utility for live trading or training that:
        1. Combines individual symbol OHLCV DataFrames.
        2. Engineers per-asset and cross-asset features.
        3. Returns the combined feature table and aligned target series.
    """
    if not symbol_data:
        raise ValueError("No symbol data provided.")

    if symbols is None:
        symbols = tuple(symbol_data.keys())

    symbols = list(symbols)

    if target_symbol not in symbols:
        raise ValueError("target_symbol must be included in the symbols iterable.")

    missing_symbols = [symbol for symbol in symbols if symbol not in symbol_data]
    if missing_symbols:
        missing_list = ", ".join(missing_symbols)
        raise KeyError(f"Missing data for symbols: {missing_list}")

    selected_data = {symbol: symbol_data[symbol] for symbol in symbols}
    df = _combine_symbol_dataframes(selected_data, drop_na=drop_na)

    feature_frames: list[pd.DataFrame] = []
    target_series: pd.Series | None = None
    closes: dict[str, pd.Series] = {}
    log_returns: dict[str, pd.Series] = {}

    for symbol in symbols:
        expected_columns = [f"{symbol}_{col}" for col in BASE_PRICE_COLUMNS]
        asset_df = df[expected_columns].rename(columns=lambda name: name.split("_", 1)[1])

        features, targets = _engineer_single_asset_features(
            asset_df,
            prefix=f"{symbol}_",
            rsi_period=rsi_period,
            atr_period=atr_period,
            bollinger_window=bollinger_window,
            volume_window=volume_window,
        )
        feature_frames.append(features)

        if symbol == target_symbol:
            target_series = targets

        closes[symbol] = asset_df["close"]
        log_returns[symbol] = np.log(asset_df["close"] / asset_df["close"].shift(1))

    if target_series is None:
        raise RuntimeError("Failed to compute target series for the selected target_symbol.")

    # Use outer join to preserve all timestamps
    combined_features = pd.concat(feature_frames, axis=1, join="outer").sort_index()

    cross_feature_frames: list[pd.DataFrame] = []
    rolling_window = 60
    # Use min_periods=1 instead of rolling_window//2 to preserve more data
    min_periods = 1

    for idx, sym_a in enumerate(symbols):
        for sym_b in symbols[idx + 1 :]:
            close_a = closes[sym_a]
            close_b = closes[sym_b]
            ret_a = log_returns[sym_a]
            ret_b = log_returns[sym_b]

            log_price_ratio = np.log(close_a / close_b)
            log_price_ratio.name = f"{sym_a}_{sym_b}_log_price_ratio"

            return_spread = ret_a - ret_b
            return_spread.name = f"{sym_a}_{sym_b}_return_spread"

            # Rolling features with min_periods=1 and forward fill
            rolling_corr = ret_a.rolling(window=rolling_window, min_periods=min_periods).corr(ret_b)
            rolling_corr.name = f"{sym_a}_{sym_b}_rolling_corr_{rolling_window}"
            rolling_corr = rolling_corr.bfill().fillna(0.0)  # Default to no correlation

            rolling_cov = ret_a.rolling(window=rolling_window, min_periods=min_periods).cov(ret_b)
            rolling_var_b = ret_b.rolling(window=rolling_window, min_periods=min_periods).var()
            beta_a_on_b = rolling_cov / (rolling_var_b + 1e-9)
            beta_a_on_b.name = f"{sym_a}_beta_on_{sym_b}_{rolling_window}"
            beta_a_on_b = beta_a_on_b.bfill().fillna(1.0)  # Default to beta of 1

            rolling_var_a = ret_a.rolling(window=rolling_window, min_periods=min_periods).var()
            beta_b_on_a = rolling_cov / (rolling_var_a + 1e-9)
            beta_b_on_a.name = f"{sym_b}_beta_on_{sym_a}_{rolling_window}"
            beta_b_on_a = beta_b_on_a.bfill().fillna(1.0)  # Default to beta of 1

            spread = close_a - close_b
            spread_mean = spread.rolling(window=rolling_window, min_periods=min_periods).mean()
            spread_std = spread.rolling(window=rolling_window, min_periods=min_periods).std()
            spread_std = spread_std.replace(0, np.nan)
            spread_zscore = (spread - spread_mean) / (spread_std + 1e-9)
            spread_zscore.name = f"{sym_a}_{sym_b}_spread_zscore_{rolling_window}"
            spread_zscore = spread_zscore.bfill().fillna(0.0)  # Default to 0 z-score

            cross_features = pd.concat(
                [
                    log_price_ratio,
                    return_spread,
                    rolling_corr,
                    beta_a_on_b,
                    beta_b_on_a,
                    spread_zscore,
                ],
                axis=1,
            )
            cross_feature_frames.append(cross_features)

    if cross_feature_frames:
        # Use outer join for cross features too
        cross_features_df = pd.concat(cross_feature_frames, axis=1, join="outer")
        combined_features = pd.concat([combined_features, cross_features_df], axis=1, join="outer")
        # Forward fill any NaN values from outer joins
        combined_features = combined_features.ffill().bfill()

    # Align targets - only drop where target is NaN (last row)
    aligned_targets = target_series.reindex(combined_features.index)
    valid_mask = aligned_targets.notna()
    combined_features = combined_features.loc[valid_mask]
    aligned_targets = aligned_targets.loc[valid_mask]

    return combined_features, aligned_targets


def prepare_multi_asset_features(
    symbol_data: Mapping[str, pd.DataFrame],
    target_symbol: str,
    symbols: Iterable[str] | None = None,
    rsi_period: int = 14,
    atr_period: int = 14,
    bollinger_window: int = 20,
    volume_window: int = 20,
    drop_na: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convenience wrapper that returns numpy arrays for modelling pipelines.
    """
    feature_table, target_series = engineer_features_from_symbol_data(
        symbol_data=symbol_data,
        target_symbol=target_symbol,
        symbols=symbols,
        rsi_period=rsi_period,
        atr_period=atr_period,
        bollinger_window=bollinger_window,
        volume_window=volume_window,
        drop_na=drop_na,
    )
    return feature_table.to_numpy(), target_series.to_numpy()

