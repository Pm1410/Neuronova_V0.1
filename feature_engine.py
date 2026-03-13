"""
features/feature_engine.py
---------------------------
Full feature engineering pipeline:
- Trend indicators (MACD, EMA, SMA)
- Momentum (RSI, Stochastic, ROC)
- Volatility (ATR, Bollinger Bands, Keltner)
- Volume (OBV, VWAP, Volume ratios)
- Microstructure (spread proxies, bar ranges)
- Lag / rolling statistical features
- Time features (hour of day, day of week)

All features are normalized where appropriate.
No lookahead bias — all computations use only past data.
"""

import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import ta
from loguru import logger
from rich.console import Console
from rich.table import Table

warnings.filterwarnings("ignore")
console = Console()


class FeatureEngine:
    """
    Transforms raw OHLCV data into ML-ready feature matrix.
    
    Design principles:
    - Zero lookahead: all rolling windows are right-aligned
    - Stationary: features are returns/ratios, not raw prices
    - Informative: diverse signal types (trend, momentum, vol, volume)
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.rsi_periods = cfg.get("rsi_periods", [7, 14, 21])
        self.macd_params = cfg.get("macd_params", {"fast": 12, "slow": 26, "signal": 9})
        self.bbands_period = cfg.get("bbands_period", 20)
        self.atr_period = cfg.get("atr_period", 14)
        self.vol_ma_periods = cfg.get("volume_ma_periods", [10, 20])
        self.lag_periods = cfg.get("lag_periods", [1, 2, 3, 5, 10])
        self.rolling_windows = cfg.get("rolling_windows", [5, 10, 20, 50])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main entry point. Takes raw OHLCV, returns feature DataFrame.
        
        Args:
            df: DataFrame with columns [open, high, low, close, volume]
        
        Returns:
            Feature DataFrame aligned to input index (NaN rows dropped)
        """
        logger.info(f"Engineering features on {len(df):,} candles")
        feats = pd.DataFrame(index=df.index)

        feats = self._add_returns(feats, df)
        feats = self._add_trend(feats, df)
        feats = self._add_momentum(feats, df)
        feats = self._add_volatility(feats, df)
        feats = self._add_volume(feats, df)
        feats = self._add_microstructure(feats, df)
        feats = self._add_lags(feats)
        feats = self._add_rolling_stats(feats, df)
        feats = self._add_time_features(feats)

        # Drop NaN rows introduced by indicator warm-up periods
        original_len = len(feats)
        feats.dropna(inplace=True)
        dropped = original_len - len(feats)
        logger.info(
            f"Features: {feats.shape[1]} columns | "
            f"Dropped {dropped} warm-up rows | {len(feats):,} usable rows"
        )
        self._print_feature_summary(feats)
        return feats

    # ─── Feature Groups ─────────────────────────────────────────────────────

    def _add_returns(self, feats: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Log returns at multiple horizons."""
        close = df["close"]
        for h in [1, 2, 3, 5, 10, 20]:
            feats[f"ret_{h}"] = np.log(close / close.shift(h))
        return feats

    def _add_trend(self, feats: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """EMA crossovers, MACD, price vs MA ratios."""
        close = df["close"]

        # EMA ratios (price normalized, avoids price-level non-stationarity)
        for period in [9, 21, 50, 100, 200]:
            ema = close.ewm(span=period, adjust=False).mean()
            feats[f"ema_ratio_{period}"] = close / ema - 1

        # MACD
        p = self.macd_params
        macd_ind = ta.trend.MACD(
            close, window_fast=p["fast"], window_slow=p["slow"], window_sign=p["signal"]
        )
        feats["macd"] = macd_ind.macd()
        feats["macd_signal"] = macd_ind.macd_signal()
        feats["macd_diff"] = macd_ind.macd_diff()  # histogram

        # ADX (trend strength)
        adx = ta.trend.ADXIndicator(df["high"], df["low"], close)
        feats["adx"] = adx.adx()
        feats["adx_pos"] = adx.adx_pos()
        feats["adx_neg"] = adx.adx_neg()

        # Ichimoku distance to cloud
        ichi = ta.trend.IchimokuIndicator(df["high"], df["low"])
        feats["ichimoku_a"] = close / ichi.ichimoku_a() - 1
        feats["ichimoku_b"] = close / ichi.ichimoku_b() - 1

        return feats

    def _add_momentum(self, feats: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """RSI, Stochastic, ROC, Williams %R."""
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # RSI at multiple periods
        for period in self.rsi_periods:
            feats[f"rsi_{period}"] = ta.momentum.RSIIndicator(close, window=period).rsi()

        # RSI slope (momentum of momentum)
        rsi14 = ta.momentum.RSIIndicator(close, window=14).rsi()
        feats["rsi_slope"] = rsi14 - rsi14.shift(3)

        # Stochastic
        stoch = ta.momentum.StochasticOscillator(high, low, close)
        feats["stoch_k"] = stoch.stoch()
        feats["stoch_d"] = stoch.stoch_signal()

        # Rate of Change
        for period in [5, 10, 20]:
            feats[f"roc_{period}"] = ta.momentum.ROCIndicator(close, window=period).roc()

        # Williams %R
        feats["williams_r"] = ta.momentum.WilliamsRIndicator(high, low, close).williams_r()

        # CCI
        feats["cci"] = ta.trend.CCIIndicator(high, low, close).cci()

        # Ultimate Oscillator
        feats["ult_osc"] = ta.momentum.UltimateOscillator(high, low, close).ultimate_oscillator()

        return feats

    def _add_volatility(self, feats: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """ATR, Bollinger Bands, realized vol, Keltner Channel."""
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # ATR (normalized by price)
        atr = ta.volatility.AverageTrueRange(high, low, close, window=self.atr_period).average_true_range()
        feats["atr_pct"] = atr / close

        # Bollinger Bands position & width
        bb = ta.volatility.BollingerBands(close, window=self.bbands_period)
        feats["bb_pband"] = bb.bollinger_pband()    # 0=lower, 1=upper band position
        feats["bb_wband"] = bb.bollinger_wband()    # band width (volatility measure)
        feats["bb_hband_ratio"] = close / bb.bollinger_hband() - 1
        feats["bb_lband_ratio"] = close / bb.bollinger_lband() - 1

        # Keltner Channel
        kc = ta.volatility.KeltnerChannel(high, low, close)
        feats["kc_pband"] = kc.keltner_channel_pband()
        feats["kc_wband"] = kc.keltner_channel_wband()

        # Realized volatility (rolling std of returns)
        log_ret = np.log(close / close.shift(1))
        for w in [10, 20, 60]:
            feats[f"realized_vol_{w}"] = log_ret.rolling(w).std() * np.sqrt(252 * 24)

        # Volatility ratio (short vs long vol)
        feats["vol_ratio"] = feats["realized_vol_10"] / feats["realized_vol_60"]

        return feats

    def _add_volume(self, feats: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """OBV, VWAP deviation, volume ratios, money flow."""
        close = df["close"]
        volume = df["volume"]

        # OBV normalized
        obv = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
        feats["obv_slope"] = obv.diff(5) / (obv.rolling(20).std() + 1e-9)

        # Volume moving average ratios
        for period in self.vol_ma_periods:
            vol_ma = volume.rolling(period).mean()
            feats[f"vol_ratio_{period}"] = volume / vol_ma

        # Money Flow Index
        feats["mfi"] = ta.volume.MFIIndicator(
            df["high"], df["low"], close, volume, window=14
        ).money_flow_index()

        # Chaikin Money Flow
        feats["cmf"] = ta.volume.ChaikinMoneyFlowIndicator(
            df["high"], df["low"], close, volume
        ).chaikin_money_flow()

        # VWAP deviation (rolling 20-period)
        typical = (df["high"] + df["low"] + close) / 3
        cum_tp_vol = (typical * volume).rolling(20).sum()
        cum_vol = volume.rolling(20).sum()
        vwap = cum_tp_vol / cum_vol
        feats["vwap_dev"] = close / vwap - 1

        # Log volume
        feats["log_volume"] = np.log1p(volume)
        feats["log_volume_zscore"] = (
            (feats["log_volume"] - feats["log_volume"].rolling(50).mean())
            / (feats["log_volume"].rolling(50).std() + 1e-9)
        )

        return feats

    def _add_microstructure(self, feats: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Bar-level microstructure: range, body ratio, candle patterns."""
        close = df["close"]
        open_ = df["open"]
        high = df["high"]
        low = df["low"]

        # Bar range relative to price
        feats["bar_range_pct"] = (high - low) / close

        # Body vs shadow ratio (encodes candle pattern info)
        body = (close - open_).abs()
        shadow = high - low
        feats["body_ratio"] = body / (shadow + 1e-9)

        # Candle direction
        feats["candle_dir"] = np.sign(close - open_)

        # Upper / lower wick
        upper_wick = high - pd.concat([close, open_], axis=1).max(axis=1)
        lower_wick = pd.concat([close, open_], axis=1).min(axis=1) - low
        feats["upper_wick_pct"] = upper_wick / (shadow + 1e-9)
        feats["lower_wick_pct"] = lower_wick / (shadow + 1e-9)

        # Gap from previous close
        feats["open_gap"] = (open_ - close.shift(1)) / close.shift(1)

        # Close position within bar (0=low, 1=high)
        feats["close_position"] = (close - low) / (high - low + 1e-9)

        return feats

    def _add_lags(self, feats: pd.DataFrame) -> pd.DataFrame:
        """Lag the key feature columns to add temporal memory."""
        core_cols = [c for c in feats.columns if c.startswith(("ret_", "rsi_14", "macd_diff", "atr_pct"))]
        for col in core_cols:
            for lag in self.lag_periods:
                feats[f"{col}_lag{lag}"] = feats[col].shift(lag)
        return feats

    def _add_rolling_stats(self, feats: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling mean, std, skew, kurtosis of returns."""
        log_ret = np.log(df["close"] / df["close"].shift(1))
        for w in self.rolling_windows:
            feats[f"roll_mean_{w}"] = log_ret.rolling(w).mean()
            feats[f"roll_std_{w}"] = log_ret.rolling(w).std()
            feats[f"roll_skew_{w}"] = log_ret.rolling(w).skew()
            feats[f"roll_kurt_{w}"] = log_ret.rolling(w).kurt()

            # Rolling min/max (support/resistance proxy)
            feats[f"roll_max_ratio_{w}"] = df["close"] / df["close"].rolling(w).max()
            feats[f"roll_min_ratio_{w}"] = df["close"] / df["close"].rolling(w).min()

        return feats

    def _add_time_features(self, feats: pd.DataFrame) -> pd.DataFrame:
        """Cyclical time encodings — hour, day of week, month."""
        idx = feats.index
        feats["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
        feats["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
        feats["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
        feats["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
        feats["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
        feats["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
        return feats

    # ─── Utilities ──────────────────────────────────────────────────────────

    def _print_feature_summary(self, feats: pd.DataFrame):
        groups = {
            "Returns": [c for c in feats.columns if c.startswith("ret_")],
            "Trend": [c for c in feats.columns if any(x in c for x in ["ema", "macd", "adx", "ichi"])],
            "Momentum": [c for c in feats.columns if any(x in c for x in ["rsi", "stoch", "roc", "williams", "cci", "ult"])],
            "Volatility": [c for c in feats.columns if any(x in c for x in ["atr", "bb_", "kc_", "vol_"])],
            "Volume": [c for c in feats.columns if any(x in c for x in ["obv", "mfi", "cmf", "vwap", "volume"])],
            "Microstructure": [c for c in feats.columns if any(x in c for x in ["bar_", "body", "candle", "wick", "gap", "close_pos"])],
            "Lags": [c for c in feats.columns if "lag" in c],
            "Rolling Stats": [c for c in feats.columns if "roll_" in c],
            "Time": [c for c in feats.columns if any(x in c for x in ["sin", "cos"])],
        }

        table = Table(title="Feature Summary", show_header=True, header_style="bold magenta")
        table.add_column("Group", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("Sample Features")

        for group, cols in groups.items():
            if cols:
                samples = ", ".join(cols[:3]) + ("..." if len(cols) > 3 else "")
                table.add_row(group, str(len(cols)), samples)

        console.print(table)

    def save(self, feats: pd.DataFrame, symbol: str, timeframe: str, out_dir: str = "features") -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        safe = symbol.replace("/", "_")
        path = out_dir / f"{safe}_{timeframe}_features.parquet"
        feats.to_parquet(path, compression="snappy")
        logger.info(f"Features saved → {path}")
        return path

    @staticmethod
    def load(symbol: str, timeframe: str, feat_dir: str = "features") -> pd.DataFrame:
        safe = symbol.replace("/", "_")
        path = Path(feat_dir) / f"{safe}_{timeframe}_features.parquet"
        if not path.exists():
            raise FileNotFoundError(f"No features at {path}. Run: python main.py features")
        return pd.read_parquet(path)
