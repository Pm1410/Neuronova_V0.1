"""
labels/labeller.py
------------------
Generates ML training labels from OHLCV data.

Supported methods:
1. Triple-Barrier (Marcos Lopez de Prado's method)
   - Label = which barrier is hit first: upper (+1), lower (-1), vertical (0)
2. Fixed Horizon
   - Label = sign of forward return at fixed horizon
3. Trend-Following
   - Label based on future directional trend

Triple-Barrier is the most rigorous — avoids look-ahead and captures
asymmetric risk. It's the default and recommended method.
"""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table

console = Console()


class Labeller:
    """
    Generates training labels with zero lookahead bias.
    
    Triple-Barrier label:
        +1 = Long signal (upper barrier hit first)
        -1 = Short signal (lower barrier hit first)
         0 = Hold/neutral (vertical barrier hit first / time out)
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.method = cfg.get("method", "triple_barrier")
        self.pt_sl_ratio = cfg.get("pt_sl_ratio", 1.0)
        self.sl_atr_mult = cfg.get("stop_loss_atr_mult", 1.5)
        self.max_holding = cfg.get("max_holding_bars", 24)
        self.min_ret = cfg.get("min_return_threshold", 0.003)

    def generate(
        self,
        df_raw: pd.DataFrame,
        df_features: pd.DataFrame | None = None,
        method: str | None = None,
    ) -> pd.Series:
        """
        Generate labels aligned to the feature index.
        
        Args:
            df_raw:      Raw OHLCV DataFrame
            df_features: Feature DataFrame (labels will match this index)
            method:      Override config method
        
        Returns:
            Series of labels: -1, 0, +1
        """
        method = method or self.method
        idx = df_features.index if df_features is not None else df_raw.index

        if method == "triple_barrier":
            labels = self._triple_barrier(df_raw, idx)
        elif method == "fixed_horizon":
            labels = self._fixed_horizon(df_raw, idx)
        elif method == "trend":
            labels = self._trend_label(df_raw, idx)
        else:
            raise ValueError(f"Unknown labelling method: {method}")

        labels = labels.loc[idx].dropna()
        self._print_distribution(labels, method)
        return labels

    # ─── Label Methods ───────────────────────────────────────────────────────

    def _triple_barrier(self, df: pd.DataFrame, idx: pd.Index) -> pd.Series:
        """
        Marcos Lopez de Prado's Triple-Barrier labelling.
        
        For each bar:
        1. Compute ATR-based stop loss and profit target
        2. Scan forward bars until a barrier is hit
        3. Label accordingly
        """
        logger.info("Generating Triple-Barrier labels...")
        close = df["close"]
        
        # Compute ATR for dynamic barrier sizing
        high = df["high"]
        low = df["low"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)
        atr = tr.ewm(span=14, adjust=False).mean()

        labels = pd.Series(index=idx, dtype=float)
        close_arr = close.values
        atr_arr = atr.values
        ts_arr = close.index

        for i, ts in enumerate(idx):
            if ts not in close.index:
                continue
            bar_i = close.index.get_loc(ts)
            price_0 = close_arr[bar_i]
            atr_0 = atr_arr[bar_i]

            sl = atr_0 * self.sl_atr_mult
            pt = sl * self.pt_sl_ratio

            # Must be meaningful signal
            if pt / price_0 < self.min_ret:
                labels[ts] = 0
                continue

            upper = price_0 + pt
            lower = price_0 - sl
            label = 0  # default: vertical barrier

            # Scan forward up to max_holding bars
            end_i = min(bar_i + self.max_holding + 1, len(close_arr))
            high_arr = df["high"].values
            low_arr  = df["low"].values

            for j in range(bar_i + 1, end_i):
                if high_arr[j] >= upper:
                    label = 1
                    break
                if low_arr[j] <= lower:
                    label = -1
                    break


            labels[ts] = label

        return labels

    def _fixed_horizon(self, df: pd.DataFrame, idx: pd.Index, horizon: int = 12) -> pd.Series:
        """
        Simple forward return sign labelling.
        +1 if ret > threshold, -1 if ret < -threshold, else 0
        """
        logger.info(f"Generating Fixed-Horizon labels (h={horizon})...")
        close = df["close"]
        fwd_ret = close.shift(-horizon) / close - 1

        labels = pd.Series(index=close.index, dtype=float)
        labels[fwd_ret > self.min_ret] = 1
        labels[fwd_ret < -self.min_ret] = -1
        labels[(fwd_ret >= -self.min_ret) & (fwd_ret <= self.min_ret)] = 0

        return labels.reindex(idx)

    def _trend_label(self, df: pd.DataFrame, idx: pd.Index, window: int = 10) -> pd.Series:
        """
        Label based on whether price is trending up/down over next N bars.
        Uses linear regression slope.
        """
        logger.info(f"Generating Trend labels (w={window})...")
        close = df["close"]
        x = np.arange(window)

        def slope(arr):
            if len(arr) < window:
                return np.nan
            s, _ = np.polyfit(x, arr[-window:], 1)
            return s / arr[-1]  # normalize by price

        # Compute forward slopes by reversing
        slopes = close[::-1].rolling(window).apply(slope, raw=True)[::-1]
        threshold = slopes.std() * 0.5

        labels = pd.Series(0.0, index=close.index)
        labels[slopes > threshold] = 1
        labels[slopes < -threshold] = -1

        return labels.reindex(idx)

    # ─── Utilities ──────────────────────────────────────────────────────────

    def _print_distribution(self, labels: pd.Series, method: str):
        counts = labels.value_counts().sort_index()
        total = len(labels)

        table = Table(title=f"Label Distribution ({method})", header_style="bold green")
        table.add_column("Label", style="cyan")
        table.add_column("Description")
        table.add_column("Count", justify="right")
        table.add_column("Pct", justify="right")

        descs = {-1.0: "SHORT", 0.0: "HOLD", 1.0: "LONG"}
        colors = {-1.0: "red", 0.0: "yellow", 1.0: "green"}

        for val, desc in descs.items():
            cnt = counts.get(val, 0)
            pct = cnt / total * 100
            color = colors[val]
            table.add_row(
                f"[{color}]{int(val):+d}[/{color}]",
                f"[{color}]{desc}[/{color}]",
                str(cnt),
                f"{pct:.1f}%",
            )

        table.add_row("[bold]Total[/bold]", "", str(total), "100%")
        console.print(table)

        # Warn if highly imbalanced
        max_pct = counts.max() / total
        if max_pct > 0.7:
            console.print(
                f"[yellow]⚠ Warning: Labels are imbalanced ({max_pct:.0%} majority class). "
                "Consider class_weight or resampling.[/yellow]"
            )

    def save(
        self,
        labels: pd.Series,
        symbol: str,
        timeframe: str,
        method: str,
        out_dir: str = "labels",
    ) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        safe = symbol.replace("/", "_")
        path = out_dir / f"{safe}_{timeframe}_{method}_labels.parquet"
        labels.to_frame("label").to_parquet(path)
        logger.info(f"Labels saved → {path}")
        return path

    @staticmethod
    def load(
        symbol: str, timeframe: str, method: str = "triple_barrier", label_dir: str = "labels"
    ) -> pd.Series:
        safe = symbol.replace("/", "_")
        path = Path(label_dir) / f"{safe}_{timeframe}_{method}_labels.parquet"
        if not path.exists():
            raise FileNotFoundError(f"No labels at {path}. Run: python main.py label")
        return pd.read_parquet(path)["label"]
