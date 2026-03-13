"""
backtest/backtester.py
----------------------
Vectorized backtesting engine with:
- Realistic slippage and commission modeling
- Position sizing (fixed % risk per trade)
- Stop-loss and take-profit enforcement
- Long and short positions
- Comprehensive performance metrics:
  * Sharpe, Sortino, Calmar ratios
  * Max drawdown, drawdown duration
  * Win rate, profit factor
  * PnL curves + trade log

Runs on model predictions from the feature DataFrame.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@dataclass
class BacktestConfig:
    initial_capital: float = 10_000
    position_size_pct: float = 0.1    # 10% per trade
    commission_pct: float = 0.001     # 0.1%
    slippage_pct: float = 0.0005      # 0.05%
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    min_confidence: float = 0.55      # Min probability to enter
    max_positions: int = 1            # Concurrent positions


@dataclass
class BacktestResults:
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series = field(default_factory=pd.Series)
    metrics: dict = field(default_factory=dict)


class Backtester:
    """
    Event-driven vectorized backtester.
    Simulates realistic execution with slippage, fees, and risk management.
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.cfg = BacktestConfig(
            initial_capital=cfg.get("initial_capital", 10_000),
            position_size_pct=cfg.get("position_size_pct", 0.1),
            commission_pct=cfg.get("commission_pct", 0.001),
            slippage_pct=cfg.get("slippage_pct", 0.0005),
            stop_loss_pct=cfg.get("stop_loss_pct", 0.02),
            take_profit_pct=cfg.get("take_profit_pct", 0.04),
            min_confidence=cfg.get("min_confidence", 0.55),
        )

    def run(
        self,
        df: pd.DataFrame,
        predictions: pd.Series,
        probabilities: pd.DataFrame,  # columns: [short_prob, hold_prob, long_prob]
        symbol: str = "BTC/USDT",
    ) -> BacktestResults:
        """
        Run the backtest.
        
        Args:
            df:           OHLCV DataFrame
            predictions:  Series of -1/0/1 signals aligned to df
            probabilities: DataFrame of class probabilities
            symbol:       For logging
        
        Returns:
            BacktestResults with trades, equity curve, metrics
        """
        console.print(Panel.fit(
            f"[bold yellow]Running Backtest[/bold yellow]\n"
            f"Symbol: {symbol} | "
            f"Capital: ${self.cfg.initial_capital:,.0f} | "
            f"Bars: {len(df):,}",
            title="📊 Backtesting"
        ))

        trades = []
        capital = self.cfg.initial_capital
        equity = {df.index[0]: capital}
        position = None  # current open position dict

        # Align all series
        common = df.index.intersection(predictions.index)
        df_bt = df.loc[common]
        preds = predictions.loc[common]
        probs = probabilities.loc[common] if probabilities is not None else None

        for i, (ts, row) in enumerate(df_bt.iterrows()):
            price = row["close"]
            signal = preds.loc[ts]

            # Check if we have an open position → manage it
            if position is not None:
                pnl, closed = self._check_exit(position, price, ts)
                if closed:
                    capital += pnl
                    trades.append({**position, "exit_ts": ts, "exit_price": price, "pnl": pnl})
                    position = None

            # Enter new position if signal and no position open
            if position is None and signal != 0:
                confidence = self._get_confidence(probs, ts, signal)
                if confidence >= self.cfg.min_confidence:
                    position = self._open_position(capital, price, signal, ts, confidence)

            if position:
                unrealized = self._unrealized_pnl(position, price)
                equity[ts] = capital + unrealized
            else:
                equity[ts] = capital

        # Close any open position at end
        if position is not None:
            last_price = df_bt["close"].iloc[-1]
            pnl, _ = self._check_exit(position, last_price, df_bt.index[-1], force=True)

            capital += pnl   # correct
            trades.append({
        **position,
        "exit_ts": df_bt.index[-1],
        "exit_price": last_price,
        "pnl": pnl
    })


            
        equity_series = pd.Series(equity)
        trade_df = pd.DataFrame(trades)
        returns = equity_series.pct_change().dropna()
        metrics = self._compute_metrics(equity_series, trade_df, returns)

        results = BacktestResults(
            trades=trade_df,
            equity_curve=equity_series,
            returns=returns,
            metrics=metrics,
        )
        self._print_metrics(metrics, symbol)
        return results

    # ─── Position Management ─────────────────────────────────────────────────

    def _open_position(self, capital, price, signal, ts, confidence) -> dict:
        size_usd = capital * self.cfg.position_size_pct
        exec_price = price * (1 + self.cfg.slippage_pct * signal)
        fee = size_usd * self.cfg.commission_pct
        qty = size_usd / exec_price
        capital -= fee

        
        sl = exec_price * (1 - signal * self.cfg.stop_loss_pct)
        tp = exec_price * (1 + signal * self.cfg.take_profit_pct)

        return {
            "entry_ts": ts,
            "entry_price": exec_price,
            "signal": signal,
            "qty": qty,
            "size_usd": size_usd,
            "stop_loss": sl,
            "take_profit": tp,
            "entry_fee": fee,
            "confidence": confidence,
        }

    def _check_exit(self, pos, price, ts, force=False) -> tuple[float, bool]:
        signal = pos["signal"]
        pnl = signal * (price - pos["entry_price"]) * pos["qty"]
        exit_fee = pos["size_usd"] * self.cfg.commission_pct
        net_pnl = pnl - pos["entry_fee"] - exit_fee

        hit_sl = (signal == 1 and price <= pos["stop_loss"]) or \
                 (signal == -1 and price >= pos["stop_loss"])
        hit_tp = (signal == 1 and price >= pos["take_profit"]) or \
                 (signal == -1 and price <= pos["take_profit"])

        if hit_sl or hit_tp or force:
            return net_pnl, True
        return net_pnl, False

    def _unrealized_pnl(self, pos, price) -> float:
        if pos is None:
            return 0.0
        return pos["signal"] * (price - pos["entry_price"]) * pos["qty"]

    def _get_confidence(self, probs, ts, signal) -> float:
        if probs is None:
            return 1.0
        try:
            row = probs.loc[ts]
            if signal == 1:
                return float(row.iloc[2])   # long prob
            elif signal == -1:
                return float(row.iloc[0])   # short prob
        except Exception:
            pass
        return 0.6

    # ─── Metrics ─────────────────────────────────────────────────────────────

    def _compute_metrics(
        self, equity: pd.Series, trades: pd.DataFrame, returns: pd.Series
    ) -> dict:
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        n_trades = len(trades)

        # Annualization factor (assume hourly bars unless index tells us otherwise)
        ann_factor = 24 * 365

        sharpe = self._sharpe(returns, ann_factor)
        sortino = self._sortino(returns, ann_factor)
        max_dd, dd_duration = self._max_drawdown(equity)
        calmar = (total_return / abs(max_dd)) if max_dd != 0 else 0

        if n_trades > 0:
            win_trades = trades[trades["pnl"] > 0]
            lose_trades = trades[trades["pnl"] <= 0]
            win_rate = len(win_trades) / n_trades
            avg_win = win_trades["pnl"].mean() if len(win_trades) > 0 else 0
            avg_loss = lose_trades["pnl"].mean() if len(lose_trades) > 0 else 0
            gross_profit = win_trades["pnl"].sum()
            gross_loss = abs(lose_trades["pnl"].sum())
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.inf
            long_trades = trades[trades["signal"] == 1]
            short_trades = trades[trades["signal"] == -1]
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0
            long_trades = short_trades = pd.DataFrame()

        return {
            "total_return_pct": total_return * 100,
            "final_equity": equity.iloc[-1],
            "n_trades": n_trades,
            "n_long": len(long_trades),
            "n_short": len(short_trades),
            "win_rate_pct": win_rate * 100,
            "profit_factor": profit_factor,
            "avg_win_usd": avg_win,
            "avg_loss_usd": avg_loss,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "max_drawdown_pct": max_dd * 100,
            "max_dd_duration_bars": dd_duration,
            "total_pnl_usd": trades["pnl"].sum() if n_trades > 0 else 0,
        }

    @staticmethod
    def _sharpe(returns: pd.Series, ann_factor: int) -> float:
        if returns.std() == 0:
            return 0.0
        return float((returns.mean() / returns.std()) * np.sqrt(ann_factor))

    @staticmethod
    def _sortino(returns: pd.Series, ann_factor: int) -> float:
        down = returns[returns < 0]
        if len(down) == 0 or down.std() == 0:
            return 0.0
        return float((returns.mean() / down.std()) * np.sqrt(ann_factor))

    @staticmethod
    def _max_drawdown(equity: pd.Series) -> tuple[float, int]:
        peak = equity.cummax()
        dd = (equity - peak) / peak
        max_dd = float(dd.min())
        # Duration of longest drawdown
        in_dd = dd < 0
        dd_duration = 0
        curr = 0
        for v in in_dd:
            curr = curr + 1 if v else 0
            dd_duration = max(dd_duration, curr)
        return max_dd, dd_duration

    # ─── Output ──────────────────────────────────────────────────────────────

    def _print_metrics(self, m: dict, symbol: str):
        ret_color = "green" if m["total_return_pct"] > 0 else "red"
        sharpe_color = "green" if m["sharpe_ratio"] > 1 else "yellow" if m["sharpe_ratio"] > 0 else "red"

        table = Table(title=f"Backtest Results — {symbol}", header_style="bold cyan")
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right")

        rows = [
            ("Total Return", f"[{ret_color}]{m['total_return_pct']:+.2f}%[/{ret_color}]"),
            ("Final Equity", f"${m['final_equity']:,.2f}"),
            ("Total PnL", f"${m['total_pnl_usd']:,.2f}"),
            ("Sharpe Ratio", f"[{sharpe_color}]{m['sharpe_ratio']:.3f}[/{sharpe_color}]"),
            ("Sortino Ratio", f"{m['sortino_ratio']:.3f}"),
            ("Calmar Ratio", f"{m['calmar_ratio']:.3f}"),
            ("Max Drawdown", f"[red]{m['max_drawdown_pct']:.2f}%[/red]"),
            ("Max DD Duration", f"{m['max_dd_duration_bars']} bars"),
            ("Total Trades", str(m["n_trades"])),
            ("Long / Short", f"{m['n_long']} / {m['n_short']}"),
            ("Win Rate", f"{m['win_rate_pct']:.1f}%"),
            ("Profit Factor", f"{m['profit_factor']:.3f}"),
            ("Avg Win", f"${m['avg_win_usd']:.2f}"),
            ("Avg Loss", f"${m['avg_loss_usd']:.2f}"),
        ]

        for k, v in rows:
            table.add_row(k, v)

        console.print(table)

    def save_results(self, results: BacktestResults, symbol: str, out_dir: str = "backtest") -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        safe = symbol.replace("/", "_")

        # Save trades CSV
        trades_path = out_dir / f"{safe}_trades.csv"
        results.trades.to_csv(trades_path)

        # Save equity curve
        equity_path = out_dir / f"{safe}_equity.parquet"
        results.equity_curve.to_frame("equity").to_parquet(equity_path)

        # Save metrics JSON
        import json
        metrics_path = out_dir / f"{safe}_metrics.json"
        metrics_path.write_text(json.dumps(results.metrics, indent=2, default=str))

        logger.info(f"Backtest results saved → {out_dir}/")
        return out_dir
