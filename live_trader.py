import os
import csv
import time
import signal
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import signal
import threading


import pandas as pd
from loguru import logger
from rich.console import Console
from rich.live import Live
from rich.panel import Panel

console = Console()

try:
    from pybit.unified_trading import HTTP as BybitHTTP
except ImportError:
    logger.warning("pybit not installed. pip install pybit")
    BybitHTTP = None


class LiveTrader:

    def __init__(self, symbol, timeframe, feature_engine, model, config=None):

        cfg = config or {}
        trading_cfg = cfg.get("trading", {})
        exchange_cfg = cfg.get("exchange", {})
        log_cfg = cfg.get("logging", {})

        self.symbol = symbol
        self.timeframe = timeframe
        self.fe = feature_engine
        self.model = model

        # ───────────────── CONFIG ─────────────────

        self.mode = trading_cfg.get("mode", "paper")

        self.check_interval = trading_cfg.get(
            "check_interval_sec", {}
        ).get(timeframe, 60)

        self.lookback = trading_cfg.get(
            "lookback_candles", {}
        ).get(timeframe, 300)

        self.max_positions = trading_cfg.get("max_open_positions", 2)
        self.risk_pct = trading_cfg.get("risk_per_trade_pct", 0.02)
        self.leverage = trading_cfg.get("leverage", 1)

        self.order_type = trading_cfg.get("order_type", "market")
        self.limit_offset = trading_cfg.get("limit_offset_pct", 0.0002)

        self.min_confidence = trading_cfg.get("min_confidence", 0.40)

        self.daily_loss_limit = trading_cfg.get("daily_loss_limit_pct", 0.05)

        self.sl_pct = trading_cfg.get("stop_loss_pct", 0.015)
        self.tp_pct = trading_cfg.get("take_profit_pct", 0.03)

        self.trail_trigger_pct = trading_cfg.get("trail_trigger_pct", 0.008)
        self.trail_distance_pct = trading_cfg.get("trail_distance_pct", 0.006)

        self.paper_balance = trading_cfg.get("paper_balance", 10000)

        # ───────────────── STATE ─────────────────

        self.open_positions = {}

        self.daily_pnl = 0.0
        self.total_pnl = 0.0

        self.trade_count = 0
        self.win_count = 0

        self.session_start = datetime.now(timezone.utc)

        self._last_day = datetime.now(timezone.utc).date()

        self.running = False

        self._tick_count = 0

        self._last_signal = 0
        self._last_conf = 0.0
        self._last_price = 0.0

        self._last_block_reason = ""

        self._last_candle_ts = None
        self._stale_ticks = 0

        # ───────────────── LOGGING ─────────────────

        self.log_dir = Path(log_cfg.get("log_dir", "logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.trade_log_path = self.log_dir / "trades.csv"
        self.signal_log_path = self.log_dir / "signals.csv"

        self._init_csv_files()

        # ───────────────── EXCHANGE ─────────────────

        self.client = self._init_client(exchange_cfg)

        self._print_startup_banner()

    # ═══════════════════════════════════════════════════════
    # MAIN LOOP
    # ═══════════════════════════════════════════════════════

    def run(self):

        self.running = True

    # Register signals only in main thread
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self._shutdown)
            signal.signal(signal.SIGTERM, self._shutdown)

        with Live(self._render_dashboard(), refresh_per_second=1, console=console) as live:

            while self.running:

                try:
                    self._tick(live)
                    time.sleep(self.check_interval)

                except KeyboardInterrupt:
                    self.running = False

                except Exception as e:
                    logger.error(f"Tick error: {e}\n{traceback.format_exc()}")
                    time.sleep(30)

        self._flush_trade_log()

    console.print("[green]Session ended[/green]")

    # ═══════════════════════════════════════════════════════
    # TICK
    # ═══════════════════════════════════════════════════════

    def _tick(self, live):

        self._tick_count += 1

        now = datetime.now(timezone.utc)

        if now.date() != self._last_day:
            self.daily_pnl = 0
            self._last_day = now.date()

        df = self._fetch_recent_candles(self.lookback)

        if df is None or len(df) < 100:
            live.update(self._render_dashboard())
            return

        self._last_price = float(df["close"].iloc[-1])

        self._manage_positions(self._last_price, now)

        feats = self.fe.transform(df)

        if feats is None or len(feats) == 0:
            return

        last_row = feats.iloc[[-1]]

        signals, proba = self.model.predict(last_row)

        sig = int(signals[0])
        conf = float(proba[0].max())

        self._last_signal = sig
        self._last_conf = conf

        block = self._check_trade_gates(sig, conf)

        self._log_signal(now, sig, conf, self._last_price, block)

        if block == "":
            self._execute_trade(sig, conf, self._last_price, now)

        live.update(self._render_dashboard())

    # ═══════════════════════════════════════════════════════
    # TRADE GATES
    # ═══════════════════════════════════════════════════════

    def _check_trade_gates(self, sig, conf):

        if sig == 0:
            return "Signal HOLD"

        if conf < self.min_confidence:
            return f"Low confidence {conf:.1%}"

        if len(self.open_positions) >= self.max_positions:
            return "Max positions reached"

        return ""

    # ═══════════════════════════════════════════════════════
    # EXECUTE TRADE
    # ═══════════════════════════════════════════════════════

    def _execute_trade(self, sig, conf, price, ts):

        side = "Buy" if sig == 1 else "Sell"

        size_usd = self.paper_balance * self.risk_pct

        qty = round(size_usd / price, 6)

        if sig == 1:
            sl = price * (1 - self.sl_pct)
            tp = price * (1 + self.tp_pct)
        else:
            sl = price * (1 + self.sl_pct)
            tp = price * (1 - self.tp_pct)

        pos_id = f"P{ts.strftime('%H%M%S')}"

        self.open_positions[pos_id] = {
            "id": pos_id,
            "side": side,
            "signal": sig,
            "entry_price": price,
            "qty": qty,
            "size_usd": size_usd,
            "confidence": conf,
            "entry_time": ts,
            "sl": sl,
            "tp": tp,
            "bars_held": 0,
        }

        logger.success(
            f"OPEN {side} {self.symbol} entry={price:.2f} SL={sl:.2f} TP={tp:.2f}"
        )

    # ═══════════════════════════════════════════════════════
    # POSITION MANAGEMENT
    # ═══════════════════════════════════════════════════════

    def _manage_positions(self, price, now):

        to_close = []

        for pid, pos in self.open_positions.items():

            sig = pos["signal"]
            entry = pos["entry_price"]

            hit_sl = (sig == 1 and price <= pos["sl"]) or (sig == -1 and price >= pos["sl"])

            hit_tp = (sig == 1 and price >= pos["tp"]) or (sig == -1 and price <= pos["tp"])

            if hit_sl or hit_tp:

                pnl = sig * (price - entry) * pos["qty"]

                self.total_pnl += pnl
                self.daily_pnl += pnl

                self.trade_count += 1

                if pnl > 0:
                    self.win_count += 1

                self._write_trade_row({
                    **pos,
                    "exit_price": price,
                    "exit_time": now.isoformat(),
                    "pnl": pnl
                })

                to_close.append(pid)

        for pid in to_close:
            del self.open_positions[pid]

    # ═══════════════════════════════════════════════════════
    # FETCH CANDLES
    # ═══════════════════════════════════════════════════════

    def _fetch_recent_candles(self, lookback):

        try:

            if self.client:

                resp = self.client.get_kline(
                    category="linear",
                    symbol=self.symbol.replace("/", ""),
                    interval=self._bybit_interval(self.timeframe),
                    limit=lookback,
                )

                klines = resp["result"]["list"]

                df = pd.DataFrame(
                    klines,
                    columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"]
                )

                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

                df.set_index("timestamp", inplace=True)

                return df.astype(float)

            else:

                from fetch_ohlcv import load_ohlcv

                return load_ohlcv(self.symbol, self.timeframe).tail(lookback)

        except Exception as e:

            logger.error(f"Fetch failed: {e}")

            return None

    # ═══════════════════════════════════════════════════════
    # CSV LOGGING
    # ═══════════════════════════════════════════════════════

    def _init_csv_files(self):

        if not self.trade_log_path.exists():

            with open(self.trade_log_path, "w", newline="") as f:

                writer = csv.writer(f)

                writer.writerow([
                    "id", "symbol", "side", "entry_price", "exit_price", "pnl"
                ])

        if not self.signal_log_path.exists():

            with open(self.signal_log_path, "w", newline="") as f:

                writer = csv.writer(f)

                writer.writerow(["timestamp", "signal", "confidence", "price", "action"])

    def _write_trade_row(self, row):

        with open(self.trade_log_path, "a", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                row["id"],
                self.symbol,
                row["side"],
                row["entry_price"],
                row["exit_price"],
                row["pnl"]
            ])

    def _log_signal(self, ts, sig, conf, price, block):

        action = "TRADE" if block == "" else "BLOCKED"

        with open(self.signal_log_path, "a", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                ts.isoformat(),
                sig,
                conf,
                price,
                action
            ])

    # ═══════════════════════════════════════════════════════
    # DASHBOARD
    # ═══════════════════════════════════════════════════════

    def _render_dashboard(self):

        win_rate = self.win_count / self.trade_count if self.trade_count else 0

        body = f"""
Mode: {self.mode}
Symbol: {self.symbol}
TF: {self.timeframe}
Open positions: {len(self.open_positions)}
"""

        return Panel(body, title="QuantBot")

    # ═══════════════════════════════════════════════════════
    # UTILITIES
    # ═══════════════════════════════════════════════════════

    def _init_client(self, exchange_cfg):

        if BybitHTTP is None:
            return None

        key = exchange_cfg.get("api_key")
        secret = exchange_cfg.get("api_secret")
        testnet = exchange_cfg.get("testnet", True)

        if not key or not secret:
            return None

        return BybitHTTP(
            testnet=testnet,
            api_key=key,
            api_secret=secret
        )

    def _shutdown(self, *_):
        self.running = False

    @staticmethod
    def _bybit_interval(tf):

        return {
            "1m": "1",
            "5m": "5",
            "15m": "15",
            "1h": "60",
            "4h": "240",
            "1d": "D"
        }.get(tf, "60")

    def _flush_trade_log(self):

        if not self.open_positions:
            return

        for pos in self.open_positions.values():

            self._write_trade_row({
                **pos,
                "exit_price": self._last_price,
                "pnl": pos["signal"] * (self._last_price - pos["entry_price"]) * pos["qty"]
            })

    def _print_startup_banner(self):

        console.print(
            Panel(
                f"""
QuantBot Live Trader

Symbol: {self.symbol}
TF: {self.timeframe}
Mode: {self.mode}

Check interval: {self.check_interval}s
Lookback: {self.lookback}

Logs: {self.log_dir}
""",
                title="Startup"
            )
        )
