"""
multi_train.py
--------------
Master orchestrator for QuantBot — trains and trades across
multiple symbols * timeframes in one command.

Profitable timeframes (by typical crypto quant research):
  5m  — intraday scalp, high frequency signals
  15m — short swing, good signal/noise balance
  1h  — swing, cleanest signals, most researched
  4h  — position trading, low noise

Top 3 symbols: BTC/USDT, ETH/USDT, SOL/USDT
That gives 12 models total (3 symbols * 4 timeframes).

Commands
--------
  # Fetch raw OHLCV for all symbols * timeframes
  python3 multi_train.py fetch

  # Engineer features for all
  python3 multi_train.py features

  # Generate labels for all
  python3 multi_train.py label

  # Train all 12 models (sequential, GPU memory safe)
  python3 multi_train.py train

  # Run full pipeline fetch→features→label→train for all
  python3 multi_train.py pipeline

  # Backtest all trained models and print comparison table
  python3 multi_train.py backtest

  # Start paper trading — all 12 bots running in parallel threads
  python3 multi_train.py trade

  # Start live trading (set BYBIT_TESTNET=false + real keys)
  python3 multi_train.py trade --mode live

  # Operate on a single symbol/timeframe subset
  python3 multi_train.py train --symbols BTC/USDT --timeframes 1h 4h
  python3 multi_train.py trade --symbols BTC/USDT ETH/USDT --timeframes 1h

  # Check status of all models (trained / missing / last trained)
  python3 multi_train.py status

Flags
-----
  --symbols     Override symbols list   (space-separated, e.g. BTC/USDT ETH/USDT)
  --timeframes  Override timeframes     (e.g. 5m 15m 1h 4h)
  --mode        paper | live            (trade command only)
  --years       Years of history        (fetch command, default per-tf)
  --tune        Enable Optuna HPO       (train command, slow but better)
  --config      Path to config.yaml     (default: config.yaml)
  --workers     Parallel fetch workers  (default: 3)
  --dry-run     Print plan, don't execute
"""

import argparse
import sys
import os
import time
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from tensorboard import summary
from tensorboard import summary
import yaml
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

console = Console()

# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

# Top 3 symbols by volume + research coverage
DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

# Timeframes ordered by profitability evidence in crypto quant literature
# 4h > 1h > 15m > 5m  for trend-following / ML signals
DEFAULT_TIMEFRAMES = ["5m", "15m", "1h", "4h"]

# How many years of history to fetch per timeframe
# More data = better model, but 5m data is huge so we cap it
HISTORY_YEARS = {
    "5m":  0.5,   # ~52k candles per 6 months — enough, manageable size
    "15m": 1.0,
    "1h":  3.0,
    "4h":  5.0,
}

# Training config overrides per timeframe
# Faster timeframes need more estimators (noisier data)
TRAIN_OVERRIDES = {
    "5m":  {"n_estimators": 800,  "num_leaves": 127},
    "15m": {"n_estimators": 600,  "num_leaves": 127},
    "1h":  {"n_estimators": 500,  "num_leaves": 63},
    "4h":  {"n_estimators": 300,  "num_leaves": 63},
}

# Trading check intervals per timeframe (seconds)
CHECK_INTERVALS = {
    "5m":  300,    # 5 min
    "15m": 900,    # 15 min
    "1h":  3600,   # 1 hour
    "4h":  14400,  # 4 hours
}

# Lookback candles per timeframe (enough for all indicators to warm up)
LOOKBACKS = {
    "5m":  400,
    "15m": 300,
    "1h":  250,
    "4h":  200,
}


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        raw = f.read()
    import re
    for match in re.findall(r'\$\{([^}]+)\}', raw):
        raw = raw.replace(f"${{{match}}}", os.getenv(match, ""))
    return yaml.safe_load(raw)


def model_exists(sym: str, tf: str) -> tuple[bool, str]:
    """Returns (exists, path_or_reason)."""
    safe_sym = sym.replace("/", "_")
    path = Path(f"models/saved/{safe_sym}_{tf}_latest.lgb")
    if path.exists():
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        return True, f"{path} (trained {mtime.strftime('%Y-%m-%d %H:%M')})"
    return False, str(path)


def get_pairs(symbols: list, timeframes: list) -> list[tuple]:
    """Returns all (symbol, timeframe) combinations."""
    return [(s, t) for s in symbols for t in timeframes]


def print_plan(command: str, pairs: list[tuple], extra: str = ""):
    table = Table(title=f"📋 Plan: {command} — {len(pairs)} jobs", show_lines=True)
    table.add_column("Symbol",    style="cyan")
    table.add_column("Timeframe", style="yellow")
    table.add_column("Model",     style="green")
    for sym, tf in pairs:
        exists, info = model_exists(sym, tf)
        table.add_row(sym, tf, "✅ trained" if exists else "⬜ not trained")
    console.print(table)
    if extra:
        console.print(f"[dim]{extra}[/dim]\n")


def run_step(label: str, fn, *args, **kwargs):
    """Run a pipeline step with timing + error handling."""
    start = time.time()
    try:
        fn(*args, **kwargs)
        elapsed = time.time() - start
        console.print(f"  [green]✅ {label}[/green] [dim]({elapsed:.1f}s)[/dim]")
        return True
    except Exception as e:
        elapsed = time.time() - start
        console.print(f"  [red]❌ {label} FAILED ({elapsed:.1f}s)[/red]: {e}")
        logger.error(f"{label} failed:\n{traceback.format_exc()}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Step implementations
# ══════════════════════════════════════════════════════════════════════════════

def step_fetch(sym: str, tf: str, cfg: dict, years: float | None = None):
    sys.path.insert(0, "data")
    from fetch_ohlcv import OHLCVFetcher
    fetcher = OHLCVFetcher(
        exchange_id=cfg["exchange"]["name"],
        raw_dir=cfg["data"]["raw_dir"],
    )
    y = years or HISTORY_YEARS.get(tf, 1.0)
    fetcher.fetch(sym, timeframe=tf, years=y)


def step_features(sym: str, tf: str, cfg: dict):
    from fetch_ohlcv import load_ohlcv
    from feature_engine import FeatureEngine
    df    = load_ohlcv(sym, tf, raw_dir=cfg["data"]["raw_dir"])
    eng   = FeatureEngine(config=cfg.get("features", {}))
    feats = eng.transform(df)
    eng.save(feats, sym, tf)


def step_label(sym: str, tf: str, cfg: dict):
    from feature_engine import FeatureEngine
    from fetch_ohlcv import load_ohlcv
    from labeller import Labeller
    df      = load_ohlcv(sym, tf, raw_dir=cfg["data"]["raw_dir"])
    feats   = FeatureEngine.load(sym, tf)
    lbl     = Labeller(config=cfg.get("labels", {}))
    labels  = lbl.generate(df, feats, method=cfg["labels"]["method"])
    lbl.save(labels, sym, tf, cfg["labels"]["method"])


def step_train(sym: str, tf: str, cfg: dict, tune: bool = False):
    from feature_engine import FeatureEngine
    from labeller import Labeller
    from trainer import LGBMTrainer

    feats  = FeatureEngine.load(sym, tf)
    labels = Labeller.load(sym, tf, cfg["labels"]["method"])

    # Merge base config with per-timeframe overrides
    trainer = LGBMTrainer(config=build_trainer_config(cfg, tf, tune))

    results = trainer.train(feats, labels, sym, tf)
    return results


def step_backtest(sym: str, tf: str, cfg: dict) -> dict:
    import pandas as pd
    from fetch_ohlcv import load_ohlcv
    from feature_engine import FeatureEngine
    from trainer import LGBMTrainer
    from backtester import Backtester

    df      = load_ohlcv(sym, tf, raw_dir=cfg["data"]["raw_dir"])
    feats   = FeatureEngine.load(sym, tf)
    trainer = LGBMTrainer(config=build_trainer_config(cfg, tf))

    trainer.load(sym, tf)

    sigs, proba = trainer.predict(feats)
    pred_series = pd.Series(sigs, index=feats.index)
    prob_df     = pd.DataFrame(proba, index=feats.index, columns=["short", "hold", "long"])

    bt      = Backtester(config=cfg.get("backtest", {}))
    results = bt.run(df, pred_series, prob_df, symbol=sym)
    bt.save_results(results, sym)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Commands
# ══════════════════════════════════════════════════════════════════════════════

def cmd_status(args, cfg, symbols, timeframes):
    """Show which models are trained and when."""
    pairs = get_pairs(symbols, timeframes)
    table = Table(title="🤖 QuantBot Model Status", show_lines=True, header_style="bold cyan")
    table.add_column("Symbol",     style="cyan",   width=12)
    table.add_column("Timeframe",  style="yellow",  width=10)
    table.add_column("Status",     style="green",   width=12)
    table.add_column("Trained At",               width=20)
    table.add_column("Model File",               width=40)

    for sym, tf in pairs:
        safe_sym = sym.replace("/", "_")
        path     = Path(f"models/saved/{safe_sym}_{tf}_latest.lgb")
        meta_p   = Path(f"models/saved/{safe_sym}_{tf}_latest_meta.json")

        if path.exists():
            mtime    = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            acc_str  = ""
            if meta_p.exists():
                import json
                try:
                    meta    = json.loads(meta_p.read_text())
                    avg_acc = meta.get("avg_accuracy", 0)
                    acc_str = f"acc={avg_acc:.1%}"
                except Exception:
                    pass
            table.add_row(sym, tf, f"[green]✅ Ready[/green] {acc_str}", mtime, str(path))
        else:
            table.add_row(sym, tf, "[red]❌ Missing[/red]", "—", str(path))

    console.print(table)
    console.print(f"\n[dim]Run: python3 multi_train.py pipeline  — to train all missing models[/dim]")


def cmd_fetch_all(args, cfg, symbols, timeframes):
    pairs  = get_pairs(symbols, timeframes)
    years  = args.years
    console.print(Rule("[cyan]📥 Fetching OHLCV data[/cyan]"))
    console.print(f"  {len(pairs)} jobs: {len(symbols)} symbols * {len(timeframes)} timeframes\n")

    if args.dry_run:
        print_plan("fetch", pairs)
        return

    ok = fail = 0
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  BarColumn(), TimeElapsedColumn(), console=console) as prog:
        task = prog.add_task("Fetching...", total=len(pairs))

        def do_fetch(sym_tf):
            sym, tf = sym_tf
            try:
                step_fetch(sym, tf, cfg, years)
                return sym, tf, True, None
            except Exception as e:
                return sym, tf, False, str(e)

        workers = min(args.workers, len(pairs))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(do_fetch, p): p for p in pairs}
            for fut in as_completed(futures):
                sym, tf, success, err = fut.result()
                prog.advance(task)
                if success:
                    ok += 1
                    logger.info(f"✅ Fetched {sym} {tf}")
                else:
                    fail += 1
                    logger.error(f"❌ {sym} {tf}: {err}")

    console.print(f"\n[green]✅ {ok} fetched[/green]  [red]❌ {fail} failed[/red]")


def cmd_features_all(args, cfg, symbols, timeframes):
    pairs = get_pairs(symbols, timeframes)
    console.print(Rule("[cyan]⚙️  Engineering Features[/cyan]"))

    if args.dry_run:
        print_plan("features", pairs)
        return

    ok = fail = 0
    for sym, tf in pairs:
        label = f"features {sym} {tf}"
        console.print(f"  [dim]→ {label}[/dim]")
        if run_step(label, step_features, sym, tf, cfg):
            ok += 1
        else:
            fail += 1

    console.print(f"\n[green]✅ {ok} done[/green]  [red]❌ {fail} failed[/red]")


def cmd_label_all(args, cfg, symbols, timeframes):
    pairs = get_pairs(symbols, timeframes)
    console.print(Rule("[green]🏷️  Generating Labels[/green]"))

    if args.dry_run:
        print_plan("label", pairs)
        return

    ok = fail = 0
    for sym, tf in pairs:
        label = f"label {sym} {tf}"
        console.print(f"  [dim]→ {label}[/dim]")
        if run_step(label, step_label, sym, tf, cfg):
            ok += 1
        else:
            fail += 1

    console.print(f"\n[green]✅ {ok} done[/green]  [red]❌ {fail} failed[/red]")


def cmd_train_all(args, cfg, symbols, timeframes):
    pairs = get_pairs(symbols, timeframes)
    console.print(Rule("[magenta]🧠 Training Models[/magenta]"))
    console.print(f"  {len(pairs)} models: {symbols} * {timeframes}")
    console.print(f"  HPO tuning: {'[yellow]ON (slow)[/yellow]' if args.tune else '[dim]OFF[/dim]'}\n")

    if args.dry_run:
        print_plan("train", pairs)
        return

    results_table = Table(title="Training Results", show_lines=True, header_style="bold magenta")
    results_table.add_column("Symbol")
    results_table.add_column("TF")
    results_table.add_column("Status")
    results_table.add_column("Accuracy")
    results_table.add_column("F1")
    results_table.add_column("Samples")
    results_table.add_column("Time")

    ok = fail = 0
    # Sequential — GPU can only do one model at a time
    for sym, tf in pairs:
        console.print(Rule(f"[magenta]{sym} {tf}[/magenta]", style="dim"))
        t0 = time.time()
        try:
            results = step_train(sym, tf, cfg, tune=args.tune)
            elapsed = time.time() - t0
            cv = results.get("cv_metrics", [])
            avg_acc = sum(f["accuracy"] for f in cv) / len(cv) if cv else 0
            avg_f1  = sum(f["f1_macro"] for f in cv) / len(cv) if cv else 0
            results_table.add_row(
                sym, tf,
                "[green]✅ OK[/green]",
                f"{avg_acc:.1%}", f"{avg_f1:.3f}",
                f"{results.get('n_samples', 0):,}",
                f"{elapsed:.0f}s",
            )
            ok += 1
        except Exception as e:
            elapsed = time.time() - t0
            results_table.add_row(sym, tf, "[red]❌ FAILED[/red]", "—", "—", "—", f"{elapsed:.0f}s")
            logger.error(f"Train {sym} {tf}:\n{traceback.format_exc()}")
            fail += 1

    console.print(results_table)
    console.print(f"\n[green]✅ {ok} trained[/green]  [red]❌ {fail} failed[/red]")


def cmd_backtest_all(args, cfg, symbols, timeframes):
    pairs = get_pairs(symbols, timeframes)
    console.print(Rule("[yellow]📊 Backtesting All Models[/yellow]"))

    if args.dry_run:
        print_plan("backtest", pairs)
        return

    summary = Table(title="Backtest Summary", show_lines=True, header_style="bold yellow")
    summary.add_column("Symbol")
    summary.add_column("TF")
    summary.add_column("Return")
    summary.add_column("Sharpe")
    summary.add_column("Win Rate")
    summary.add_column("Max DD")
    summary.add_column("Trades")

    for sym, tf in pairs:
        exists, _ = model_exists(sym, tf)
        if not exists:
            console.print(f"  [yellow]⚠ Skipping {sym} {tf} — model not trained[/yellow]")
            continue
        try:
            console.print(f"  [dim]→ backtest {sym} {tf}[/dim]")
            res = step_backtest(sym, tf, cfg)
            if isinstance(res, dict):
                m = res.get("metrics", {})
            else:
                m = getattr(res, "metrics", {})
            total_return = m.get("total_return_pct") or m.get("return", 0)/100
            sharpe = m.get("sharpe_ratio") or m.get("sharpe", 0)
            win_rate = m.get("win_rate_pct", 0)
            max_dd = m.get("max_drawdown_pct") or m.get("max_dd", 0)/100
            trades = m.get("n_trades") or m.get("trades", 0)
            

            summary.add_row(
    sym, tf,
    f"[{'green' if total_return/100 > 0 else 'red'}]{total_return/100:.1%}[/{'green' if total_return > 0 else 'red'}]",
    f"{sharpe:.2f}",
    f"{win_rate/100:.1%}",
    f"[red]{max_dd/100:.1%}[/red]",
    str(trades),
)

        except Exception as e:
            summary.add_row(sym, tf, "[red]ERROR[/red]", "—", "—", "—", "—")
            logger.error(f"Backtest {sym} {tf}: {e}")

    console.print(summary)


def cmd_pipeline_all(args, cfg, symbols, timeframes):
    """Full pipeline: fetch → features → label → train for all pairs."""
    pairs = get_pairs(symbols, timeframes)
    console.print(Panel.fit(
        f"[bold cyan]Full Multi-Symbol Pipeline[/bold cyan]\n"
        f"Symbols:    {', '.join(symbols)}\n"
        f"Timeframes: {', '.join(timeframes)}\n"
        f"Models:     {len(pairs)} total\n"
        f"Steps:      fetch → features → label → train",
        title="🔄 Pipeline",
        border_style="cyan",
    ))

    if args.dry_run:
        print_plan("pipeline", pairs)
        return

    t_start = time.time()

    console.print(Rule("[cyan]Step 1/4 — Fetch[/cyan]"))
    cmd_fetch_all(args, cfg, symbols, timeframes)

    console.print(Rule("[cyan]Step 2/4 — Features[/cyan]"))
    cmd_features_all(args, cfg, symbols, timeframes)

    console.print(Rule("[cyan]Step 3/4 — Labels[/cyan]"))
    cmd_label_all(args, cfg, symbols, timeframes)

    console.print(Rule("[cyan]Step 4/4 — Train[/cyan]"))
    cmd_train_all(args, cfg, symbols, timeframes)

    elapsed = time.time() - t_start
    console.print(Panel.fit(
        f"[bold green]✅ Pipeline complete![/bold green]\n"
        f"Total time: {elapsed/60:.1f} minutes\n"
        f"Models ready in: models/saved/",
        border_style="green",
    ))


def cmd_trade_all(args, cfg, symbols, timeframes):
    """
    Launch one trading bot per (symbol, timeframe) pair, all in parallel threads.
    Only launches bots for pairs that have a trained model.
    """
    mode  = args.mode or cfg["inference"].get("mode", "paper")
    pairs = get_pairs(symbols, timeframes)

    # Filter to only pairs with trained models
    ready = [(s, t) for s, t in pairs if model_exists(s, t)[0]]
    missing = [(s, t) for s, t in pairs if not model_exists(s, t)[0]]

    console.print(Panel.fit(
        f"[bold]Multi-Bot Trading[/bold]\n"
        f"Mode:    [{('green' if mode=='paper' else 'red')}]{mode.upper()}[/{'green' if mode=='paper' else 'red'}]\n"
        f"Ready:   {len(ready)} bots\n"
        f"Missing: {len(missing)} models (skipped)\n\n"
        + ("\n".join(f"  ✅ {s} {t}" for s, t in ready))
        + ("\n" + "\n".join(f"  [dim]❌ {s} {t} — no model[/dim]" for s, t in missing) if missing else ""),
        title="🤖 QuantBot Fleet",
        border_style="green" if mode == "paper" else "red",
    ))

    if not ready:
        console.print("[red]No trained models found. Run: python3 multi_train.py pipeline[/red]")
        return

    if args.dry_run:
        console.print("[dim]Dry run — would launch bots for:[/dim]")
        for s, t in ready:
            console.print(f"  {s} {t}  interval={CHECK_INTERVALS[t]}s")
        return

    if mode == "live":
        console.print("\n[bold red]⚠  LIVE MODE — real orders will be placed![/bold red]")
        console.print("[yellow]Press ENTER to confirm, or Ctrl+C to abort...[/yellow]")
        try:
            input()
        except KeyboardInterrupt:
            console.print("Aborted.")
            return

    # ── Launch one thread per bot ────────────────────────────────────────────
    threads  = []
    stop_evt = threading.Event()

    def launch_bot(sym: str, tf: str):
        """Thread target — loads model + runs trading loop."""
        bot_label = f"{sym} {tf}"
        try:
            from feature_engine import FeatureEngine
            from trainer import LGBMTrainer
            from live_trader import LiveTrader

            trainer = LGBMTrainer(config=build_trainer_config(cfg, tf))

            trainer.load(sym, tf)

            engine = FeatureEngine(config=cfg.get("features", {}))

            trader = LiveTrader(sym, tf, engine, trainer, config=cfg)


            logger.info(f"[BOT] Started: {bot_label}")
            trader.run()
        except Exception as e:
            logger.error(f"[BOT] {bot_label} crashed: {e}\n{traceback.format_exc()}")

    console.print(f"\n[bold green]Launching {len(ready)} bots...[/bold green]")
    console.print("[yellow]Press Ctrl+C to stop ALL bots gracefully[/yellow]\n")

    for sym, tf in ready:
        t = threading.Thread(target=launch_bot, args=(sym, tf), name=f"bot-{sym}-{tf}", daemon=True)
        threads.append(t)
        t.start()
        time.sleep(1)   # stagger starts to avoid API rate limits

    # ── Status loop — print a summary every 60s ──────────────────────────────
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(60)
            alive = sum(1 for t in threads if t.is_alive())
            console.print(f"[dim]{datetime.now(timezone.utc).strftime('%H:%M UTC')} — {alive}/{len(threads)} bots running[/dim]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping all bots...[/yellow]")
        stop_evt.set()
        for t in threads:
            t.join(timeout=10)
        console.print("[green]All bots stopped.[/green]")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def build_trainer_config(cfg, tf, tune=False):
    trainer_cfg = {**cfg.get("model", {})}

    base_params = dict(trainer_cfg.get("params") or {})
    base_params.update(TRAIN_OVERRIDES.get(tf, {}))

    trainer_cfg["params"] = base_params
    trainer_cfg["tune_hyperparams"] = tune

    return trainer_cfg


def main():
    parser = argparse.ArgumentParser(
        prog="multi_train.py",
        description="QuantBot — Multi-Symbol * Multi-Timeframe Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 multi_train.py status
  python3 multi_train.py pipeline
  python3 multi_train.py train --tune
  python3 multi_train.py trade
  python3 multi_train.py trade --mode live
  python3 multi_train.py train --symbols BTC/USDT --timeframes 1h 4h
  python3 multi_train.py backtest --symbols BTC/USDT ETH/USDT
        """,
    )

    parser.add_argument(
        "command",
        choices=["fetch", "features", "label", "train", "backtest", "trade", "pipeline", "status"],
    )
    parser.add_argument("--symbols",    nargs="+", default=None, help="Override symbols (e.g. BTC/USDT ETH/USDT)")
    parser.add_argument("--timeframes", nargs="+", default=None, help="Override timeframes (e.g. 1h 4h)")
    parser.add_argument("--mode",       default=None,  help="paper | live (trade command)")
    parser.add_argument("--years",      type=float, default=None, help="Years of history to fetch")
    parser.add_argument("--tune",       action="store_true", help="Enable Optuna HPO (slow)")
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--workers",    type=int, default=3, help="Parallel fetch workers")
    parser.add_argument("--dry-run",    action="store_true", help="Print plan without executing")

    args = parser.parse_args()

    # Load config
    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        logger.error(f"Config not found: {args.config}  (try --config config.yaml)")
        sys.exit(1)

    # Resolve symbols + timeframes
    symbols    = args.symbols    or DEFAULT_SYMBOLS
    timeframes = args.timeframes or DEFAULT_TIMEFRAMES

    # Validate
    valid_tfs = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"}
    for tf in timeframes:
        if tf not in valid_tfs:
            console.print(f"[red]Invalid timeframe: {tf}. Valid: {sorted(valid_tfs)}[/red]")
            sys.exit(1)

    # Print header
    console.print(Panel.fit(
        f"[bold cyan]QuantBot Multi-Trainer[/bold cyan]\n"
        f"Symbols: {', '.join(symbols)}\n"
        f"Timeframes: {', '.join(timeframes)}\n"
        f"Command: [bold]{args.command}[/bold]"
        + ("  [yellow](DRY RUN)[/yellow]" if args.dry_run else ""),
        border_style="cyan",
    ))

    dispatch = {
        "status":   cmd_status,
        "fetch":    cmd_fetch_all,
        "features": cmd_features_all,
        "label":    cmd_label_all,
        "train":    cmd_train_all,
        "backtest": cmd_backtest_all,
        "pipeline": cmd_pipeline_all,
        "trade":    cmd_trade_all,
    }

    dispatch[args.command](args, cfg, symbols, timeframes)


if __name__ == "__main__":
    main()