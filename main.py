"""
main.py
-------
CLI orchestrator for the full QuantBot pipeline.

Commands:
  python3 main.py fetch    -- Download OHLCV data
  python3 main.py features -- Engineer features
  python3 main.py label    -- Generate training labels
  python3 main.py train    -- Train LightGBM model
  python3 main.py backtest -- Backtest model
  python3 main.py trade    -- Start live/paper trading
  python3 main.py pipeline -- Run full pipeline end-to-end
"""

import argparse
import sys
from pathlib import Path

import yaml
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

console = Console()

# ─── Load Config ──────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        raw = f.read()
    import os
    # Expand env vars like ${BYBIT_API_KEY}
    import re
    for match in re.findall(r'\$\{([^}]+)\}', raw):
        raw = raw.replace(f"${{{match}}}", os.getenv(match, ""))
    return yaml.safe_load(raw)


# ─── Command Handlers ─────────────────────────────────────────────────────────

def cmd_fetch(args, cfg):
    from fetch_ohlcv import OHLCVFetcher

    symbols = [args.symbol] if args.symbol else cfg["data"]["symbols"]
    fetcher = OHLCVFetcher(
        exchange_id=cfg["exchange"]["name"],
        raw_dir=cfg["data"]["raw_dir"],
    )
    for sym in symbols:
        if args.incremental:
            fetcher.fetch_incremental(sym, args.timeframe or cfg["data"]["timeframe"])
        else:
            fetcher.fetch(
                sym,
                timeframe=args.timeframe or cfg["data"]["timeframe"],
                years=args.years or cfg["data"]["history_years"],
            )


def cmd_features(args, cfg):
    from fetch_ohlcv import load_ohlcv
    from feature_engine import FeatureEngine

    symbols = [args.symbol] if args.symbol else cfg["data"]["symbols"]
    engine = FeatureEngine(config=cfg.get("features", {}))

    for sym in symbols:
        console.print(Rule(f"[cyan]Features: {sym}[/cyan]"))
        df = load_ohlcv(sym, args.timeframe or cfg["data"]["timeframe"], raw_dir=cfg["data"]["raw_dir"])
        feats = engine.transform(df)
        engine.save(feats, sym, args.timeframe or cfg["data"]["timeframe"])


def cmd_label(args, cfg):
    from fetch_ohlcv import load_ohlcv
    from feature_engine import FeatureEngine
    from labeller import Labeller

    symbols = [args.symbol] if args.symbol else cfg["data"]["symbols"]
    tf = args.timeframe or cfg["data"]["timeframe"]
    method = args.method or cfg["labels"]["method"]

    engine = FeatureEngine(config=cfg.get("features", {}))
    labeller = Labeller(config=cfg.get("labels", {}))

    for sym in symbols:
        console.print(Rule(f"[green]Labels: {sym}[/green]"))
        df = load_ohlcv(sym, tf, raw_dir=cfg["data"]["raw_dir"])
        feats = FeatureEngine.load(sym, tf)
        labels = labeller.generate(df, feats, method=method)
        labeller.save(labels, sym, tf, method)


def cmd_train(args, cfg):
    from feature_engine import FeatureEngine
    from labeller import Labeller
    from trainer import LGBMTrainer

    symbols = [args.symbol] if args.symbol else cfg["data"]["symbols"]
    tf = args.timeframe or cfg["data"]["timeframe"]
    method = cfg["labels"]["method"]

    trainer_cfg = cfg.get("model", {})
    if args.tune:
        trainer_cfg["tune_hyperparams"] = True

    for sym in symbols:
        console.print(Rule(f"[magenta]Training: {sym}[/magenta]"))
        feats = FeatureEngine.load(sym, tf)
        labels = Labeller.load(sym, tf, method)
        trainer = LGBMTrainer(config=trainer_cfg)
        results = trainer.train(feats, labels, sym, tf)
        logger.success(f"Model saved: {results['model_path']}")


def cmd_backtest(args, cfg):
    from fetch_ohlcv import load_ohlcv
    from feature_engine import FeatureEngine
    from trainer import LGBMTrainer
    from backtester import Backtester
    import pandas as pd

    symbols = [args.symbol] if args.symbol else cfg["data"]["symbols"]
    tf = args.timeframe or cfg["data"]["timeframe"]

    bt = Backtester(config=cfg.get("backtest", {}))

    for sym in symbols:
        console.print(Rule(f"[yellow]Backtest: {sym}[/yellow]"))
        df = load_ohlcv(sym, tf, raw_dir=cfg["data"]["raw_dir"])
        feats = FeatureEngine.load(sym, tf)
        trainer = LGBMTrainer(config=cfg.get("model", {}))
        trainer.load(sym, tf)

        # Run inference on all features
        signal_labels, probabilities = trainer.predict(feats)
        pred_series = pd.Series(signal_labels, index=feats.index)
        prob_df = pd.DataFrame(probabilities, index=feats.index, columns=["short", "hold", "long"])

        results = bt.run(df, pred_series, prob_df, symbol=sym)
        bt.save_results(results, sym)


def cmd_trade(args, cfg):
    from feature_engine import FeatureEngine
    from trainer import LGBMTrainer
    from live_trader import LiveTrader

    sym = args.symbol or cfg["data"]["symbols"][0]
    tf = args.timeframe or cfg["data"]["timeframe"]
    mode = args.mode or cfg["inference"]["mode"]

    console.print(Panel.fit(
        f"[bold]Starting {'PAPER' if mode == 'paper' else '⚠ LIVE'} trading[/bold]\n"
        f"Symbol: {sym} | Timeframe: {tf}",
        border_style="green" if mode == "paper" else "red",
    ))

    # Load model
    trainer = LGBMTrainer(config=cfg.get("model", {}))
    trainer.load(sym, tf)

    # Load feature engine
    engine = FeatureEngine(config=cfg.get("features", {}))

    # Inference config
    inf_cfg = {
    **cfg.get("inference", {}),
    "mode": mode,
    "log_dir": cfg["logging"]["log_dir"],
    "api_key": cfg["exchange"]["api_key"],
    "api_secret": cfg["exchange"]["api_secret"],
    "testnet": cfg["exchange"].get("testnet", True)
}


    trader = LiveTrader(sym, tf, engine, trainer, config=inf_cfg)
    trader.run()


def cmd_pipeline(args, cfg):
    """Run the full pipeline end-to-end."""
    console.print(Panel.fit(
        "[bold cyan]Running full pipeline[/bold cyan]\n"
        "fetch → features → label → train → backtest",
        title="🔄 Full Pipeline"
    ))
    cmd_fetch(args, cfg)
    cmd_features(args, cfg)
    cmd_label(args, cfg)
    cmd_train(args, cfg)
    cmd_backtest(args, cfg)
    console.print("\n[bold green]✓ Pipeline complete![/bold green]")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="QuantBot — Autonomous AI Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 main.py fetch --symbol BTC/USDT --years 3
  python3 main.py features --symbol BTC/USDT
  python3 main.py label --symbol BTC/USDT --method triple_barrier
  python3 main.py train --symbol BTC/USDT
  python3 main.py backtest --symbol BTC/USDT
  python3 main.py trade --symbol BTC/USDT --mode paper
  python3 main.py pipeline --symbol BTC/USDT
  python3 main.py pipeline --symbol BTC/USDT --timeframe 1m --years 1
        """
    )
    parser.add_argument("command", choices=["fetch", "features", "label", "train", "backtest", "trade", "pipeline"])
    parser.add_argument("--symbol", default=None, help="Trading pair, e.g. BTC/USDT")
    parser.add_argument("--timeframe", default=None, help="Candle timeframe, e.g. 1h")
    parser.add_argument("--years", type=float, default=None, help="Years of history to fetch")
    parser.add_argument("--method", default=None, help="Labelling method: triple_barrier, fixed_horizon, trend")
    parser.add_argument("--mode", default=None, help="Trading mode: paper | live")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--incremental", action="store_true", help="Incremental data fetch")
    parser.add_argument("--tune", action="store_true", help="Run Optuna HPO during training")

    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        logger.error(f"Config not found: {args.config}")
        sys.exit(1)

    cmd_map = {
        "fetch": cmd_fetch,
        "features": cmd_features,
        "label": cmd_label,
        "train": cmd_train,
        "backtest": cmd_backtest,
        "trade": cmd_trade,
        "pipeline": cmd_pipeline,
    }

    cmd_map[args.command](args, cfg)


if __name__ == "__main__":
    main()
