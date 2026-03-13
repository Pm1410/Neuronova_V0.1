"""
dashboard/app.py
----------------
Interactive Streamlit dashboard for QuantBot.

Tabs:
1. 📊 Data Explorer  — Visualize OHLCV + indicators
2. 🧠 Model         — Feature importance, CV metrics, confusion matrix
3. 📈 Backtest      — Equity curve, trade log, performance metrics
4. 🤖 Live Trading  — Real-time status, open positions, P&L
5. ⚙  Settings      — Config editor (YAML)

Run: streamlit run dashboard/app.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import streamlit as st
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="QuantBot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Space+Grotesk:wght@400;600;700&display=swap');

html, body, [data-testid="stApp"] {
    background: #0a0e1a;
    color: #e0e6f0;
    font-family: 'Space Grotesk', sans-serif;
}
.metric-card {
    background: linear-gradient(135deg, #111827 0%, #1a2233 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 4px 0;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: #00d4ff;
}
.metric-label {
    font-size: 0.75rem;
    color: #8899aa;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.positive { color: #00ff88 !important; }
.negative { color: #ff4466 !important; }
.neutral  { color: #ffaa00 !important; }
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    color: #8899aa;
}
.stTabs [aria-selected="true"] {
    color: #00d4ff !important;
    border-bottom: 2px solid #00d4ff !important;
}
h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif;
    color: #e0e6f0;
}
code, .stCode {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem;
}
[data-testid="stSidebar"] {
    background: #080c16;
    border-right: 1px solid #1e3a5f;
}
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🤖 QuantBot")
    st.markdown("*Autonomous AI Trading System*")
    st.divider()

    cfg_path = Path("config.yaml")
    cfg = {}
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text())

    symbols = cfg.get("data", {}).get("symbols", ["BTC/USDT"])
    timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]

    selected_symbol = st.selectbox("Symbol", symbols)
    selected_tf = st.selectbox("Timeframe", timeframes, index=timeframes.index(cfg.get("data", {}).get("timeframe", "1h")))

    st.divider()
    st.markdown("**Pipeline Status**")

    def check_file(label, path):
        exists = Path(path).exists()
        icon = "✅" if exists else "❌"
        st.markdown(f"{icon} {label}")

    raw_path = f"data/raw/{selected_symbol.replace('/', '_')}_{selected_tf}.parquet"
    feat_path = f"features/{selected_symbol.replace('/', '_')}_{selected_tf}_features.parquet"
    label_path = f"labels/{selected_symbol.replace('/', '_')}_{selected_tf}_triple_barrier_labels.parquet"
    model_path = f"models/saved/{selected_symbol.replace('/', '_')}_{selected_tf}_latest.lgb"
    trades_path = f"backtest/{selected_symbol.replace('/', '_')}_trades.csv"

    check_file("Data fetched", raw_path)
    check_file("Features built", feat_path)
    check_file("Labels generated", label_path)
    check_file("Model trained", model_path)
    check_file("Backtest run", trades_path)


# ─── Tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Explorer",
    "🧠 Model",
    "📈 Backtest",
    "🤖 Live Trading",
    "⚙ Settings",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Data Explorer
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("## 📊 Data Explorer")

    raw_p = Path(raw_path)
    if not raw_p.exists():
        st.warning(f"No data found. Run: `python3 main.py fetch --symbol {selected_symbol}`")
        st.stop()

    @st.cache_data
    def load_raw(path):
        return pd.read_parquet(path)

    df = load_raw(str(raw_p))

    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Candles", f"{len(df):,}")
    with col2:
        st.metric("Date Range", f"{df.index[0].date()} → {df.index[-1].date()}")
    with col3:
        ret = df["close"].iloc[-1] / df["close"].iloc[0] - 1
        st.metric("Total Return", f"{ret:+.1%}", delta=f"{ret:+.1%}")
    with col4:
        vol = df["close"].pct_change().std() * np.sqrt(365 * 24) * 100
        st.metric("Ann. Volatility", f"{vol:.1f}%")
    with col5:
        st.metric("Avg Daily Volume", f"${df['volume'].mean() * df['close'].mean():,.0f}")

    st.divider()

    # Date range filter
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("From", value=df.index[-500].date())
    with col2:
        end_date = st.date_input("To", value=df.index[-1].date())

    mask = (df.index.date >= start_date) & (df.index.date <= end_date)
    df_view = df[mask]

    # OHLCV Chart
    indicators = st.multiselect(
        "Overlay Indicators",
        ["EMA 20", "EMA 50", "EMA 200", "Bollinger Bands", "VWAP"],
        default=["EMA 20", "EMA 50"],
    )

    fig = sp.make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.02,
        subplot_titles=["Price", "Volume", "RSI 14"],
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df_view.index,
        open=df_view["open"],
        high=df_view["high"],
        low=df_view["low"],
        close=df_view["close"],
        name="OHLCV",
        increasing_line_color="#00ff88",
        decreasing_line_color="#ff4466",
    ), row=1, col=1)

    # Indicator overlays
    for ind in indicators:
        if ind == "EMA 20":
            ema = df_view["close"].ewm(span=20).mean()
            fig.add_trace(go.Scatter(x=df_view.index, y=ema, name="EMA20", line=dict(color="#00d4ff", width=1)), row=1, col=1)
        elif ind == "EMA 50":
            ema = df_view["close"].ewm(span=50).mean()
            fig.add_trace(go.Scatter(x=df_view.index, y=ema, name="EMA50", line=dict(color="#ffaa00", width=1)), row=1, col=1)
        elif ind == "EMA 200":
            ema = df_view["close"].ewm(span=200).mean()
            fig.add_trace(go.Scatter(x=df_view.index, y=ema, name="EMA200", line=dict(color="#ff6688", width=1)), row=1, col=1)
        elif ind == "Bollinger Bands":
            ma = df_view["close"].rolling(20).mean()
            std = df_view["close"].rolling(20).std()
            fig.add_trace(go.Scatter(x=df_view.index, y=ma + 2*std, name="BB Upper", line=dict(color="#8844ff", dash="dot", width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_view.index, y=ma - 2*std, name="BB Lower", line=dict(color="#8844ff", dash="dot", width=1), fill="tonexty", fillcolor="rgba(136,68,255,0.05)"), row=1, col=1)

    # Volume bars
    colors = ["#00ff88" if c >= o else "#ff4466" for c, o in zip(df_view["close"], df_view["open"])]
    fig.add_trace(go.Bar(x=df_view.index, y=df_view["volume"], name="Volume", marker_color=colors, opacity=0.7), row=2, col=1)

    # RSI
    delta = df_view["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / loss))
    fig.add_trace(go.Scatter(x=df_view.index, y=rsi, name="RSI 14", line=dict(color="#ffaa00", width=1.5)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#ff4466", opacity=0.5, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#00ff88", opacity=0.5, row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0a0e1a",
        xaxis_rangeslider_visible=False,
        height=700,
        showlegend=True,
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Raw data preview
    with st.expander("View raw data"):
        st.dataframe(df_view.tail(100).style.format("{:.4f}"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Model
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("## 🧠 Model Analysis")

    safe = selected_symbol.replace("/", "_")
    meta_path = Path(f"models/saved/{safe}_{selected_tf}_latest_meta.json")

    if not meta_path.exists():
        st.warning(f"No trained model found. Run: `python3 main.py train --symbol {selected_symbol}`")
    else:
        meta = json.loads(meta_path.read_text())

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Accuracy", f"{meta['avg_accuracy']:.3f}")
        col2.metric("Avg F1 Macro", f"{meta['avg_f1']:.3f}")
        col3.metric("Features", str(len(meta["feature_names"])))
        col4.metric("Trained At", meta["trained_at"])

        st.divider()

        col1, col2 = st.columns([1, 1])

        with col1:
            # CV metrics chart
            cv = pd.DataFrame(meta["cv_metrics"])
            fig_cv = go.Figure()
            fig_cv.add_trace(go.Bar(x=cv["fold"], y=cv["accuracy"], name="Accuracy", marker_color="#00d4ff"))
            fig_cv.add_trace(go.Bar(x=cv["fold"], y=cv["f1_macro"], name="F1 Macro", marker_color="#00ff88"))
            fig_cv.update_layout(
                title="Walk-Forward CV Performance",
                template="plotly_dark", paper_bgcolor="#111827",
                plot_bgcolor="#111827", barmode="group",
                xaxis_title="Fold", yaxis_title="Score",
                yaxis=dict(range=[0, 1]),
                height=350,
            )
            st.plotly_chart(fig_cv, use_container_width=True)

        with col2:
            # Feature importance (from meta if stored, else placeholder)
            feat_names = meta["feature_names"][:20]
            # Show feature groups as pie
            groups = {
                "Returns": sum(1 for f in meta["feature_names"] if f.startswith("ret_")),
                "Trend": sum(1 for f in meta["feature_names"] if any(x in f for x in ["ema", "macd", "adx"])),
                "Momentum": sum(1 for f in meta["feature_names"] if any(x in f for x in ["rsi", "stoch", "roc"])),
                "Volatility": sum(1 for f in meta["feature_names"] if any(x in f for x in ["atr", "bb_", "kc_"])),
                "Volume": sum(1 for f in meta["feature_names"] if any(x in f for x in ["obv", "mfi", "vwap"])),
                "Micro": sum(1 for f in meta["feature_names"] if any(x in f for x in ["bar_", "body", "wick"])),
                "Lags": sum(1 for f in meta["feature_names"] if "lag" in f),
                "Rolling": sum(1 for f in meta["feature_names"] if "roll_" in f),
                "Time": sum(1 for f in meta["feature_names"] if "sin" in f or "cos" in f),
            }
            fig_pie = go.Figure(go.Pie(
                labels=list(groups.keys()),
                values=list(groups.values()),
                hole=0.4,
                marker_colors=["#00d4ff","#00ff88","#ffaa00","#ff6688","#8844ff","#44ffcc","#ff8844","#88ff44","#4488ff"],
            ))
            fig_pie.update_layout(
                title="Feature Groups",
                template="plotly_dark", paper_bgcolor="#111827",
                height=350,
                showlegend=True,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # Model params
        with st.expander("Model Hyperparameters"):
            st.json(meta["params"])

        # Feature list
        with st.expander(f"All {len(meta['feature_names'])} Features"):
            cols = st.columns(3)
            for i, feat in enumerate(meta["feature_names"]):
                cols[i % 3].code(feat)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Backtest
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("## 📈 Backtest Results")

    trades_p = Path(trades_path)
    equity_p = Path(f"backtest/{safe}_equity.parquet")
    metrics_p = Path(f"backtest/{safe}_metrics.json")

    if not trades_p.exists():
        st.warning(f"No backtest results. Run: `python3 main.py backtest --symbol {selected_symbol}`")
    else:
        trades = pd.read_csv(trades_p, parse_dates=["entry_ts", "exit_ts"])
        equity = pd.read_parquet(equity_p)["equity"]
        metrics = json.loads(metrics_p.read_text())

        # Key metrics
        cols = st.columns(7)
        ret = metrics["total_return_pct"]
        cols[0].metric("Total Return", f"{ret:+.2f}%", delta=f"{ret:+.2f}%")
        cols[1].metric("Sharpe", f"{metrics['sharpe_ratio']:.3f}")
        cols[2].metric("Sortino", f"{metrics['sortino_ratio']:.3f}")
        cols[3].metric("Max DD", f"{metrics['max_drawdown_pct']:.2f}%")
        cols[4].metric("Win Rate", f"{metrics['win_rate_pct']:.1f}%")
        cols[5].metric("Profit Factor", f"{metrics['profit_factor']:.3f}")
        cols[6].metric("Trades", str(metrics["n_trades"]))

        st.divider()

        # Equity curve
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=equity.index, y=equity.values,
            fill="tozeroy",
            fillcolor="rgba(0,212,255,0.08)",
            line=dict(color="#00d4ff", width=2),
            name="Portfolio",
        ))
        # Mark trade entries
        if len(trades) > 0:
            # ── Timezone normalisation ──────────────────────────────────────
            # equity.index may be UTC-aware (datetime64[ns, UTC]) while
            # trade timestamps loaded from CSV are tz-naive (datetime64[ns]).
            # Align both to UTC-aware so reindex() can compare them.
            eq_index = equity.index
            if eq_index.tz is None:
                eq_index = eq_index.tz_localize("UTC")
                equity = equity.copy()
                equity.index = eq_index

            def to_utc(ts_series):
                """Ensure a timestamp Series is UTC-aware."""
                s = pd.to_datetime(ts_series, utc=False)
                if s.dt.tz is None:
                    s = s.dt.tz_localize("UTC")
                else:
                    s = s.dt.tz_convert("UTC")
                return s

            long_t  = trades[trades["signal"] == 1].copy()
            short_t = trades[trades["signal"] == -1].copy()

            long_ts  = to_utc(long_t["entry_ts"])
            short_ts = to_utc(short_t["entry_ts"])

            fig_eq.add_trace(go.Scatter(
                x=long_ts,
                y=equity.reindex(long_ts, method="nearest").values,
                mode="markers", marker=dict(color="#00ff88", size=8, symbol="triangle-up"),
                name="Long Entry",
            ))
            fig_eq.add_trace(go.Scatter(
                x=short_ts,
                y=equity.reindex(short_ts, method="nearest").values,
                mode="markers", marker=dict(color="#ff4466", size=8, symbol="triangle-down"),
                name="Short Entry",
            ))

        fig_eq.update_layout(
            title="Equity Curve",
            template="plotly_dark", paper_bgcolor="#111827", plot_bgcolor="#111827",
            height=400, xaxis_title="Date", yaxis_title="Portfolio Value (USD)",
            yaxis_tickprefix="$",
        )
        st.plotly_chart(fig_eq, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            # Drawdown chart
            peak = equity.cummax()
            dd = (equity - peak) / peak * 100
            fig_dd = go.Figure(go.Scatter(
                x=dd.index, y=dd.values,
                fill="tozeroy", fillcolor="rgba(255,68,102,0.15)",
                line=dict(color="#ff4466", width=1.5),
                name="Drawdown",
            ))
            fig_dd.update_layout(
                title="Drawdown (%)",
                template="plotly_dark", paper_bgcolor="#111827", plot_bgcolor="#111827",
                height=280, yaxis_ticksuffix="%",
            )
            st.plotly_chart(fig_dd, use_container_width=True)

        with col2:
            # PnL distribution
            if len(trades) > 0:
                fig_pnl = go.Figure(go.Histogram(
                    x=trades["pnl"],
                    nbinsx=30,
                    marker_color=["#00ff88" if p > 0 else "#ff4466" for p in trades["pnl"]],
                    name="Trade PnL",
                ))
                fig_pnl.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
                fig_pnl.update_layout(
                    title="Trade PnL Distribution",
                    template="plotly_dark", paper_bgcolor="#111827", plot_bgcolor="#111827",
                    height=280, xaxis_title="PnL (USD)", yaxis_title="Count",
                )
                st.plotly_chart(fig_pnl, use_container_width=True)

        # Trade log table
        st.markdown("### Trade Log")
        if len(trades) > 0:
            display_trades = trades[["entry_ts", "exit_ts", "signal", "entry_price", "exit_price", "pnl", "confidence"]].copy()
            display_trades["signal"] = display_trades["signal"].map({1: "🟢 LONG", -1: "🔴 SHORT"})
            display_trades["pnl"] = display_trades["pnl"].round(2)
            st.dataframe(display_trades, use_container_width=True, height=300)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: Live Trading
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("## 🤖 Live Trading Monitor")

    import time as _time

    # Auto-refresh using only built-in Streamlit — no extra packages needed
    REFRESH_INTERVAL = 30  # seconds
    if "last_refresh" not in st.session_state:
        st.session_state["last_refresh"] = _time.time()

    elapsed   = _time.time() - st.session_state["last_refresh"]
    remaining = max(0, int(REFRESH_INTERVAL - elapsed))

    _rc1, _rc2 = st.columns([5, 1])
    with _rc1:
        st.caption(f"⏱ Auto-refreshes every {REFRESH_INTERVAL}s — next in **{remaining}s**")
    with _rc2:
        if st.button("🔄 Refresh", key="manual_refresh"):
            st.session_state["last_refresh"] = _time.time()
            st.rerun()

    if elapsed >= REFRESH_INTERVAL:
        st.session_state["last_refresh"] = _time.time()
        st.rerun()

    trade_log_p = Path("logs/trades.csv")

    col1, col2, col3 = st.columns(3)
    with col1:
        mode_badge = "🟡 PAPER" if cfg.get("inference", {}).get("mode") == "paper" else "🔴 LIVE"
        st.metric("Trading Mode", mode_badge)
    with col2:
        st.metric("Symbol", selected_symbol)
    with col3:
        st.metric("Auto-refresh", f"{remaining}s")

    st.divider()

    if trade_log_p.exists():
        live_trades = pd.read_csv(trade_log_p)
        session_pnl = live_trades["pnl"].sum() if "pnl" in live_trades.columns else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Session Trades", str(len(live_trades)))
        col2.metric("Session PnL", f"${session_pnl:+.2f}", delta=f"${session_pnl:+.2f}")
        col3.metric("Win Rate", f"{(live_trades['pnl'] > 0).mean():.1%}" if len(live_trades) > 0 else "—")

        if len(live_trades) > 0:
            st.markdown("### Recent Trades")
            st.dataframe(live_trades.tail(20), use_container_width=True)

            # Running PnL
            live_trades_sorted = live_trades.sort_index()
            live_trades_sorted["cumulative_pnl"] = live_trades_sorted["pnl"].cumsum()
            fig_live = px.line(
                live_trades_sorted, y="cumulative_pnl",
                title="Session Cumulative PnL",
                template="plotly_dark",
                color_discrete_sequence=["#00d4ff"],
            )
            fig_live.update_layout(paper_bgcolor="#111827", plot_bgcolor="#111827", height=300)
            st.plotly_chart(fig_live, use_container_width=True)
    else:
        st.info("No live trade log found. Start trading with:\n```\npython main.py trade --symbol BTC/USDT --mode paper\n```")

    st.divider()
    st.markdown("### Start Trading")
    col1, col2 = st.columns(2)
    with col1:
        trade_mode = st.radio("Mode", ["paper", "live"], horizontal=True)
        trade_sym = st.selectbox("Symbol", symbols, key="trade_sym")
    with col2:
        st.code(f"python3 main.py trade --symbol {trade_sym} --mode {trade_mode}", language="bash")
        st.warning("⚠ Live mode places real orders. Use testnet first." if trade_mode == "live" else "")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: Settings
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown("## ⚙ Configuration")

    if cfg_path.exists():
        raw_cfg = cfg_path.read_text()
        edited_cfg = st.text_area("config.yaml", value=raw_cfg, height=600, key="cfg_editor")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Config", type="primary"):
                try:
                    yaml.safe_load(edited_cfg)  # Validate YAML
                    cfg_path.write_text(edited_cfg)
                    st.success("Config saved!")
                except yaml.YAMLError as e:
                    st.error(f"Invalid YAML: {e}")
        with col2:
            if st.button("🔄 Reset to Default"):
                st.info("Edit config.yaml manually to reset.")
    else:
        st.error("config.yaml not found")

    st.divider()
    st.markdown("### Quick Commands")
    st.code(f"""
# Fetch data
python3 main.py fetch --symbol {selected_symbol} --timeframe {selected_tf} --years 3

# Build features
python3 main.py features --symbol {selected_symbol}

# Generate labels
python3 main.py label --symbol {selected_symbol} --method triple_barrier

# Train model (add --tune for Optuna HPO)
python3 main.py train --symbol {selected_symbol}

# Backtest
python3 main.py backtest --symbol {selected_symbol}

# Paper trade
python3 main.py trade --symbol {selected_symbol} --mode paper

# Full pipeline
python3 main.py pipeline --symbol {selected_symbol}

# Launch this dashboard
streamlit run app.py

# See which models are trained / missing
python3 multi_train.py status

# Full pipeline for all 12 models (BTC+ETH+SOL × 5m+15m+1h+4h)
python3 multi_train.py pipeline

# Individual steps
python3 multi_train.py fetch
python3 multi_train.py features
python3 multi_train.py label
python3 multi_train.py train
python3 multi_train.py train --tune        # with Optuna HPO

# Backtest all and compare in one table
python3 multi_train.py backtest

# Launch ALL 12 bots in parallel threads
python3 multi_train.py trade
python3 multi_train.py trade --mode live   # ⚠ real money

# Subset operations
python3 multi_train.py train --symbols BTC/USDT --timeframes 1h 4h
python3 multi_train.py trade --symbols BTC/USDT ETH/USDT --timeframes 1h

# Preview without running
python3 multi_train.py pipeline --dry-run
""", language="bash")