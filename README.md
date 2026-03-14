# 🚀 Neuronova V0.1 — Crypto ML Trading Prototype

<p align="center">
Machine learning research prototype for cryptocurrency trading.<br>
Collect data → engineer features → train ML model → evaluate through backtesting and paper trading.
</p>

<p align="center">
<img src="https://img.shields.io/badge/Python-3.10+-blue">
<img src="https://img.shields.io/badge/ML-LightGBM-green">
<img src="https://img.shields.io/badge/Data-CCXT-orange">
<img src="https://img.shields.io/badge/Status-Research-yellow">
<img src="https://img.shields.io/badge/License-MIT-purple">
</p>

---

# 🧠 Overview

**Neuronova V0.1** is a lightweight research framework designed to explore whether **machine learning models can extract trading signals from cryptocurrency market data**.

The project provides a simple end-to-end pipeline that:

- collects historical market data  
- generates technical features  
- trains a machine learning model  
- evaluates performance through backtesting  
- simulates trading via paper trading

The goal is **research and experimentation**, not production trading.

---

# ⚙️ System Pipeline

The workflow follows a straightforward ML trading pipeline:

```
Exchange Data (CCXT)
        │
        ▼
Feature Engineering
        │
        ▼
Label Generation
        │
        ▼
Dataset Storage
        │
        ▼
Model Training (LightGBM)
        │
        ▼
Backtesting Engine
        │
        ▼
Paper Trading Simulation
```

This pipeline allows quick iteration when testing new features, models, or strategies.

---

# ✨ Key Features

### 📊 Market Data Collection
Fetch historical **OHLCV cryptocurrency data** directly from exchanges using CCXT.

### 🧮 Feature Engineering
Generate technical indicators and engineered features from raw price data.

### 🏷 Label Generation
Create supervised learning labels for classification-based trading signals.

### 🤖 Machine Learning Training
Train a **LightGBM model** to detect potential trading opportunities.

### 📉 Strategy Backtesting
Evaluate trading strategies on historical data to estimate performance.

### 💰 Paper Trading Simulation
Simulate trades using model predictions without risking real capital.

---

# 🏗️ Project Structure

```
Neuronova_V0.1/
│
├── data/                 # Raw market datasets
├── features/             # Feature engineered datasets
├── labels/               # Labelled datasets
├── models/               # Trained LightGBM models
├── backtest/             # Backtesting engine
├── logs/                 # Paper trading simulation logs
│
├── fetch_ohlcv.py        # Collect OHLCV data using CCXT
├── feature_engine.py     # Feature engineering pipeline
├── labeller.py           # Label generation
├── trainer.py            # Train ML model
├── backtester.py         # Strategy backtesting
├── live_trader.py        # Paper trading simulator
├── multi_train.py        # Run multiple training experiments
│
├── app.py                # Streamlit dashboard interface
├── main.py               # Pipeline automation script
├── config.yaml           # Project configuration
│
├── requirements.txt      # Python dependencies
└── README.md
```

---

# 🧰 Technology Stack

| Component | Technology |
|----------|------------|
| Data Collection | CCXT |
| Data Processing | Pandas, NumPy |
| Machine Learning | LightGBM |
| Model Evaluation | Scikit-learn |
| Dashboard | Streamlit |
| Programming Language | Python |

---

# ⚡ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/neuronova_v0.1.git
cd Neuronova_V0.1
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Usage

## 1️⃣ Fetch Market Data

Download historical cryptocurrency data.

```bash
python fetch_ohlcv.py
```

---

## 2️⃣ Generate Features

Create technical indicators and engineered features.

```bash
python feature_engine.py
```

---

## 3️⃣ Generate Labels

Prepare training labels for the ML model.

```bash
python labeller.py
```

---

## 4️⃣ Train Machine Learning Model

Train the LightGBM classifier.

```bash
python trainer.py
```

---

## 5️⃣ Run Backtesting

Evaluate the trading strategy using historical data.

```bash
python backtester.py
```

---

## 6️⃣ Simulate Paper Trading

Run simulated trading without real money.

```bash
python live_trader.py
```

---

# 📊 Backtesting Metrics

The backtesting module evaluates strategies using common trading metrics:

| Metric | Description |
|------|-------------|
| Total Return | Overall strategy profitability |
| Sharpe Ratio | Risk-adjusted returns |
| Win Rate | Percentage of profitable trades |
| Maximum Drawdown | Largest portfolio decline |
| Number of Trades | Total executed trades |

---

# ⚠️ Limitations

This project is intentionally simplified and has several limitations:

- Limited feature engineering
- Simplified backtesting engine
- No transaction cost modeling
- No slippage modeling
- No portfolio management
- No risk management module
- No real-time execution validation

The framework is intended for **research experiments only**.

---

# 🔬 Future Improvements

Planned enhancements for future versions:

- Advanced feature engineering
- Walk-forward validation
- Robust backtesting engine
- Risk management module
- Model ensembling
- Reinforcement learning agents
- Real-time trading integration
- Portfolio optimization

---

# ⚠️ Disclaimer

This repository is intended **for educational and research purposes only**.

It does **not provide financial advice** and should **not be used for real trading decisions**.  
Cryptocurrency trading carries significant financial risk.

---

# 🤝 Contributing

Contributions are welcome.

Steps:

1. Fork the repository
2. Create a new feature branch

```bash
git checkout -b feature/new-feature
```

3. Commit your changes

```bash
git commit -m "Add new feature"
```

4. Push the branch

```bash
git push origin feature/new-feature
```

5. Open a Pull Request

---

# 📜 License

MIT License

You are free to use, modify, and distribute this project.

---

# ⭐ Support

If you find this project useful, consider giving it a **star ⭐ on GitHub**.
