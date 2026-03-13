# Crypto ML Trading Prototype

A simple research prototype for cryptocurrency trading using machine
learning.\
The project collects market data, generates features, trains a model,
and evaluates the strategy through backtesting and paper trading.

This repository is **not a production trading system**. It is an
experimental prototype built for research and testing purposes.

------------------------------------------------------------------------

## Overview

The pipeline is intentionally simple:

1.  Fetch historical crypto market data
2.  Generate features from price data
3.  Create labels for supervised learning
4.  Train a LightGBM model
5.  Run backtesting on historical data
6.  Simulate trades using paper trading

The goal of this project is to test whether basic ML models can learn
useful trading signals from engineered features.

------------------------------------------------------------------------

## Workflow

Exchange Data (CCXT) │ ▼ Feature Engineering │ ▼ Label Generation │ ▼
Dataset Storage │ ▼ LightGBM Training │ ▼ Backtesting │ ▼ Paper Trading
Simulation

------------------------------------------------------------------------

## Project Structure

Neuronova_V0.1/

data/ \# Saved datasets\
features/ \# Featured datasets\
labels/ \# Labelled datasets\
models/ \# Trained LightGBM models\
backtest/ \# Backtesting engine\
logs/ \# Paper trading simulation

fetch_ohlcv.py \# Collect data using CCXT\
feature_engine.py \# Feature engineering\
labeller.py \# Label generation\
trainer.py \# Train LightGBM model\
backtester.py \# Strategy backtesting\
live_trader.py \# Paper trading simulation
app.py \# Shows interactive Streamlit Webpage
config.yaml \# Contain Configurations
main.py \# To automate all steps in one go
multi_train.py \# single script to run pipeline

requirements.txt
README.md

------------------------------------------------------------------------

## Installation

Clone the repository:

git clone https://github.com/yourusername/neuronova_v0.1.git cd
Neuronova_V0.1

Install dependencies:

pip install -r requirements.txt

------------------------------------------------------------------------

## Dependencies

Main libraries used in the project:

-   ccxt
-   pandas
-   numpy
-   lightgbm
-   scikit-learn

------------------------------------------------------------------------

## Usage

### 1. Fetch Data

Fetch historical OHLCV data from exchanges.

python fetch_ohlcv.py

### 2. Generate Features

Create technical indicators and engineered features.

python feature_engine.py

### 3. Generate Labels

Label the dataset for supervised learning.

python labeller.py

### 4. Train Model

Train the LightGBM classifier on the dataset.

python trainer.py

### 5. Run Backtest

Evaluate strategy performance on historical data.

python backtester.py

### 6. Paper Trading

Simulate trades using the trained model.

python live_trader.py

Note: Paper trading currently simulates trades only. Live trading with
real funds has **not been tested**.

------------------------------------------------------------------------

## Backtesting

The project includes a simple backtesting module that:

-   Loads trained models
-   Generates predictions
-   Simulates trades
-   Computes basic performance metrics

Example metrics include:

-   Total return
-   Sharpe ratio
-   Win rate
-   Maximum drawdown
-   Number of trades

------------------------------------------------------------------------

## Limitations

This repository is a **basic prototype** and has several limitations:

-   Limited feature set
-   Simplified backtesting engine
-   No slippage modeling
-   No transaction cost modeling
-   No risk management module
-   No portfolio optimization
-   No live trading validation

The system should **not be used with real funds**.

------------------------------------------------------------------------

## Disclaimer

This project is for **educational and research purposes only**.

It does not provide financial advice and should not be used to make real
trading decisions. Cryptocurrency trading carries significant risk.

------------------------------------------------------------------------

## Future Work

Possible improvements include:

-   Better feature engineering
-   Walk-forward validation
-   Robust backtesting engine
-   Risk management system
-   Model ensembling
-   Reinforcement learning agents
-   Live trading integration

------------------------------------------------------------------------

Thank You....
