# MarsFinance

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**[中文说明 →](README.zh-CN.md)**

An open-source quantitative research framework for the Chinese A-share market:
data loading via QMT/xtquant, a volume-price strategy family, event-loop
backtesting with realistic trading frictions, and an XGBoost return-prediction
pipeline.

Built 2024–2026.03 as the research infrastructure behind my personal systematic
investing practice, and kept public as a research record. Its successor — a
fundamentals-driven investment decision system — is developed privately.

## Highlights

- **Structural look-ahead protection.** On day *i* the strategy function
  receives only `data.iloc[:i]` (data through day *i−1*), while execution uses
  day-*i* prices. Signals are separated from execution by construction, not by
  after-the-fact checks. (`backtest/stock_backtest.py`)
- **Realistic frictions.** Commission and slippage are modeled on both sides:
  buys fill above market at `price × (1 + slippage)` plus commission, sells fill
  below market at `price × (1 − slippage)` minus commission.
- **Volume-price strategy family.** A base volume/price signal hardened by three
  filters: volatility-adaptive thresholds, a 3σ abnormal-volume filter, and
  multi-timeframe momentum confirmation — conflicting signals mean no trade.
  (`strategies/volume_price_strategy.py`)
- **XGBoost return prediction.** 30-day forward-return regression over four
  feature families (momentum, volume/money-flow, technical, volatility &
  sentiment); time-ordered train/test split, scaler fit on the training set
  only, TimeSeriesSplit cross-validation; direction accuracy reported alongside
  MSE/MAE/R². (`machine_learning/xgboost_prediction_framework.py`)
- **Portfolio layer.** Equal-capital allocation across the CSI 300 universe,
  aggregated portfolio metrics (return, Sharpe, drawdown, win rate, share of
  profitable names), normalized equity-curve comparison against a benchmark.
- **Research reproductions** (notebooks under `app/`): Piotroski F-Score
  applied to A-shares, CSCV probability of backtest overfitting, and
  risk-budgeting / ML-based asset allocation — see
  [`src/research_papers/README.md`](src/research_papers/README.md) for the
  papers.

## Layout

```
MarsFinance/
├── backtest/             # single-stock & portfolio engines, multi-strategy comparison
├── strategies/           # volume-price family + classic baselines (MA cross, RSI, Bollinger)
├── machine_learning/     # XGBoost return-prediction pipeline
├── data_loader/          # QMT/xtdata market-data loaders
├── app/                  # research notebooks + evolved engine (daily position snapshots)
├── examples/             # runnable usage example
└── src/research_papers/  # reproduced papers (citations; PDFs not redistributed)
```

## Quick start

```bash
pip install -r requirements.txt
```

```python
from data_loader.data_loader import DataLoader
from backtest.portfolio_backtest import PortfolioBacktest
from strategies.volume_price_strategy import enhanced_volume_price_strategy

loader = DataLoader()
stock_data = loader.load_hs300_data("20230101", "20231231")

backtest = PortfolioBacktest(initial_capital=1_000_000)
backtest.run_stock_universe_backtest(
    stock_data_dict=stock_data,
    strategy_function=enhanced_volume_price_strategy,
    capital_per_stock=20_000,
)
backtest.print_detailed_report()
```

See [`examples/basic_usage.py`](examples/basic_usage.py) for the runnable
version.

## Data sources

- **A-shares:** `xtquant`/`xtdata`, which ships with the QMT / MiniQMT trading
  terminal (not on PyPI — copy it from your QMT installation or add it to
  `PYTHONPATH`). Imports are lazy, so the rest of the package works without it.
- **US-market experiments:** `yfinance`.

## Project status, honestly

- **Archived** (developed 2024.01–2026.03). Maintained as a research record,
  not as a production system.
- Notebooks under `app/` are research artifacts in varying stages of maturity;
  all outputs are stripped.
- Strategy parameters are hand-tuned defaults, not systematically calibrated.
  That lesson is carried into the successor system as a hard constraint: no
  parameter enters production without out-of-sample validation.

## Disclaimer

For research and education only. Nothing in this repository is investment
advice.

## License

[MIT](LICENSE) — © Long Huang (黄隆)
