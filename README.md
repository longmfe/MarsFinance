# MarsFinance

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**[中文说明 →](README.zh-CN.md)**

An open-source quantitative research framework for the Chinese A-share market:
akshare-backed market data (stocks, ETFs, index futures), a volume-price
strategy family, ETF trend strategies, an index-futures CTA engine,
event-loop backtesting with realistic trading frictions, and an XGBoost
return-prediction pipeline.

Built 2024.01–2026.03 as the research infrastructure behind my personal
systematic investing practice, extended 2026.08 with ETF trend strategies and
the futures CTA engine, and kept public as a research record. Its successor —
a fundamentals-driven investment decision system — is developed privately.

## Highlights

- **akshare data layer.** Sina/Tencent-sourced daily bars for stocks, ETFs
  and index futures (`IF0` main-continuous, roll-adjusted), with a parquet
  disk cache (atomic writes, offline replays) and proxy handling for direct
  connections. Endpoint availability is documented honestly in
  `research/datafeed/akshare_source.py` — the Eastmoney `push2` endpoints do
  not work on this machine. (`research/datafeed/`)
- **Structural look-ahead protection.** On day *i* the strategy function
  receives only `data.iloc[:i]` (data through day *i−1*), while execution uses
  day-*i* prices. Signals are separated from execution by construction, not by
  after-the-fact checks. (`backtest/stock_backtest.py`, `backtest/futures_backtest.py`)
- **Realistic frictions.** Commission and slippage are modeled on both sides:
  buys fill above market at `price × (1 + slippage)` plus commission, sells fill
  below market at `price × (1 − slippage)` minus commission.
- **Volume-price strategy family.** A base volume/price signal hardened by three
  filters: volatility-adaptive thresholds, a 3σ abnormal-volume filter, and
  multi-timeframe momentum confirmation — conflicting signals mean no trade.
  (`strategies/volume_price_strategy.py`)
- **ETF trend strategies.** Trend-following for broad-index ETFs (e.g. 510310):
  dual-MA cross, regime-filtered MA cross, and Donchian breakout with an
  ATR trailing stop. Includes a runnable backtest with buy-and-hold benchmarks
  and a parameter-sensitivity sweep. (`strategies/etf_trend_strategy.py`,
  `examples/hs300_etf_backtest.py`)
- **Index-futures CTA engine.** Long/short, integer contract sizing,
  volatility-targeted positions (ATR-scaled, capped by margin) and forced
  liquidation on margin shortfall — the risk model that keeps leveraged
  backtests honest. Symmetric long/short trend strategies included.
  (`backtest/futures_backtest.py`, `strategies/futures_cta.py`,
  `examples/hs300_futures_cta.py`)
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
├── backtest/             # single-stock, portfolio & index-futures engines
├── strategies/           # volume-price family, ETF trend, futures CTA, classic baselines
├── machine_learning/     # XGBoost return-prediction pipeline (yfinance-based US data)
├── data_loader/          # QMT/xtquant loaders (optional; lazy imports)
├── research/             # akshare datafeed, FFScore, cross-sectional backtests, metrics
├── app/                  # research notebooks + evolved engines (some need QMT)
├── examples/             # runnable examples (akshare-based by default)
├── tests/                # offline test suite (pytest, no network by default)
└── src/research_papers/  # reproduced papers (citations; PDFs not redistributed)
```

## Quick start

```bash
pip install -r requirements.txt
```

The default data path is akshare (no QMT terminal required). First run fetches
data from Sina (~seconds per series) and caches it under `data/akshare_cache`;
every later run is offline:

```bash
python examples/hs300_etf_backtest.py       # CSI 300 ETF (510310) trend strategies
python examples/hs300_futures_cta.py        # IF0 index-futures CTA (long/short)
```

Use `--force` to refetch data and `--plot` to save equity curves.

The QMT/xtquant path still works if you have a QMT/MiniQMT terminal:

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

See [`examples/basic_usage.py`](examples/basic_usage.py).

## Examples

| Example | Asset | Needs |
|---|---|---|
| `examples/hs300_etf_backtest.py` | CSI 300 ETF (510310) | akshare |
| `examples/hs300_futures_cta.py` | HS300 index futures (IF0) | akshare |
| `examples/yjyg_event_example.py` | Earnings-forecast event strategy | akshare + QMT (prices) |
| `examples/basic_usage.py` | CSI 300 stock universe | QMT/MiniQMT |

## Data sources

- **akshare (default):** daily bars for A-shares, ETFs and index futures from
  Sina and Tencent hosts, plus Eastmoney fundamentals. Verified on this
  machine: Sina and Tencent endpoints work; Eastmoney `push2` endpoints are
  blocked — see the availability table in
  `research/datafeed/akshare_source.py`. All fetches go through the parquet
  cache, so research runs are reproducible offline.
- **QMT/xtquant (optional):** `xtquant`/`xtdata` ships with the QMT/MiniQMT
  trading terminal (not on PyPI — copy it from your QMT installation or add it
  to `PYTHONPATH`). Imports are lazy, so the rest of the package works without it.
- **US-market experiments:** `yfinance` (declared in `requirements.txt`, used by
  the XGBoost pipeline).

Dependencies split: `requirements.txt` covers the core stack (engines,
strategies, akshare data layer, examples); `requirements-research.txt` adds
the heavier extras for the paper reproductions under `research/`.

## Tests

```bash
pytest            # offline by default; -m network enables integration tests
```

## Project status, honestly

- Maintained as a research record, not as a production system.
- Notebooks under `app/` are research artifacts in varying stages of maturity;
  all outputs are stripped.
- Strategy parameters are hand-tuned defaults, not systematically calibrated —
  the example scripts include parameter-sensitivity sweeps to make that
  visible rather than hiding it. That lesson is carried into the successor
  system as a hard constraint: no parameter enters production without
  out-of-sample validation.

## Disclaimer

For research and education only. Nothing in this repository is investment
advice.

## License

[MIT](LICENSE) — © Long Huang (黄隆)
