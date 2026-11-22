# Sparse Index Tracking (S&P 500)

Evaluating sparse index tracking methods—Constrained Least Squares (CLS), Ridge Regression, PCA + Linear/Ridge Regression, and LASSO—on their ability to replicate the SPDR S&P 500 ETF (`SPY`) using a small subset of $k \ll n$ S&P 500 constituents.

See [Team23_Milestone_Report.pdf](Team23_Milestone_Report.pdf) for the project background and modeling plan.

---

## Repository Structure

```text
SparseIndexTracking/
├── collect_data.py              # Script to build the 2021–2025 S&P 500 dataset
├── Team23_Milestone_Report.pdf  # Project milestone report
└── data/                        # Generated dataset (created by collect_data.py)
    ├── adj_close.parquet        # Daily Adjusted Close prices (SPY + 593 tickers)
    ├── constituents_mask.parquet# Daily boolean S&P 500 membership mask
    ├── rebalance_events.csv     # 2021–2025 S&P 500 additions and removals
    └── ticker_coverage.csv      # Per-ticker price and membership coverage stats
```

---

## Environment Setup & Data Collection

This project uses `uv` with the local `.venv` environment.

To regenerate the dataset from scratch:

```bash
uv run python collect_data.py
```

`collect_data.py` fetches current S&P 500 constituents and historical component changes from Wikipedia, reconstructs daily historical membership from `2021-01-01` through `2025-12-31`, and downloads daily **Adjusted Close** (`Adj Close`) prices from Yahoo Finance aligned to `SPY`'s NYSE trading calendar.

---

## Dataset Overview

| File | Shape / Size | Description |
| :--- | :--- | :--- |
| `data/adj_close.parquet` | `(1255, 594)` | Daily `Adj Close` prices from `2021-01-04` to `2025-12-31` for `SPY` and all 593 historical S&P 500 tickers in that window. Accounts for stock splits and dividends. |
| `data/constituents_mask.parquet` | `(1255, 593)` | Boolean mask (`True`/`False`) indexed by the same trading days, indicating whether each ticker was an S&P 500 member on that date (504–508 members/day). |
| `data/rebalance_events.csv` | `99` rows | S&P 500 addition and removal events (`date`, `added`, `removed`, `reason`) within 2021–2025. |
| `data/ticker_coverage.csv` | `593` rows | Summary of `valid_price_days`, `sp500_member_days`, `complete_over_5yr`, `complete_while_member`, and GICS sector per ticker. |

### Coverage Notes
* **529 / 593** tickers have complete `Adj Close` data across all 1,255 trading days.
* **47** historical tickers returned no data from Yahoo Finance (primarily delisted, acquired, or renamed companies such as `ATVI`, `TWTR`, `FRC`, `SIVB`, `XLNX`).
* **3** tickers have partial price data while active S&P 500 members.

---

## How to Use the Dataset

### 1. Load Prices and Compute Simple Daily Returns
Use **simple returns** ($r_t = \frac{P_t - P_{t-1}}{P_{t-1}}$) rather than log returns so that portfolio returns are linear in asset weights ($r_{\text{portfolio}} = Xw$).

```python
import pandas as pd

adj_close = pd.read_parquet("data/adj_close.parquet")
mask = pd.read_parquet("data/constituents_mask.parquet")

# Simple daily returns
returns = adj_close.pct_change().iloc[1:]
mask = mask.loc[returns.index]

y = returns["SPY"]
X_all = returns.drop(columns=["SPY"])
```

### 2. Slice a 1-Year Train / 1-Year Test Window (Walk-Forward)
To avoid survivorship bias and lookahead bias, select the eligible universe for a given training year (e.g., train on `2023`, evaluate on `2024`) using tickers that:
1. Were S&P 500 members at the end of the training year (or throughout the training year), **and**
2. Have complete price/return data over the training year (and optionally the test year, depending on your delisting policy).

```python
train_slice = slice("2023-01-01", "2023-12-31")
test_slice = slice("2024-01-01", "2024-12-31")

X_train_raw = X_all.loc[train_slice]
y_train = y.loc[train_slice]
X_test_raw = X_all.loc[test_slice]
y_test = y.loc[test_slice]

# Universe: S&P 500 members as of the last trading day of the training year
# with complete return data across the training year
last_train_day = X_train_raw.index[-1]
is_member = mask.loc[last_train_day]
has_complete_train = X_train_raw.notna().all(axis=0)

eligible_tickers = is_member.index[is_member & has_complete_train]

X_train = X_train_raw[eligible_tickers]
X_test = X_test_raw[eligible_tickers].fillna(0.0)  # or drop/handle delistings explicitly
```
