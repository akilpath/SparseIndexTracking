"""Collect 2021-2025 S&P 500 daily Adjusted Close prices, SPY benchmark, and daily membership mask."""

from __future__ import annotations

import io
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests
import yfinance as yf

START_DATE = "2021-01-01"
END_DATE = "2026-01-01"  # Exclusive upper bound in yfinance -> covers through 2025-12-31
BENCHMARK_TICKER = "SPY"
WIKI_CURRENT_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKI_CHANGES_URL = "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"
DATA_DIR = Path(__file__).resolve().parent / "data"


def normalize_ticker(ticker: object) -> str | None:
    """Normalize ticker symbols to Yahoo Finance format (e.g., BRK.B -> BRK-B)."""
    if pd.isna(ticker):
        return None
    s = str(ticker).strip().upper()
    if not s or s in {"NAN", "NONE", "N/A", "-"}:
        return None
    # Strip any footnote markers like [1] if present
    if "[" in s:
        s = s.split("[", 1)[0].strip()
    return s.replace(".", "-") if s else None


def fetch_sp500_wikipedia() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch current S&P 500 constituents and historical changes from Wikipedia."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    resp_curr = requests.get(WIKI_CURRENT_URL, headers=headers, timeout=30)
    resp_curr.raise_for_status()
    curr_tables = pd.read_html(io.StringIO(resp_curr.text))
    if not curr_tables:
        raise RuntimeError("No tables found on current S&P 500 Wikipedia page")

    current_df = curr_tables[0].copy()
    current_df["Symbol"] = current_df["Symbol"].map(normalize_ticker)
    current_df = current_df.dropna(subset=["Symbol"]).drop_duplicates(subset=["Symbol"])

    resp_chg = requests.get(WIKI_CHANGES_URL, headers=headers, timeout=30)
    resp_chg.raise_for_status()
    chg_tables = pd.read_html(io.StringIO(resp_chg.text))
    if not chg_tables:
        raise RuntimeError("No tables found on historical S&P 500 components Wikipedia page")

    raw_changes = chg_tables[0].copy()
    if isinstance(raw_changes.columns, pd.MultiIndex):
        flat_cols = []
        for col in raw_changes.columns:
            parts = [str(c).strip() for c in col if str(c).strip() and not str(c).startswith("Unnamed")]
            flat_cols.append("_".join(dict.fromkeys(parts)))
        raw_changes.columns = flat_cols

    # Identify relevant columns robustly
    date_col = next(c for c in raw_changes.columns if "date" in c.lower())
    added_col = next(c for c in raw_changes.columns if "added" in c.lower() and "ticker" in c.lower())
    removed_col = next(c for c in raw_changes.columns if "removed" in c.lower() and "ticker" in c.lower())
    reason_col = next((c for c in raw_changes.columns if "reason" in c.lower()), None)

    changes_df = pd.DataFrame(
        {
            "date": pd.to_datetime(raw_changes[date_col], errors="coerce"),
            "added": raw_changes[added_col].map(normalize_ticker),
            "removed": raw_changes[removed_col].map(normalize_ticker),
            "reason": raw_changes[reason_col].astype(str) if reason_col else "",
        }
    )
    changes_df = changes_df.dropna(subset=["date"]).sort_values("date", ascending=False).reset_index(drop=True)
    return current_df, changes_df


def reconstruct_daily_membership(
    trading_dates: pd.DatetimeIndex,
    current_tickers: set[str],
    changes_df: pd.DataFrame,
) -> tuple[pd.DataFrame, set[str]]:
    """Walk backward from current constituents to build a daily boolean membership mask."""
    # Group changes by date so same-day additions/removals are applied atomically
    changes_by_date: dict[pd.Timestamp, list[tuple[str | None, str | None]]] = {}
    for row in changes_df.itertuples(index=False):
        changes_by_date.setdefault(row.date, []).append((row.added, row.removed))

    sorted_change_dates = sorted(changes_by_date.keys(), reverse=True)
    active = set(current_tickers)
    change_idx = 0

    daily_sets: dict[pd.Timestamp, set[str]] = {}
    all_seen_tickers: set[str] = set()

    # Walk trading dates in reverse chronological order
    for dt in sorted(trading_dates, reverse=True):
        while change_idx < len(sorted_change_dates) and sorted_change_dates[change_idx] > dt:
            c_date = sorted_change_dates[change_idx]
            for added, removed in changes_by_date[c_date]:
                if isinstance(added, str) and added:
                    active.discard(added)
                if isinstance(removed, str) and removed:
                    active.add(removed)
            change_idx += 1

        daily_sets[dt] = set(active)
        all_seen_tickers.update(active)

    all_tickers_sorted = sorted(all_seen_tickers)
    mask_data = np.zeros((len(trading_dates), len(all_tickers_sorted)), dtype=bool)
    ticker_to_col = {t: i for i, t in enumerate(all_tickers_sorted)}

    for row_idx, dt in enumerate(trading_dates):
        for t in daily_sets[dt]:
            mask_data[row_idx, ticker_to_col[t]] = True

    mask_df = pd.DataFrame(mask_data, index=trading_dates, columns=all_tickers_sorted)
    return mask_df, all_seen_tickers


def download_adj_close(
    tickers: list[str],
    start: str,
    end: str,
    batch_size: int = 75,
) -> pd.DataFrame:
    """Download daily Adjusted Close prices in batches with single-ticker fallback for failed batches."""
    frames: list[pd.DataFrame] = []
    total = len(tickers)

    for i in range(0, total, batch_size):
        batch = tickers[i : i + batch_size]
        print(f"Downloading batch {i // batch_size + 1}/{(total + batch_size - 1) // batch_size} ({len(batch)} tickers)...")
        data = yf.download(
            batch,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            threads=True,
        )

        if data.empty:
            continue

        if isinstance(data.columns, pd.MultiIndex):
            if "Adj Close" in data.columns.get_level_values(0):
                adj = data["Adj Close"].copy()
            elif "Close" in data.columns.get_level_values(0):
                adj = data["Close"].copy()
            else:
                continue
        else:
            # Single-ticker response
            col = "Adj Close" if "Adj Close" in data.columns else "Close"
            adj = data[[col]].rename(columns={col: batch[0]})

        if isinstance(adj, pd.Series):
            adj = adj.to_frame(name=batch[0])

        frames.append(adj)
        time.sleep(0.3)

    if not frames:
        raise RuntimeError("Failed to download any price data from Yahoo Finance.")

    combined = pd.concat(frames, axis=1)
    # Remove any duplicate columns if retries/overlaps occurred
    combined = combined.loc[:, ~combined.columns.duplicated()]
    combined.index = pd.to_datetime(combined.index).tz_localize(None)
    return combined.sort_index()


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Fetching S&P 500 constituents and historical changes from Wikipedia...")
    current_df, changes_df = fetch_sp500_wikipedia()
    current_tickers = set(current_df["Symbol"].tolist())
    print(f"   Current S&P 500 tickers: {len(current_tickers)}")
    print(f"   Total historical change rows parsed: {len(changes_df)}")

    print(f"2. Downloading {BENCHMARK_TICKER} benchmark ({START_DATE} to {END_DATE}) to establish trading calendar...")
    spy_df = download_adj_close([BENCHMARK_TICKER], START_DATE, END_DATE, batch_size=1)
    spy_series = spy_df[BENCHMARK_TICKER].dropna()
    trading_dates = spy_series.index
    print(f"   Trading days in window: {len(trading_dates)} ({trading_dates.min().date()} to {trading_dates.max().date()})")

    print("3. Reconstructing daily S&P 500 membership mask...")
    mask_df, all_historical_tickers = reconstruct_daily_membership(
        trading_dates=trading_dates,
        current_tickers=current_tickers,
        changes_df=changes_df,
    )
    print(f"   Unique tickers in S&P 500 during 2021-2025: {len(all_historical_tickers)}")

    print("4. Downloading daily Adjusted Close prices for all historical constituents...")
    stock_tickers = sorted(all_historical_tickers)
    stocks_adj = download_adj_close(stock_tickers, START_DATE, END_DATE, batch_size=75)

    # Align everything strictly to SPY's NYSE trading calendar
    aligned_cols: dict[str, pd.Series] = {BENCHMARK_TICKER: spy_series}
    for t in stock_tickers:
        if t in stocks_adj.columns:
            aligned_cols[t] = stocks_adj[t].reindex(trading_dates)
        else:
            aligned_cols[t] = pd.Series(np.nan, index=trading_dates, dtype=float)
    adj_close = pd.DataFrame(aligned_cols, index=trading_dates)

    # Filter rebalance events to our 2021-2025 window (plus slightly before/after for context)
    window_changes = changes_df[
        (changes_df["date"] >= pd.Timestamp(START_DATE))
        & (changes_df["date"] < pd.Timestamp(END_DATE))
    ].sort_values("date")

    # Build ticker coverage summary
    coverage_rows = []
    sector_map = current_df.set_index("Symbol")["GICS Sector"].to_dict() if "GICS Sector" in current_df.columns else {}
    name_map = current_df.set_index("Symbol")["Security"].to_dict() if "Security" in current_df.columns else {}

    for t in stock_tickers:
        col = adj_close[t]
        valid_days = int(col.notna().sum())
        member_days = int(mask_df[t].sum())
        member_valid_days = int((col.notna() & mask_df[t]).sum())
        coverage_rows.append(
            {
                "ticker": t,
                "security": name_map.get(t, ""),
                "gics_sector": sector_map.get(t, ""),
                "valid_price_days": valid_days,
                "sp500_member_days": member_days,
                "member_days_with_price": member_valid_days,
                "complete_over_5yr": valid_days == len(trading_dates),
                "complete_while_member": member_valid_days == member_days,
            }
        )
    coverage_df = pd.DataFrame(coverage_rows)

    # Save outputs
    adj_path = DATA_DIR / "adj_close.parquet"
    mask_path = DATA_DIR / "constituents_mask.parquet"
    events_path = DATA_DIR / "rebalance_events.csv"
    coverage_path = DATA_DIR / "ticker_coverage.csv"

    adj_close.to_parquet(adj_path)
    mask_df.to_parquet(mask_path)
    window_changes.to_csv(events_path, index=False)
    coverage_df.to_csv(coverage_path, index=False)

    missing_tickers = coverage_df.loc[coverage_df["valid_price_days"] == 0, "ticker"].tolist()
    incomplete_member = coverage_df.loc[
        (coverage_df["valid_price_days"] > 0) & (~coverage_df["complete_while_member"]),
        "ticker",
    ].tolist()

    print("\n=== Dataset Build Complete ===")
    print(f"Saved: {adj_path}  shape={adj_close.shape}")
    print(f"Saved: {mask_path}  shape={mask_df.shape}")
    print(f"Saved: {events_path}  rows={len(window_changes)}")
    print(f"Saved: {coverage_path}  rows={len(coverage_df)}")
    print(f"Tickers with complete 5-year prices: {int(coverage_df['complete_over_5yr'].sum())}/{len(stock_tickers)}")
    print(f"Tickers with 0 price rows (likely delisted/renamed): {len(missing_tickers)}")
    if missing_tickers:
        print(f"   Missing tickers: {', '.join(missing_tickers[:25])}" + (" ..." if len(missing_tickers) > 25 else ""))
    if incomplete_member:
        print(f"Tickers with partial prices while an S&P 500 member: {len(incomplete_member)}")


if __name__ == "__main__":
    main()
