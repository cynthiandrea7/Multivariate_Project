"""
fetch_sp500_data.py

Builds an S&P 500 dataset with:
- 5y daily OHLCV prices (yfinance)
- Sector metadata (Wikipedia: List of S&P 500 companies)
- Index weights (SlickCharts S&P 500 weights)

Outputs:
- data/sp500_sectors.csv
- data/sp500_weights_slickcharts.csv
- data/sp500_5yr_with_sectors_weights.csv
- data/sp500_5yr_with_sectors_weights.parquet
"""

import os
import re
import time
import requests
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from bs4 import BeautifulSoup


WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SLICK_URL = "https://www.slickcharts.com/sp500"


# ----------------------------
# Helpers
# ----------------------------
def ensure_data_dir(path: str = "data") -> None:
    os.makedirs(path, exist_ok=True)


def standardize_symbol(sym: str) -> str:
    """
    Standardize tickers for Yahoo Finance format.
    Wikipedia uses BRK.B; yfinance uses BRK-B.
    """
    if sym is None:
        return sym
    sym = str(sym).strip().upper()
    sym = sym.replace(".", "-")
    return sym


# ----------------------------
# STEP 1: Fetch sectors (Wikipedia)
# ----------------------------
def fetch_sp500_sectors(output_path: str = "data/sp500_sectors.csv") -> pd.DataFrame:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    resp = requests.get(WIKI_URL, headers=headers, timeout=30)
    resp.raise_for_status()

    tables = pd.read_html(resp.text)

    # On Wikipedia page, constituents table is typically the first large table.
    # In your earlier code you used tables[1]; to be safer, search by columns.
    df = None
    for t in tables:
        if {"Symbol", "Security", "GICS Sector"}.issubset(set(t.columns)):
            df = t.copy()
            break
    if df is None:
        raise RuntimeError("Could not find S&P 500 constituents table on Wikipedia page.")

    df = df.rename(columns={"GICS Sector": "Sector"})
    df = df[["Symbol", "Security", "Sector"]].copy()
    df["Symbol"] = df["Symbol"].apply(standardize_symbol)

    df = df.drop_duplicates(subset="Symbol").reset_index(drop=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[OK] Saved sectors: {len(df)} rows -> {output_path}")
    return df


# ----------------------------
# STEP 2: Fetch weights (SlickCharts)
# ----------------------------
def fetch_sp500_weights(output_path: str = "data/sp500_weights_slickcharts.csv") -> pd.DataFrame:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    resp = requests.get(SLICK_URL, headers=headers, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"class": re.compile(r".*table.*")})
    if table is None:
        raise RuntimeError("Could not find weights table on SlickCharts.")

    # Parse header
    thead = table.find("thead")
    ths = thead.find_all("th")
    columns = [th.get_text(strip=True) for th in ths]

    # Parse rows
    rows = []
    tbody = table.find("tbody")
    for tr in tbody.find_all("tr"):
        tds = tr.find_all("td")
        rows.append([td.get_text(strip=True) for td in tds])

    df = pd.DataFrame(rows, columns=columns)

    # Standardize expected columns
    # SlickCharts typically has: "#", "Company", "Symbol", "Weight", "Price", "Chg", "% Chg"
    if "Symbol" not in df.columns or "Weight" not in df.columns:
        raise RuntimeError(f"Unexpected SlickCharts columns: {df.columns.tolist()}")

    df["Symbol"] = df["Symbol"].apply(standardize_symbol)

    # Clean numeric fields
    df["Weight"] = (
        df["Weight"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce") / 100.0

    # Optional: clean Price / Chg / % Chg if present
    for col in ["Price", "Chg"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "% Chg" in df.columns:
        df["% Chg"] = (
            df["% Chg"].astype(str)
            .str.replace("%", "", regex=False)
            .str.replace("(", "-", regex=False)
            .str.replace(")", "", regex=False)
            .str.replace("--", "-", regex=False)
            .str.strip()
        )
        df["% Chg"] = pd.to_numeric(df["% Chg"], errors="coerce") / 100.0

    df = df.drop_duplicates(subset="Symbol").reset_index(drop=True)

    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[OK] Saved weights: {len(df)} rows -> {output_path}")
    return df


# ----------------------------
# STEP 3: Download prices (yfinance)
# ----------------------------
def download_prices(
    tickers: list[str],
    period: str = "5y",
    interval: str = "1d",
    sleep_seconds: float = 0.0,
) -> pd.DataFrame:
    all_data = []

    for ticker in tqdm(tickers, desc="Downloading price data"):
        try:
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=True,
            )

            if df is None or df.empty:
                continue

            df = df.copy()
            df["Symbol"] = standardize_symbol(ticker)

            df = df.reset_index()

            # Flatten MultiIndex columns if needed
            df.columns = [c if not isinstance(c, tuple) else c[0] for c in df.columns]

            keep_cols = ["Date", "Open", "High", "Low", "Close", "Volume", "Symbol"]
            df = df[[c for c in keep_cols if c in df.columns]]

            all_data.append(df)

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        except Exception as e:
            print(f"[WARN] Error downloading {ticker}: {e}")
            continue

    if not all_data:
        raise RuntimeError("No price data downloaded. Check tickers / connection / yfinance limits.")

    return pd.concat(all_data, axis=0, ignore_index=True)


# ----------------------------
# STEP 4: Merge + Save
# ----------------------------
def build_dataset(
    out_csv: str = "data/sp500_5yr_with_sectors_weights.csv",
    out_parquet: str = "data/sp500_5yr_with_sectors_weights.parquet",
    period: str = "5y",
    interval: str = "1d",
) -> pd.DataFrame:
    ensure_data_dir("data")

    # Fetch metadata
    df_sectors = fetch_sp500_sectors("data/sp500_sectors.csv")
    df_weights = fetch_sp500_weights("data/sp500_weights_slickcharts.csv")

    # Merge sectors + weights into one meta table
    df_meta = df_sectors.merge(df_weights[["Symbol", "Weight"]], on="Symbol", how="left")

    # Download prices for all tickers from meta
    tickers = df_meta["Symbol"].dropna().unique().tolist()
    print(f"[INFO] Loaded {len(tickers)} tickers")

    df_prices = download_prices(tickers, period=period, interval=interval)

    # Standardize symbols
    df_prices["Symbol"] = df_prices["Symbol"].apply(standardize_symbol)
    df_meta["Symbol"] = df_meta["Symbol"].apply(standardize_symbol)

    # Merge prices with meta
    df_final = df_prices.merge(df_meta, on="Symbol", how="left")

    # Save
    df_final.to_csv(out_csv, index=False)
    df_final.to_parquet(out_parquet, index=False)

    print("\n[DONE] Saved:")
    print(f" - {out_csv}")
    print(f" - {out_parquet}")
    print("\n[PREVIEW]")
    print(df_final.head())

    return df_final


def main():
    build_dataset()


if __name__ == "__main__":
    main()