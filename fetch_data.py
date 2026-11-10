"""Download daily index data for the Great Depression and dot-com eras into data/."""

from pathlib import Path

import pandas as pd
import yfinance as yf

# (file name, ticker, start, end inclusive)
SERIES = [
    ("depression_sp500", "^GSPC", "1927-12-30", "1939-12-31"),
    ("dotcom_sp500", "^GSPC", "1995-01-01", "2007-12-31"),
    ("dotcom_nasdaq", "^IXIC", "1995-01-01", "2007-12-31"),
    ("sp500_full", "^GSPC", "1927-12-30", "2026-10-31"),
    # used to cross-check the best-buy-date result (buy_day_verify.py)
    ("sp500_tr", "^SP500TR", "1988-01-01", "2026-10-31"),
    ("nasdaq_full", "^IXIC", "1971-02-05", "2026-10-31"),
    ("dow", "^DJI", "1992-01-01", "2026-10-31"),
    ("russell2000", "^RUT", "1987-09-10", "2026-10-31"),
    ("ftse100", "^FTSE", "1984-01-01", "2026-10-31"),
    ("dax", "^GDAXI", "1987-12-30", "2026-10-31"),
    ("nikkei225", "^N225", "1965-01-01", "2026-10-31"),
]

out = Path(__file__).parent / "data"
out.mkdir(exist_ok=True)

for name, ticker, start, end in SERIES:
    # yfinance's end is exclusive, so pass the next day
    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).date()
    df = yf.Ticker(ticker).history(start=start, end=end_excl, auto_adjust=False)
    if df.empty:
        raise SystemExit(f"{ticker}: no data returned")
    df.index = df.index.tz_localize(None).date
    df.index.name = "Date"
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    # Yahoo's pre-1962 rows copy Close into O/H/L and report zero volume; keep only the real column
    if (df[["Open", "High", "Low"]].eq(df["Close"], axis=0).all().all()) and (df["Volume"] == 0).all():
        df = df[["Close"]]
    df.to_csv(out / f"{name}.csv")
    print(f"{name}: {ticker} {len(df)} rows {df.index[0]} -> {df.index[-1]}")
