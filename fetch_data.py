"""Download daily index data from Yahoo Finance, and monthly inflation, interest-rate and valuation data
(FRED, Robert Shiller) into data/.

Only what is missing is downloaded: a file that is not there yet is fetched in full, and an open-ended series
(end None) gets the rows after its last stored date appended. Yahoo or FRED revisions of rows already stored are
not picked up; delete a file to re-download it in full."""

import urllib.request
from pathlib import Path

import pandas as pd
import yfinance as yf

# (file name, ticker, start, end inclusive or None for up to yesterday)
SERIES = [
    ("depression_sp500", "^GSPC", "1927-12-30", "1939-12-31"),
    ("dotcom_sp500", "^GSPC", "1995-01-01", "2007-12-31"),
    ("dotcom_nasdaq", "^IXIC", "1995-01-01", "2007-12-31"),
    ("sp500_full", "^GSPC", "1927-12-30", None),
    # used to cross-check the best-buy-date result (buy_day_verify.py)
    ("sp500_tr", "^SP500TR", "1988-01-01", None),
    ("nasdaq_full", "^IXIC", "1971-02-05", None),
    ("dow", "^DJI", "1992-01-01", None),
    ("russell2000", "^RUT", "1987-09-10", None),
    ("ftse100", "^FTSE", "1984-01-01", None),
    ("dax", "^GDAXI", "1987-12-30", None),
    ("nikkei225", "^N225", "1965-01-01", None),
]

FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
# (file name, url, grows): saved as downloaded; monthly.py reads them. Growing files get new months appended.
RAW = [
    # S&P price, dividends, CPI, 10-year yield and CAPE, monthly from 1871 (the file ends September 2023)
    ("shiller_ie_data.xls", "http://www.econ.yale.edu/~shiller/data/ie_data.xls", False),
    ("fred_CPIAUCNS.csv", FRED.format("CPIAUCNS"), True),  # consumer prices, from 1913
    ("fred_TB3MS.csv", FRED.format("TB3MS"), True),  # 3-month Treasury bill rate, from 1934
    ("fred_M1329AUSM193NNBR.csv", FRED.format("M1329AUSM193NNBR"), False),  # NBER short-term government rate, 1920-1934
    ("fred_GS10.csv", FRED.format("GS10"), True),  # 10-year Treasury yield, from 1953
]

out = Path(__file__).parent / "data"
out.mkdir(exist_ok=True)
# yfinance's end is exclusive: stopping before today keeps an unfinished trading day out of the files
today = pd.Timestamp.today().normalize()

for name, ticker, start, end in SERIES:
    path = out / f"{name}.csv"
    old = pd.read_csv(path, index_col="Date") if path.exists() else None
    if old is not None and end is not None:
        print(f"{name}: complete, skipped")
        continue
    if old is not None:
        start = (pd.Timestamp(old.index[-1]) + pd.Timedelta(days=1)).date()
    end_excl = today.date() if end is None else (pd.Timestamp(end) + pd.Timedelta(days=1)).date()
    if old is not None and start >= end_excl:
        print(f"{name}: up to date at {old.index[-1]}")
        continue
    df = yf.Ticker(ticker).history(start=start, end=end_excl, auto_adjust=False)
    if df.empty:
        if old is None:
            raise SystemExit(f"{ticker}: no data returned")
        print(f"{name}: no new rows after {old.index[-1]}")
        continue
    df.index = pd.Index([d.isoformat() for d in df.index.tz_localize(None).date], name="Date")
    if old is not None:
        df = pd.concat([old, df[old.columns]])
    else:
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        # Yahoo's pre-1962 rows copy Close into O/H/L and report zero volume; keep only the real column
        if (df[["Open", "High", "Low"]].eq(df["Close"], axis=0).all().all()) and (df["Volume"] == 0).all():
            df = df[["Close"]]
    added = len(df) - (0 if old is None else len(old))
    df.to_csv(path)
    print(f"{name}: {ticker} +{added} rows, {df.index[0]} -> {df.index[-1]}")

for name, url, grows in RAW:
    path = out / name
    if path.exists() and not grows:
        print(f"{name}: complete, skipped")
        continue
    if path.exists():
        lines = path.read_text().splitlines()
        last = lines[-1].split(",")[0]
        url += f"&cosd={last}"  # FRED: observations from this date on, the stored last month included
    with urllib.request.urlopen(url, timeout=120) as r:
        body = r.read()
    if path.exists():
        new = [l for l in body.decode().splitlines()[1:] if l.split(",")[0] > last]
        path.write_text("\n".join(lines + new) + "\n")
        print(f"{name}: +{len(new)} months, last {(new or lines)[-1].split(',')[0]}")
    else:
        path.write_bytes(body)
        print(f"{name}: {len(body):,} bytes from {url}")
