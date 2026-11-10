"""Worst single-day close-to-close drops per dataset."""

from pathlib import Path

import pandas as pd

data = Path(__file__).parent / "data"

for name in ["depression_sp500", "dotcom_sp500", "dotcom_nasdaq"]:
    close = pd.read_csv(data / f"{name}.csv", parse_dates=["Date"], index_col="Date")["Close"]
    ret = close.pct_change().dropna()
    print(f"== {name}")
    for date, r in ret.nsmallest(5).items():
        prev = close.shift(1)[date]
        print(f"  {date.date()}  {r:+.2%}  ({prev:.2f} -> {close[date]:.2f})")
    print(f"  days with a drop of 10% or more: {(ret <= -0.10).sum()}")
