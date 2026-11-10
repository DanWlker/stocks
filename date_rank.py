"""Normal monthly DCA on each date of the month (1st..31st), ranked, over the last 10, 15 and 20 years of the S&P 500.

$BUY each month on date D, bought on the first trading day on or after D (the month's last trading day
if none). Ranked by final value at the last close. Buying later in the month puts each dollar in the
market later, which costs a little in a rising market; "vs surrounding month" removes that effect
(close vs the centred 21-trading-day average, 10% trimmed mean, the measure from buy_day.py).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from buy_day import by_calendar_date, trimmed_mean, vs_centred_window
from dca import BUY, result

YEARS = 10
WINDOWS = [10, 15, 20]  # years, the windows the page offers


def rank_dates(close, years=YEARS):
    """close must end on a month's last trading day (sp500_full is fetched to a month end)."""
    periods = close.index.to_period("M")
    first = periods[-1] - (12 * years - 1)
    win = close[periods >= first]
    months = [m for _, m in win.groupby(win.index.to_period("M"))]
    # centred window over the full series, so the first months in the window still have neighbours
    rel = by_calendar_date(vs_centred_window(close))
    rel = rel[rel["month"] >= first]
    vs_month = rel.groupby("date")["rel"].apply(trimmed_mean) * 100
    rows = []
    for d in range(1, 32):
        dates = [m.index[min(np.searchsorted(m.index.day, d), len(m) - 1)] for m in months]
        invested, value, avg, _ = result(win, dates, BUY)
        rows.append({"date": d, "invested": invested, "value": value, "avg_cost": avg, "vs_month_%": vs_month[d]})
    df = pd.DataFrame(rows).set_index("date")
    df["vs_best_%"] = (df["value"] / df["value"].max() - 1) * 100
    df["rank"] = df["value"].rank(ascending=False, method="min").astype(int)
    df["vs_month_rank"] = df["vs_month_%"].rank(method="min").astype(int)
    meta = {"first": str(win.index[0].date()), "last": str(win.index[-1].date()), "months": len(months),
            "vs_month_months": rel["month"].nunique()}
    return df.sort_values("rank"), meta


def load():
    return pd.read_csv(Path(__file__).parent / "data" / "sp500_full.csv", parse_dates=["Date"], index_col="Date")["Close"]


if __name__ == "__main__":
    close = load()
    pd.set_option("display.width", 140)
    for years in WINDOWS:
        df, meta = rank_dates(close, years)
        # cross-check: date 1 is plain DCA on each month's first trading day
        periods = close.index.to_period("M")
        win = close[periods >= periods[-1] - (12 * years - 1)]
        assert np.isclose(df.loc[1, "value"], result(win, list(win.groupby(win.index.to_period("M")).head(1).index), BUY)[1])
        if years != WINDOWS[0]:
            print()
        print(f"S&P 500 {meta['first']} -> {meta['last']}, {meta['months']} months of ${BUY:,.0f} "
              f"(vs-month column over {meta['vs_month_months']} months; the last month lacks a full window)")
        print(df.to_string(formatters={"invested": "${:,.0f}".format, "value": "${:,.0f}".format, "avg_cost": "{:.2f}".format,
                                       "vs_month_%": "{:+.2f}".format, "vs_best_%": "{:+.2f}".format}))
