"""Which date of the month is cheapest to buy on?

Each day's close is compared with the average close of the surrounding 21 trading days (centred,
about one month): -0.30% means a buy that day paid 0.30% less than prices around it. A centred
window cancels the market's upward drift; comparing with the calendar month's own average instead
(also printed) makes early dates look cheap only because prices rise during a month.
For calendar date D the buy happens on the first
trading day on or after D (the month's last trading day if none), like an automatic DCA order.
Outlier months are handled with a 10% trimmed mean (and the median); ci95 is the 95% confidence
interval of the plain mean.
"""

from pathlib import Path

import numpy as np
import pandas as pd

TRIM = 0.10
MIN_DAYS = 15  # drops the partial months at the ends of a dataset
WINDOW = 21  # trading days in the centred comparison window


def trimmed_mean(x, trim=TRIM):
    x = np.sort(np.asarray(x))
    k = int(len(x) * trim)
    return x[k:len(x) - k].mean()


def vs_month_average(close):
    return close / close.groupby(close.index.to_period("M")).transform("mean") - 1


def vs_centred_window(close):
    return close / close.rolling(WINDOW, center=True).mean() - 1


def by_calendar_date(rel):
    """Rows of (date D, month, rel on the day a buy for date D lands) for D = 1..31."""
    rows = []
    for month, m in rel.groupby(rel.index.to_period("M")):
        if len(m) < MIN_DAYS or m.isna().any():  # partial months, and the window's NaN edges
            continue
        pos = np.minimum(np.searchsorted(m.index.day, np.arange(1, 32)), len(m) - 1)
        rows += [(d, month, r) for d, r in zip(range(1, 32), m.to_numpy()[pos])]
    return pd.DataFrame(rows, columns=["date", "month", "rel"])


def summary(df, key):
    g = df.groupby(key)["rel"]
    return pd.DataFrame({
        "months": g.size(),
        "trim_mean_%": g.apply(trimmed_mean) * 100,
        "median_%": g.median() * 100,
        "mean_%": g.mean() * 100,
        "ci95_%": 1.96 * g.std() / np.sqrt(g.size()) * 100,
        "cheaper_than_avg": g.apply(lambda s: (s < 0).mean()),
    })


# self-checks: trimming drops the outlier; date 3 falls through to the 5th, date 6 to the month's last day
assert trimmed_mean([1, 2, 3, 4, 5, 6, 7, 8, 9, 1000]) == 5.5
_c = pd.Series([10, 12, 8.0] + [10.0] * 14, index=pd.to_datetime(["2001-01-01", "2001-01-02", "2001-01-05"]
               + [f"2001-01-{d}" for d in range(15, 29)]))
_r = by_calendar_date(vs_month_average(_c)).set_index("date")["rel"]
assert np.isclose(_r[3], 8 / _c.mean() - 1) and np.isclose(_r[31], 10 / _c.mean() - 1)

if __name__ == "__main__":
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", "{:.3f}".format)
    data = Path(__file__).parent / "data"
    for name in ["sp500_full", "depression_sp500", "dotcom_sp500", "dotcom_nasdaq"]:
        close = pd.read_csv(data / f"{name}.csv", parse_dates=["Date"], index_col="Date")["Close"]
        naive = summary(by_calendar_date(vs_month_average(close)), "date").sort_values("trim_mean_%")
        df = by_calendar_date(vs_centred_window(close))
        s = summary(df, "date")
        print(f"\n== {name}: {df['month'].nunique()} months {df['month'].min()} -> {df['month'].max()}")
        print("  (biased) vs month average, cheapest:", ", ".join(f"{d} ({v:+.2f}%)" for d, v in naive["trim_mean_%"].head(5).items()))
        if name == "sp500_full":
            print(s.to_string())
        ranked = s.sort_values("trim_mean_%")
        print("  cheapest dates (trimmed mean):", ", ".join(f"{d} ({v:+.2f}%)" for d, v in ranked["trim_mean_%"].head(5).items()))
        print("  dearest dates  (trimmed mean):", ", ".join(f"{d} ({v:+.2f}%)" for d, v in ranked["trim_mean_%"].tail(5)[::-1].items()))

        # paired test: cheapest vs dearest date in the same months
        wide = df.pivot(index="month", columns="date", values="rel")
        best, worst = ranked.index[0], ranked.index[-1]
        diff = (wide[best] - wide[worst]) * 100
        print(f"  date {best} vs date {worst}, same months: trimmed mean {trimmed_mean(diff):+.3f}%, "
              f"mean {diff.mean():+.3f}% ± {1.96 * diff.std() / np.sqrt(len(diff)):.3f}% (95% CI), "
              f"date {best} cheaper in {(diff < 0).mean():.0%} of months")

        # same question by decade, to see whether the pattern is stable over time (full history only)
        if name == "sp500_full":
            df["decade"] = df["month"].map(lambda p: p.year // 10 * 10)
            dec = df.groupby(["decade", "date"])["rel"].apply(trimmed_mean).unstack() * 100
            print("\n  cheapest 3 dates per decade (trimmed mean %):")
            for decade, row in dec.iterrows():
                r = row.sort_values()
                print(f"    {decade}s: " + ", ".join(f"{d} ({v:+.2f})" for d, v in r.head(3).items())
                      + f"   | 1st: {row[1]:+.2f}, 15th: {row[15]:+.2f}, 28th: {row[28]:+.2f}")
