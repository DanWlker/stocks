"""Monthly US market data 1928-01 .. 2026-10 for the macro analyses, plus a daily S&P total return.

monthly() columns, indexed by month start:
  stock_ret  S&P 500 total return (dividends reinvested), from month-average levels like Shiller's:
             Shiller's real total-return price x CPI through 1988-01, then ^SP500TR monthly averages
  price      S&P 500 price (month average), for trend signals
  cpi        consumer prices (FRED CPIAUCNS); October 2025 was never published (US government shutdown) and is
             interpolated between September and November 2025
  cash_ret   3-month bills: NBER short-term government rate through 1933-12, FRED TB3MS from 1934; annual rate / 12
  bond_ret   10-year Treasury bought at par at the end of each month and sold a month later at the new yield
             (yield: Shiller through 1953-03, FRED GS10 after)
  cape       Shiller CAPE; ends 2023-09 (NaN after)
daily_tr(): S&P daily closes with dividends accrued each trading day at Shiller's dividend yield / 252 before
            1988, and ^SP500TR from 1988.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# pandas warns that xlrd will go; it is the reader that handles Shiller's .xls today
warnings.filterwarnings("ignore", message="The xlrd engine is deprecated")

data = Path(__file__).parent / "data"
START, END = "1928-01", "2026-10"


def fred(series):
    s = pd.read_csv(data / f"fred_{series}.csv", parse_dates=["observation_date"], index_col="observation_date").iloc[:, 0]
    return pd.to_numeric(s, errors="coerce").to_period("M")


def shiller():
    raw = pd.read_excel(data / "shiller_ie_data.xls", sheet_name="Data", header=None, skiprows=8, engine="xlrd")
    raw = raw[pd.to_numeric(raw[0], errors="coerce").notna()]
    d = raw[0].astype(float)
    year = d.astype(int)
    month = ((d - year) * 100).round().astype(int)  # 1871.1 is October
    idx = pd.PeriodIndex.from_fields(year=year, month=month, freq="M")
    cols = {"P": 1, "D": 2, "CPI": 4, "GS10": 6, "real_tr": 9, "cape": 12}
    return pd.DataFrame({k: pd.to_numeric(raw[c], errors="coerce").to_numpy() for k, c in cols.items()}, index=idx)


def daily_close(name):
    return pd.read_csv(data / f"{name}.csv", parse_dates=["Date"], index_col="Date")["Close"]


def bond_returns(yield_pct):
    """Monthly return of a 10-year par bond bought at last month's yield, repriced at this month's yield."""
    y = yield_pct / 100
    c = y.shift(1)
    n = 2 * (10 - 1 / 12)  # half-year periods left after one month
    price = c / y * (1 - (1 + y / 2) ** -n) + (1 + y / 2) ** -n
    return price - 1 + c / 12


def monthly():
    sh = shiller()
    tr_shiller = (sh["real_tr"] * sh["CPI"]).pct_change()
    tr_yahoo = daily_close("sp500_tr").groupby(lambda d: d.to_period("M")).mean().pct_change()
    stock_ret = tr_shiller[:"1988-01"].combine_first(tr_yahoo["1988-02":])
    price_y = daily_close("sp500_full").groupby(lambda d: d.to_period("M")).mean()
    price = sh["P"][:"1927-12"].combine_first(price_y)
    short = fred("M1329AUSM193NNBR")[:"1933-12"].combine_first(fred("TB3MS")["1934-01":])
    gs10 = sh["GS10"][:"1953-03"].combine_first(fred("GS10")["1953-04":])
    df = pd.DataFrame({
        "stock_ret": stock_ret, "price": price, "cpi": fred("CPIAUCNS").interpolate(limit=1, limit_area="inside"),
        "cash_ret": short / 100 / 12, "bond_ret": bond_returns(gs10), "cape": sh["cape"],
    })
    df = df[START:END]
    missing = df.drop(columns="cape").isna().sum()
    assert not missing.any(), missing[missing > 0]
    return df


def daily_tr():
    px = daily_close("sp500_full")
    sh = shiller()
    dy = (sh["D"] / sh["P"]).reindex(px.index.to_period("M")).ffill().to_numpy()  # trailing 12-month dividend yield
    approx = px * np.cumprod(1 + np.nan_to_num(dy) / 252)
    tr = daily_close("sp500_tr")
    return pd.concat([approx[:tr.index[0]].iloc[:-1], tr * approx[:tr.index[0]].iloc[-1] / tr.iloc[0]])


def cagr(growth, months):
    return growth ** (12 / months) - 1


def max_drawdown(values):
    values = pd.Series(values)
    return (values / values.cummax() - 1).min()


# self-check: an unchanged yield returns exactly the coupon
assert np.allclose(bond_returns(pd.Series([5.0, 5.0, 5.0]))[1:], 0.05 / 12)

if __name__ == "__main__":
    df = monthly()
    print(f"monthly {df.index[0]} .. {df.index[-1]} ({len(df)} months); CAPE until {df['cape'].last_valid_index()}")
    for c in ["stock_ret", "cash_ret", "bond_ret"]:
        print(f"  {c:<9} annualised {cagr((1 + df[c]).prod(), len(df)):+.2%}")
    print(f"  inflation  annualised {cagr(df['cpi'].iloc[-1] / df['cpi'].iloc[0], len(df) - 1):+.2%}")

    sh = shiller()
    a = (sh["real_tr"] * sh["CPI"]).pct_change()["1988-02":"2023-08"]
    b = daily_close("sp500_tr").groupby(lambda d: d.to_period("M")).mean().pct_change()["1988-02":"2023-08"]
    print(f"\ncheck: Shiller vs ^SP500TR monthly returns 1988-02..2023-08: correlation {a.corr(b):.4f}, "
          f"annualised {cagr((1 + a).prod(), len(a)):+.2%} vs {cagr((1 + b).prod(), len(b)):+.2%}")
    y = sh["GS10"]["1953-04":"2023-08"]
    f = fred("GS10")["1953-04":"2023-08"]
    print(f"check: Shiller vs FRED 10-year yield 1953-04..2023-08: max difference {(y - f).abs().max():.3f} points")

    # the dividend approximation, tried where the real total return is known
    px = daily_close("sp500_full")["1988-01-04":"2023-08-31"]
    dy = (sh["D"] / sh["P"]).reindex(px.index.to_period("M")).ffill().to_numpy()
    approx = px * np.cumprod(1 + dy / 252)
    real = daily_close("sp500_tr")["1988-01-04":"2023-08-31"]
    yrs = (px.index[-1] - px.index[0]).days / 365.25
    print(f"check: daily dividend approximation 1988-2023: {(approx.iloc[-1] / approx.iloc[0]) ** (1 / yrs) - 1:+.2%}/yr "
          f"vs actual ^SP500TR {(real.iloc[-1] / real.iloc[0]) ** (1 / yrs) - 1:+.2%}/yr")
    t = daily_tr()
    print(f"daily_tr {t.index[0].date()} .. {t.index[-1].date()}, {len(t)} days")
