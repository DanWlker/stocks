"""#4 International: crashes and DCA on the Nikkei 225, DAX and FTSE 100 (S&P 500 for reference).
Price indices in local currency, no dividends (Yahoo has no total-return series for them), so recoveries look
slower than an investor who reinvested dividends saw; the DAX is a total-return index by design, so it is not.
"""

import pandas as pd

from crashes import from_ath
from dca import BUY, result
from monthly import daily_close
from report import section as make_section, table, text

INDICES = [("nikkei225", "Nikkei 225"), ("dax", "DAX (includes dividends)"), ("ftse100", "FTSE 100"), ("sp500_full", "S&P 500")]
MONTH = 365.25 / 12


def underwater(close, buys, amount):
    """(share of days the DCA holdings were worth less than the money put in so far, last such day or None)."""
    bought = pd.Series(buys).value_counts().reindex(close.index, fill_value=0) * amount
    inv, shares = bought.cumsum(), (bought / close).cumsum()
    below = shares * close < inv * (1 - 1e-12)
    return below.mean(), (below[below].index[-1] if below.any() else None)


# self-check: bought at 10 and 8; worth 1,800 of 2,000 on day two, 2,700 on day three, 1,800 again on day four
_d = pd.bdate_range("2001-01-01", periods=5)
assert underwater(pd.Series([10, 8, 12, 8, 12.0], index=_d), [_d[0], _d[1]], 1000) == (2 / 5, _d[3])

def section():
    rows_f, rows_d, nikkei = [], [], {}
    for name, label in INDICES:
        close = daily_close(name)
        if name == "sp500_full":
            close = close["1965":]  # same era as the others
        crashes = from_ath(close)
        for p, t, r in sorted(crashes, key=lambda c: close[c[1]] / close[c[0]])[:4]:
            rec = [f"{r.date()}", f"{(r - p).days / MONTH / 12:.1f} yrs after the peak"] if r is not None else "not yet"
            rows_f.append([[label, f"{len(crashes)} falls of 20%+ since {close.index[0].year}"], [f"{p.date()}", f"{close[p]:,.0f}"],
                           f"{close[t] / close[p] - 1:+.1%}", f"{(t - p).days / MONTH:.0f}", rec])

        p = min(crashes, key=lambda c: close[c[1]] / close[c[0]])[0]
        w = close[p:]
        months = list(w.groupby(w.index.to_period("M")).head(1).index)
        months[0] = p  # first buy on the peak day itself
        inv, val, avg, irr = result(w, months, BUY)
        _, lump_val, _, lump_irr = result(w, [p], inv)
        under, last_under = underwater(w, months, BUY)
        lump_even = w[w >= w[p]].index[1] if (w >= w[p]).sum() > 1 else None
        last = [f"{last_under.date()}", f"{(last_under - p).days / MONTH / 12:.1f} yrs after the peak"] if last_under is not None else "never"
        even = [f"{lump_even.date()}", f"{(lump_even - p).days / MONTH / 12:.1f} yrs"] if lump_even is not None else "not yet"
        rows_d.append([[label, f"peak {p.date()}"], f"${inv:,.0f}", [f"${val:,.0f}", f"{val / inv - 1:+.0%}, {irr:+.1%} a year"],
                       f"{under:.0%}", last, [f"${lump_val:,.0f}", f"{lump_val / inv - 1:+.0%}, {lump_irr:+.1%} a year"], even])
        if name == "nikkei225":
            nikkei = {"peak": p, "years": (lump_even - p).days / MONTH / 12 if lump_even is not None else None,
                      "inv": inv, "val": val, "irr": irr, "lump_irr": lump_irr, "under": under}

    lede = (f"A single market can stall for decades. The Nikkei 225's {nikkei['peak'].date()} peak took {nikkei['years']:.1f} years to "
            f"regain (price only). Investing ${BUY:,.0f} a month from that peak still turned ${nikkei['inv']:,.0f} into "
            f"${nikkei['val']:,.0f} ({nikkei['irr']:+.1%} a year), although it was worth less than the money put in on "
            f"{nikkei['under']:.0%} of days; the same total as a lump sum at the peak managed {nikkei['lump_irr']:+.1%} a year.")
    return make_section(
        "international", "Beyond the US", "Crashes and DCA in Japan, Germany and the UK", lede,
        [table("The four deepest falls from a record high in each index", ["Index", "Peak", "Fall", "Months to bottom", "Back at the peak"], rows_f),
         table(f"${BUY:,.0f} a month from each index's worst peak, to today",
               ["Index", "Put in", "DCA now", "Days underwater", "Last underwater", "Lump sum at the peak now", "Lump back to even"], rows_d,
               note="Underwater: the DCA holdings worth less than the money put in so far. The lump sum invests the same total on the peak day.")],
        note="Price indices in local currency, without dividends (Yahoo has none for these), so recoveries look slower than an investor "
             "who reinvested dividends saw; the DAX already includes dividends. Currency moves are ignored. Computed in international.py.")


if __name__ == "__main__":
    print(text(section()))
