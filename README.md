# stocks

Back-tests of buying rules on US stock-market history: how dollar-cost averaging (DCA) and its variants held up
through the Great Depression, the dot-com crash and everything since 1928. Each analysis is a small Python script
that prints its results; `export_animation.py` gathers them into one static page with an animated replay.

## Quick start

Needs [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just).

```sh
just fetch   # download the data into data/; later runs add only new rows
just serve   # build site/ and serve it at http://127.0.0.1:8000
```

`just` alone only builds `site/`. The page loads its replay files over HTTP, so open it through `just serve`
rather than double-clicking `site/index.html`. A build takes a few minutes because every analysis reruns.

Any script also runs on its own and prints its tables, e.g. `uv run python pause_check.py`.

## Data

`fetch_data.py` writes everything to `data/` (git-ignored):

- **Yahoo Finance, daily:** S&P 500 (`^GSPC`, from 1927-12-30), S&P 500 total return (`^SP500TR`), NASDAQ,
  Dow, Russell 2000, FTSE 100, DAX and Nikkei 225, plus fixed windows around the 1929 and 2000 crashes.
- **Robert Shiller's `ie_data.xls`, monthly:** prices, dividends, consumer prices and CAPE from 1871.
- **FRED, monthly:** consumer prices (`CPIAUCNS`), 3-month bills (`TB3MS`), 10-year Treasuries (`GS10`), and the
  NBER short rate (`M1329AUSM193NNBR`) to cover bills before 1934.

A fetch downloads only what is missing: absent files in full, and for the open-ended series (the full-history
indices and the growing FRED series) the rows after the last stored date, up to yesterday. The fixed crash windows,
Shiller's file (it ends 2023-09) and the NBER rate (ends 1934) are never re-downloaded. Revisions Yahoo or FRED make
to rows already stored are not picked up; delete a file to download it again in full. Yahoo's Dow starts in 1992,
so the 1929 analyses use the S&P 500.

## Scripts

Shared code:

| File | Role |
|---|---|
| `fetch_data.py` | Downloads the data above |
| `monthly.py` | Monthly table from 1928 to the last month every source has published: total return, price, consumer prices, bills, modelled 10-year bond return, CAPE; also a daily total-return series |
| `report.py` | The shape of one page section (title, summary, tables), printed in the terminal or drawn on the page |
| `export_animation.py` | Runs the analyses and renders `animation_template.html` into `site/index.html` plus `site/anim/*.json` replays |

Buying rules on the two crashes and the full history:

| File | Question |
|---|---|
| `worst_days.py` | Worst one-day drops per index |
| `dca.py` | Buy each time the price drops 10% (three readings of that rule) vs lump sum and monthly DCA |
| `month_dip.py` | Each month, buy on the first 10% (or 4%) dip, or on the last day if none |
| `bear_pause.py` | At 20% below the record, pause buying until 31 months after the peak, then buy the skipped months at once |
| `pause_check.py` | The pause rule over the full history since 1928, for several pause lengths and start dates |
| `crashes.py` | How long falls of 20% or more lasted |
| `buy_day.py`, `buy_day_verify.py` | Which date of the month is cheapest to buy on, and six cross-checks |
| `date_rank.py` | Monthly DCA on each date of the month, ranked over the last 10, 15 and 20 years |
| `lump_vs_spread.py` | Invest a lump sum at once or spread it over extra monthly buys |

Wider questions (each also a page section):

| File | Question |
|---|---|
| `four_percent.py` | When is a guaranteed 4% a year worth more than stocks, and which signs said so |
| `trend_rule.py` | Does leaving stocks below the 200-day average dodge crashes |
| `international.py` | Crashes and DCA in Japan, Germany and the UK |
| `real_returns.py` | Recovery times with dividends and after inflation |
| `cash_interest.py` | Lump vs spread and the pause rule when waiting cash earns bill interest |
| `stock_bond_mix.py` | How much a bond share softens crashes |
| `fees.py` | What an annual fee costs over decades of DCA |
| `withdrawals.py` | Does the 4% withdrawal rule last 30 years |

## Assumptions

- Buys happen at the day's close (daily data) or the month-average level (monthly data); no fractional-share
  limits.
- Trading costs and taxes are ignored unless a script says otherwise.
- Price-only results leave out dividends; the scripts that include them say so.
- Several scripts check their own logic with `assert`s at import time, so a broken rule fails loudly.

Nothing here is investment advice: it describes what happened in one country's history, not what will happen.
