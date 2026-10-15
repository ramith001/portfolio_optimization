# Portfolio Optimization: Markowitz Theory with GARCH-Informed Risk
### Built on Real Market Data
**Ramith Vaid | NTU Singapore | Data Science & Artificial Intelligence**

---

## Overview

Standard Markowitz portfolio theory builds an efficient frontier from a single, fixed historical covariance matrix. This project starts from the same textbook foundation, then asks a question the textbook version usually skips: where does the risk estimate driving the whole optimization actually come from, and does a more responsive risk estimate lead to a meaningfully better portfolio?

A baseline efficient frontier is built from a standard sample covariance matrix. A second covariance matrix is then constructed using GARCH(1,1) conditional volatility forecasts, reusing the from-scratch implementation built for the companion volatility forecasting project, combined with historical correlations, an approach related to the Constant Conditional Correlation GARCH framework (Bollerslev, 1990). The two resulting portfolios are compared directly, including a proper out-of-sample test of which risk estimate was actually more accurate.

**Asset universe:** 8 large-cap US stocks across financials, technology, energy, healthcare, consumer staples, industrials, utilities and consumer discretionary, chosen for sector diversification.

---

## Two Issues Caught and Fixed Before Trusting the Result

**Portfolio concentration.** An early version of this comparison let the optimizer allocate freely, and the GARCH-informed Max Sharpe portfolio concentrated 74% of its weight into a single stock. A result built on one concentrated bet doesn't actually test whether the covariance methodology is sound, it mostly tests whether that one stock happened to perform well. Fixed by capping any single position at 35% across every optimization, sample covariance and GARCH-informed alike.

**A single out-of-sample window is not enough to trust.** The first version of the validation tested both covariance approaches on one train/test split. Fixed by rolling the split forward across 9 separate, non-overlapping six-month windows spanning the full decade of data, and comparing average prediction error across all of them rather than relying on one snapshot.

---

## A Genuine Outlier, Identified and Handled Honestly

Running the validation across 9 windows surfaced something worth being upfront about rather than averaging away. One window, the period starting January 2020, showed prediction errors above 29% for both models, while every other window sat between roughly 1% and 12%. That window's test period runs directly into the February-March 2020 COVID crash, an extreme, genuinely unprecedented volatility event with no precedent in either model's training data.

Leaving that window in the headline number answers the wrong question, it mostly measures how badly both approaches handled an unforecastable shock, not which covariance method is more accurate under ordinary conditions. The notebook identifies this window explicitly (rather than silently dropping any inconvenient result) and reports both numbers: the full 9-window average, and the 8-window average with the crisis window excluded and clearly labelled as such.

---

## Pipeline

```
Yahoo Finance prices  --->  log returns  --->  Sample Covariance  --->  Efficient Frontier
                                                                              |
                                                                              v
                                              GARCH(1,1) per asset (from-scratch, reused    Max Sharpe /
                                              from the volatility forecasting project)  --->  Min Variance
                                                          |                                  Portfolios
                                                          v
                                              GARCH-Informed Covariance
                                              (GARCH volatility x historical correlation)
                                                          |
                                                          v
                                              Rolling 9-window out-of-sample validation
                                              (with the COVID-crash window identified and isolated)
```

---

## Key Findings

**Full-sample Max Sharpe portfolios differ meaningfully.** Using the full 10-year sample, the GARCH-informed covariance produces a Max Sharpe portfolio with an estimated volatility of 14.20%, versus 19.97% for the sample-covariance version, with noticeably different weights, GARCH shifts meaningful weight toward JPM and away from CAT and NEE, reflecting that JPM's current conditional volatility sits well below its 10-year historical average while several other names don't show the same gap.

**GARCH provides a genuinely better risk estimate under ordinary market conditions.** Excluding the one unforecastable crisis window, GARCH-informed covariance produced a 28.2% lower average prediction error than sample covariance (3.52% vs 4.90%), and was the more accurate approach in 5 of the 8 ordinary-condition windows.

**Neither approach handles a genuine tail event well, and that's an honest limitation of both, not a flaw specific to either.** Both models posted their worst error by a wide margin in the window whose test period included the COVID crash. This is reported explicitly rather than smoothed over.

---

## Known Limitations

**Correlation is static in the main analysis; Part 2 tests a dynamic alternative.** The GARCH-informed covariance matrix in the main analysis and Part 1 updates each asset's individual volatility dynamically but reuses the static historical correlation matrix. Part 2 adds a simple DCC-GARCH correlation estimate and compares the two (see below). The DCC used is the basic two-parameter version with a two-step estimator, shared by every pair of assets; richer variants (asset-specific dynamics, asymmetric terms, joint estimation) were not tried.

**Long-only, no transaction costs.** Portfolios are rebalanced fresh at each window with no penalty for turnover between periods. A real implementation would need to weigh the benefit of a more responsive covariance estimate against the trading costs of acting on it more frequently.

**Eight assets, one decade.** A larger universe and a longer or higher-frequency dataset would give the rolling validation more windows and more statistical power, particularly since excluding even one window (the COVID crash) noticeably changes the picture, a sign the sample is still relatively small.

---

## Part 1: Extended Date Range

The original run covered 2014 to 2024 across the same 8-asset universe. This extends it through October 2026 (3,214 trading days), giving 15 non-overlapping 6-month rolling windows instead of 9. The first 9 windows reproduce the original per-window errors exactly, confirming the refactor into `portfolio_utils.py` changed nothing about the underlying calculations.

The COVID window (test starting January 2020) is still the clear outlier, at 29.5% sample-covariance error -- nothing in the newly added period comes close; the next largest error anywhere in the sample is 10.0%.

Excluding COVID, GARCH-informed covariance keeps its edge over the extended sample, though the margin narrows slightly:

| | Original (8 windows) | Extended (14 windows) |
|---|---|---|
| Sample covariance, avg error | 4.90% | 5.78% |
| GARCH-informed, avg error | 3.52% | 4.32% |
| GARCH relative improvement | 28.2% | 25.4% |
| Windows won by GARCH | 5 of 8 | 10 of 14 |

Not every new window favors GARCH: the window testing from January 2025 is a clear loss for the GARCH-informed covariance (12.25% error vs. 5.44% for sample covariance) -- a reminder that the GARCH approach wins on average across windows, not in every one.

Full-sample Max Sharpe volatility is largely unchanged: 18.42% with sample covariance (was 19.97%) versus 14.48% GARCH-informed (was 14.20%).

---

## Part 2: Dynamic Correlation (DCC-GARCH) vs. Static Correlation

Part 1's covariance keeps each asset's GARCH volatility current but holds the correlation matrix fixed at its historical value. This adds a simple DCC-GARCH estimate (Engle 2002, two shared parameters `a` and `b`, estimated by the standard two-step method) so correlations can move too, and compares the two head-to-head. The implementation is in `portfolio_utils.py`; `part2_dcc_correlation.ipynb` runs the comparison on the same 8 assets, same 15 rolling 6-month windows and the same Part 1 data (end date pinned to 2026-10-13 so it reproduces).

The comparison is deliberately a one-variable change: the DCC covariance uses *exactly the same* per-asset GARCH volatilities as Part 1's "GARCH-informed" covariance (the notebook checks this), and only the correlation matrix differs, DCC forecast instead of static historical.

**Correlations do move, and persistently.** The full-sample fit gives `a` = 0.0095 and `b` = 0.9815 (`a + b` = 0.991): shocks to correlation are small but very long-lived. The average pairwise correlation across the 8 stocks ranges from 0.14 (August 2026) to 0.46 (April 2020, the COVID crash) against a single static value of 0.32. Today's correlation is well below the static average, so the DCC forecast for the next six months (0.20) is much lower than the static 0.32.

**DCC forecasts realized correlations better.** The cleanest test doesn't involve any portfolio: how far is each method's correlation matrix from the correlations actually realized over the next 6 months (mean absolute error across the 28 pairwise correlations)?

| | Static historical | DCC forecast |
|---|---|---|
| Mean correlation error, 15 windows | 0.208 | 0.168 |
| Windows closer to realized | 3 of 15 | 12 of 15 |

The improvement is consistent (paired t-statistic 3.1 over 15 windows) but it is not universal: DCC does worse in three windows, including the COVID window (Jan 2020 test start), where correlations spiked within the test period itself and neither method could have anticipated it.

**The portfolio-volatility effect is smaller and not statistically clear.** Using Part 1's metric, the average gap between a Max Sharpe portfolio's predicted and realized volatility:

| | All 15 windows | Excluding COVID window (14) |
|---|---|---|
| Sample covariance | 7.36% | 5.78% |
| GARCH + static correlation (Part 1) | 6.23% | 4.32% |
| GARCH + DCC correlation (Part 2) | 5.91% | 3.92% |

DCC has the lowest average error, and beats the static-correlation version in 10 of 15 windows (10 of 14 excluding COVID). But the improvement is about 0.3-0.4 percentage points against window-to-window standard deviations of about 4 to 9 points, and the paired t-statistics are only 1.1 (all windows) and 1.3 (excluding COVID), so with 15 windows this should be read as "consistent with a modest gain", not as proven. Excluding COVID, GARCH + DCC is 32% lower in error than sample covariance (3.92% vs. 5.78%), compared with 25% for the static-correlation version (4.32%).

There are five windows where GARCH with static correlation does worse than plain sample covariance. In four of them (Jan 2019, Jan 2020, Jan 2022, Jan 2025; GARCH errors of 7-33%) DCC doesn't help either, and both GARCH variants miss by about the same amount. That points to the volatility level, not the correlations, as the dominant error there, so the correlation model can't fix it. In the fifth (Jan 2021) DCC does fix it (3.1% down to 1.3%).

On the full sample, DCC changes the Max Sharpe weights modestly (JPM 35% to 29%, JNJ 4% to 7%, CAT 0% to 4%) and lowers the predicted volatility from 14.48% to 13.18%. That drop is a forecast of a lower-correlation regime, not evidence the portfolio would have been safer.

![DCC average pairwise correlation over time, and correlation forecast error by window](chart_part2_dcc.png)

A note on a bug caught along the way: the first version of the DCC optimizer (SLSQP with a stationarity constraint) wandered outside the constraint in one window (`a + b` = 1.009), where the correlation process explodes and the likelihood becomes spuriously low, and that window then looked artificially good. The fit now uses a parametrization that guarantees `a + b < 0.999`, and the notebook asserts this holds in every window (the largest is 0.988).

---

## Files

- `portfolio_optimization.ipynb` -- the original 2014-2024 notebook, left untouched.
- `portfolio_utils.py` -- the Markowitz and GARCH-covariance functions factored out of the original notebook (bodies unchanged), plus `run_rolling_validation()` to run the rolling-window comparison on any date range.
- `part1_extended_period.ipynb` -- Part 1: the same analysis through October 2026, using `portfolio_utils.py`.
- `chart_extended_rolling_windows.png` -- per-window prediction error, sample vs. GARCH-informed covariance, extended period.
- `part2_dcc_correlation.ipynb` -- Part 2: DCC vs. static correlation head-to-head (the DCC functions were added to `portfolio_utils.py`).
- `chart_part2_dcc.png` -- DCC average correlation over time, and correlation forecast error by window.
- `part2_rolling_dcc_comparison.csv` -- the per-window numbers behind the Part 2 tables.

---

## Stack

- **Language:** Python 3.13
- **Libraries:** pandas, numpy, scipy, yfinance, matplotlib
- **Data source:** Yahoo Finance (free, no API key required)
- **GARCH implementation:** custom, via scipy.optimize, shared with the companion volatility forecasting project

---

## Relation to the Volatility Forecasting Project

This project reuses the exact GARCH(1,1) implementation built and validated in the companion [volatility forecasting project](https://github.com/ramith001/volatility-forecasting), applied here as the risk input to an actual portfolio construction problem rather than evaluated on its own. The two together form a complete story: first build and validate a model of risk, then use that risk model to construct something real, an optimal portfolio.

---

*Built by Ramith Vaid | NTU Singapore | ramithvaid@gmail.com*
