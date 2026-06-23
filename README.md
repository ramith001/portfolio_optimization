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

**Correlation is assumed stable; only volatility is treated as time-varying.** The GARCH-informed covariance matrix updates each asset's individual volatility dynamically but reuses the static historical correlation matrix. A fuller dynamic conditional correlation model (DCC-GARCH) would let correlations shift too, a natural next step.

**Long-only, no transaction costs.** Portfolios are rebalanced fresh at each window with no penalty for turnover between periods. A real implementation would need to weigh the benefit of a more responsive covariance estimate against the trading costs of acting on it more frequently.

**Eight assets, one decade.** A larger universe and a longer or higher-frequency dataset would give the rolling validation more windows and more statistical power, particularly since excluding even one window (the COVID crash) noticeably changes the picture, a sign the sample is still relatively small.

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
