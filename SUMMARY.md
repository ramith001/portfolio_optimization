# Summary: GARCH-Informed Risk in Portfolio Optimization

**Ramith Vaid | NTU Singapore**

This project asks whether a more responsive risk estimate, volatilities from a GARCH(1,1) model and later correlations from a DCC-GARCH model, makes a better portfolio than the sample covariance matrix a textbook Markowitz optimizer uses. It answers with 15 rolling out-of-sample windows (five years of training, six months of testing) and then stress-tests the answer: transaction costs, a wider asset universe, a time-varying risk-free rate, benchmarks, a CVaR objective and different window lengths. The detail is in `README.md`; `chart_summary.png` shows one result from each part.

## What holds up

- **GARCH covariances predict portfolio volatility better.** Excluding the COVID window, the Max Sharpe portfolio's predicted volatility is off by 5.78% (sample covariance), 4.32% (GARCH, static correlation) and 3.92% (GARCH, DCC correlation) on 8 stocks, and by 3.93%, 2.75% and 2.74% on 13 assets. The advantage is there with 3- and 6-month test windows and with 3, 5 and 7 years of training, and it disappears at 12 months. It is consistent across settings but rarely statistically clear with so few windows; the strongest case is DCC with 3-month windows (22 of 30 windows, t = 2.19).
- **Trading costs are small.** At 10 bps they take about 0.1 percentage points a year from these portfolios.
- **The risk-free rate matters for measurement.** The T-bill compounded to 2.74% a year over the test windows, so Sharpe ratios against the 2% first assumed were too high by 0.04 to 0.05.
- **Risk-minimizing objectives do cut the daily tail.** Minimum-variance and minimum-CVaR portfolios have a realized daily CVaR of about 1.4% against 2.1% on 13 assets.

## What did not hold up

- **A better risk estimate did not make a better portfolio.** The Max Sharpe portfolios have a Sharpe ratio of 0.78 to 0.86 against 1.06 for equal weight of the same stocks and 0.76 for the S&P 500. Only equal weight's lead over the index is clear in a block bootstrap (96% of resamples).
- **The small Sharpe edge for GARCH is fragile.** It appears with 8 stocks and five years of training, reverses with three years (sample covariance 0.82, GARCH 0.67 and 0.62) and is absent with 13 assets.
- **For single assets GARCH is not a better volatility forecast** (a mean error of 4.93% against 4.94% for the sample volatility excluding COVID, across 182 asset-windows). It helps for long Treasuries and developed international equity and hurts for US large-cap stocks.
- **The CVaR objective did not beat variance.** Minimum CVaR and minimum variance have about the same realized tail loss and Sharpe, and they do not reliably reduce the maximum drawdown (it is deeper than for Max Sharpe on 8 stocks and shallower on 13 assets). Maximum return-per-CVaR on GARCH-filtered scenarios has the best Sharpe among the optimized portfolios (0.97 on 8 stocks), but on thin evidence.

## A practical reading

The covariance estimate is a reasonable way to describe and predict portfolio risk, and GARCH volatilities improve it for a few months at a time. It is not a reason to expect better returns: the Max Sharpe portfolios sit at the 35% position cap in 12 to 14 of 15 windows, which suggests the poorly estimated mean returns drive them at least as much as the covariance does (this was not tested directly), and an equal-weight portfolio matched or beat every optimized one at every setting tried. Filtered, GARCH-scaled scenarios predicted the next six months' tail loss better than plain historical ones for portfolios not optimized on them, which is the clearest use of the GARCH idea here.

## Limits

Fifteen windows in one period, with universes of today's large caps and ETFs chosen with hindsight, in a stretch when US large caps did well. That helps equal weight and the index as much as it helps the optimized portfolios. The GARCH models are GARCH(1,1) with normal innovations and the DCC is the basic two-parameter version; costs are a flat 10 bps. Most of the differences between strategies are within what a few windows could change.
