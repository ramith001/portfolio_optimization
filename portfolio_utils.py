"""
Shared Markowitz/GARCH-covariance functions, factored out of
portfolio_optimization.ipynb so later parts (extending history, more assets,
alternative risk models, etc.) reuse the exact same, already-validated logic
instead of re-implementing it per notebook.

Every function body here is copied verbatim from the committed notebook --
nothing about the math changed, only where it lives. The notebook itself is
left untouched; it still works standalone exactly as before, so the
original, already-reproduced headline numbers (Max Sharpe vol 19.97% sample
vs. 14.20% GARCH-informed; 9 rolling windows; 28.2% relative improvement
excluding the COVID window) are not at risk from this refactor.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def portfolio_variance(weights, cov):
    return weights @ cov @ weights


def portfolio_return(weights, mean_returns):
    return weights @ mean_returns


def min_variance_portfolio(mean_returns, cov, n_assets, max_weight=0.35):
    """max_weight caps any single asset's position, preventing the optimizer from
    concentrating the whole portfolio into one or two names."""
    bounds = [(0, max_weight)] * n_assets
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    result = minimize(portfolio_variance, np.ones(n_assets) / n_assets, args=(cov.values,),
                       method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x


def max_sharpe_portfolio(mean_returns, cov, n_assets, rf=0.02, max_weight=0.35):
    bounds = [(0, max_weight)] * n_assets
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

    def neg_sharpe(w):
        ret = portfolio_return(w, mean_returns.values)
        vol = np.sqrt(portfolio_variance(w, cov.values))
        return -(ret - rf) / vol

    result = minimize(neg_sharpe, np.ones(n_assets) / n_assets,
                       method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x


def build_efficient_frontier(mean_returns, cov, n_assets, n_points=40, max_weight=0.35):
    bounds = [(0, max_weight)] * n_assets
    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), n_points)
    frontier_vols = []
    for target in target_returns:
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w, t=target: w @ mean_returns.values - t}
        ]
        res = minimize(portfolio_variance, np.ones(n_assets) / n_assets, args=(cov.values,),
                        method='SLSQP', bounds=bounds, constraints=constraints)
        frontier_vols.append(np.sqrt(res.fun) if res.success else np.nan)
    return target_returns, np.array(frontier_vols)


def garch_neg_log_likelihood(params, returns):
    omega, alpha, beta = params
    n = len(returns)
    sigma2 = np.zeros(n)
    sigma2[0] = np.var(returns)
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
    sigma2 = np.maximum(sigma2, 1e-8)
    ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + returns ** 2 / sigma2)
    return -ll


def fit_garch(returns):
    bounds = [(1e-6, None), (0, 1), (0, 1)]
    constraints = [{'type': 'ineq', 'fun': lambda p: 0.999 - p[1] - p[2]}]
    result = minimize(garch_neg_log_likelihood, [0.1, 0.1, 0.8], args=(returns,),
                       method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 300})
    return result.x


def current_garch_volatility(returns_series, annualize=252):
    """Fit GARCH(1,1) on full history, return the current (most recent) conditional volatility, annualized."""
    r = returns_series.values * 100
    omega, alpha, beta = fit_garch(r)
    n = len(r)
    sigma2 = np.zeros(n)
    sigma2[0] = np.var(r)
    for t in range(1, n):
        sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]
    current_var = sigma2[-1] / (100 ** 2)  # rescale back from percent
    return np.sqrt(current_var * annualize)


def build_garch_covariance(returns, tickers):
    """GARCH volatilities on the diagonal direction, historical sample
    correlations for cross-asset structure -- same CCC-GARCH-style
    construction as the notebook's full-sample and rolling-window cells."""
    garch_vols = {t: current_garch_volatility(returns[t]) for t in tickers}
    garch_vols_series = pd.Series(garch_vols)
    corr = returns.corr()
    garch_vol_outer = np.outer(garch_vols_series[tickers], garch_vols_series[tickers])
    cov_garch = pd.DataFrame(garch_vol_outer * corr.values, index=tickers, columns=tickers)
    return cov_garch, garch_vols_series


def get_rolling_windows(dates, train_years=5, test_months=6):
    """Generate non-overlapping rolling train/test windows across the full date range."""
    train_window_days = int(train_years * 252)
    test_window_days = int(test_months * 21)
    n = len(dates)
    windows = []
    start = train_window_days
    while start + test_window_days <= n:
        windows.append((start - train_window_days, start, start + test_window_days))
        start += test_window_days
    return windows


def run_rolling_validation(log_returns, tickers, n_assets, annualize=252, train_years=5, test_months=6, verbose=True):
    """
    Orchestrates the notebook's section-4 rolling-window validation (same
    window generation, same per-window sample-vs-GARCH comparison) as a
    reusable function instead of a copy-pasted cell, so Part 1 (extended
    date range -> more windows) and later parts (larger universe, DCC, etc.)
    can call it unchanged.
    """
    windows = get_rolling_windows(log_returns.index, train_years=train_years, test_months=test_months)
    if verbose:
        print(f'Running {len(windows)} rolling out-of-sample windows ({train_years}yr train, {test_months}mo test each)...\n')

    results = []
    for w_idx, (train_start, train_end, test_end) in enumerate(windows):
        train_returns = log_returns.iloc[train_start:train_end]
        test_returns = log_returns.iloc[train_end:test_end]

        mean_returns_train = train_returns.mean() * annualize
        cov_sample_train = train_returns.cov() * annualize

        cov_garch_train, _ = build_garch_covariance(train_returns, tickers)

        w_sample = max_sharpe_portfolio(mean_returns_train, cov_sample_train, n_assets)
        w_garch = max_sharpe_portfolio(mean_returns_train, cov_garch_train, n_assets)

        pred_vol_sample = np.sqrt(portfolio_variance(w_sample, cov_sample_train.values))
        pred_vol_garch = np.sqrt(portfolio_variance(w_garch, cov_garch_train.values))

        realized_vol_sample = (test_returns @ w_sample).std() * np.sqrt(annualize)
        realized_vol_garch = (test_returns @ w_garch).std() * np.sqrt(annualize)

        row = {
            'window': w_idx + 1,
            'test_start': log_returns.index[train_end].date(),
            'sample_error': abs(pred_vol_sample - realized_vol_sample),
            'garch_error': abs(pred_vol_garch - realized_vol_garch),
        }
        results.append(row)
        if verbose:
            print(f'  Window {w_idx+1}/{len(windows)} (test starting {row["test_start"]}): '
                  f'sample_err={row["sample_error"]*100:.2f}%, garch_err={row["garch_error"]*100:.2f}%')

    results_df = pd.DataFrame(results)
    return windows, results_df
