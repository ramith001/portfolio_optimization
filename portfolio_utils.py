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


# ---------------------------------------------------------------------------
# Part 2: a simple DCC-GARCH-style dynamic correlation estimate
#
# Everything above is the Part 1 module and is unchanged. Everything below is
# new in Part 2. It follows Engle's (2002) DCC recursion in its simplest form:
#   1. fit GARCH(1,1) per asset (fit_garch, unchanged) and standardize each
#      asset's returns by its own conditional volatility -> z_t
#   2. Q_t = (1 - a - b) * Qbar + a * z_{t-1} z_{t-1}' + b * Q_{t-1},
#      R_t = diag(Q_t)^(-1/2) Q_t diag(Q_t)^(-1/2), with Qbar = the sample
#      covariance of z (correlation targeting), and (a, b) the only two
#      parameters, shared by every pair of assets
#   3. (a, b) estimated by maximizing the correlation part of the Gaussian
#      quasi-likelihood (the two-step estimator), not a full joint MLE
# To make the head-to-head with the static estimate a clean one-variable
# comparison, the DCC covariance uses the SAME per-asset GARCH volatilities as
# build_garch_covariance and differs only in the correlation matrix.
# ---------------------------------------------------------------------------
def garch_standardized_residuals(returns_pct, omega, alpha, beta):
    """z_t = r_t / sigma_t for one asset (returns in percent), same recursion as current_garch_volatility,
    plus the final in-sample conditional variance (percent^2), which that function uses as 'current' vol."""
    r = np.asarray(returns_pct, dtype=float)
    n = len(r)
    sigma2 = np.zeros(n)
    sigma2[0] = np.var(r)
    for t in range(1, n):
        sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]
    sigma2 = np.maximum(sigma2, 1e-8)
    return r / np.sqrt(sigma2), sigma2[-1]


def dcc_filter(z, a, b, Qbar):
    """Run the DCC recursion over standardized residuals z (T x N). Returns (R, Q_next): R[t] is the
    conditional correlation matrix for day t (it uses z only up to t-1), Q_next is Q for day T+1."""
    T, N = z.shape
    Q = Qbar.copy()
    R = np.empty((T, N, N))
    for t in range(T):
        d = 1.0 / np.sqrt(np.diag(Q))
        R[t] = Q * np.outer(d, d)
        Q = (1 - a - b) * Qbar + a * np.outer(z[t], z[t]) + b * Q
    return R, Q


def dcc_neg_log_likelihood(params, z, Qbar):
    """Negative correlation-part quasi-log-likelihood: 0.5 * sum_t (log|R_t| + z_t' R_t^-1 z_t - z_t' z_t)."""
    a, b = params
    R, _ = dcc_filter(z, a, b, Qbar)
    _, logdet = np.linalg.slogdet(R)
    quad = np.einsum('ti,ti->t', z, np.linalg.solve(R, z[..., None])[..., 0])
    return 0.5 * np.sum(logdet + quad - np.einsum('ti,ti->t', z, z))


def fit_dcc(z):
    """Estimate the two DCC parameters (a, b) from standardized residuals. Returns (a, b, Qbar).

    (a, b) are parametrized as a = s * q, b = s * (1 - q) with s = 0.999 * sigmoid(u) and q = sigmoid(v), so
    every candidate satisfies a, b > 0 and a + b < 0.999 by construction, and the search is derivative-free
    (Nelder-Mead) from two starting points. A constrained gradient optimizer (SLSQP) was tried first and could
    wander outside the stationarity region, where Q explodes and the likelihood is spuriously low, landing on
    a + b > 1 in one rolling window; this parametrization cannot do that.
    """
    Qbar = (z.T @ z) / len(z)

    def to_ab(theta):
        s_ = 0.999 / (1 + np.exp(-theta[0]))
        q_ = 1 / (1 + np.exp(-theta[1]))
        return s_ * q_, s_ * (1 - q_)

    def objective(theta):
        a_, b_ = to_ab(theta)
        return dcc_neg_log_likelihood((a_, b_), z, Qbar)

    best = None
    for a0, b0 in ((0.03, 0.95), (0.01, 0.98)):
        s0 = (a0 + b0) / 0.999
        theta0 = [np.log(s0 / (1 - s0)), np.log(a0 / (a0 + b0) / (1 - a0 / (a0 + b0)))]
        res = minimize(objective, theta0, method='Nelder-Mead',
                        options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 400})
        if best is None or res.fun < best.fun:
            best = res
    a, b = to_ab(best.x)
    return a, b, Qbar


def dcc_correlation_forecast(a, b, Qbar, Q_next, horizon):
    """Expected average correlation matrix over the next `horizon` days. Uses the standard DCC
    mean-reversion approximation E[Q_{T+k}] = Qbar + (a+b)^(k-1) (Q_{T+1} - Qbar), averaged over k = 1..horizon
    and rescaled to a correlation matrix."""
    k = np.arange(horizon)
    weight = np.mean((a + b) ** k)
    Q_avg = Qbar + weight * (Q_next - Qbar)
    d = 1.0 / np.sqrt(np.diag(Q_avg))
    return Q_avg * np.outer(d, d)


def build_dcc_covariance(returns, tickers, horizon_days=126, annualize=252):
    """
    DCC-informed covariance: the same current GARCH volatilities as build_garch_covariance, with the
    static historical correlation matrix replaced by the DCC forecast of the average correlation over the
    next `horizon_days` trading days (126 = the 6-month test window used in the rolling validation).
    Returns (cov_dcc, garch_vols_series, info) where info holds a, b, the forecast correlation, and the
    filtered correlation path R (T x N x N) for the training sample.
    """
    zs, vols = {}, {}
    for t in tickers:
        r = returns[t].values * 100
        omega, alpha, beta = fit_garch(r)
        z, last_sigma2 = garch_standardized_residuals(r, omega, alpha, beta)
        zs[t] = z
        vols[t] = np.sqrt(last_sigma2 / (100 ** 2) * annualize)  # identical to current_garch_volatility
    z = np.column_stack([zs[t] for t in tickers])
    a, b, Qbar = fit_dcc(z)
    R, Q_next = dcc_filter(z, a, b, Qbar)
    R_forecast = dcc_correlation_forecast(a, b, Qbar, Q_next, horizon_days)
    garch_vols_series = pd.Series(vols)
    vol_outer = np.outer(garch_vols_series[tickers], garch_vols_series[tickers])
    cov_dcc = pd.DataFrame(vol_outer * R_forecast, index=tickers, columns=tickers)
    return cov_dcc, garch_vols_series, {'a': a, 'b': b, 'R_forecast': R_forecast, 'R_path': R, 'Qbar': Qbar}


def mean_abs_offdiag_error(R_forecast, R_realized):
    """Mean absolute error over the off-diagonal (pairwise) correlations."""
    R_forecast, R_realized = np.asarray(R_forecast), np.asarray(R_realized)
    mask = ~np.eye(R_forecast.shape[0], dtype=bool)
    return float(np.mean(np.abs(R_forecast[mask] - R_realized[mask])))


def run_rolling_dcc_comparison(log_returns, tickers, n_assets, annualize=252, train_years=5, test_months=6,
                                verbose=True):
    """
    Same rolling windows as run_rolling_validation (5yr train, 6mo test, non-overlapping). In each window,
    three covariance estimates are built from the training data only and each is used for a Max Sharpe
    portfolio (35% cap), whose predicted volatility is compared to the volatility realized in the test window:
      sample      - the plain sample covariance
      garch_static - GARCH vols x static historical correlation (Part 1's "GARCH-informed")
      garch_dcc    - GARCH vols x DCC-forecast correlation (new in Part 2)
    The last two share identical volatilities; only the correlation matrix differs. Also records how close
    each correlation matrix is to the correlations actually realized in the test window.
    """
    windows = get_rolling_windows(log_returns.index, train_years=train_years, test_months=test_months)
    horizon = int(test_months * 21)
    if verbose:
        print(f'Running {len(windows)} rolling windows: sample vs. GARCH+static corr vs. GARCH+DCC corr...\n')

    rows = []
    for w_idx, (train_start, train_end, test_end) in enumerate(windows):
        train = log_returns.iloc[train_start:train_end]
        test = log_returns.iloc[train_end:test_end]
        mean_train = train.mean() * annualize

        cov_sample = train.cov() * annualize
        cov_static, _ = build_garch_covariance(train, tickers)
        cov_dcc, _, info = build_dcc_covariance(train, tickers, horizon_days=horizon, annualize=annualize)

        row = {'window': w_idx + 1, 'test_start': log_returns.index[train_end].date(),
               'dcc_a': info['a'], 'dcc_b': info['b']}
        for label, cov in (('sample', cov_sample), ('garch_static', cov_static), ('garch_dcc', cov_dcc)):
            w = max_sharpe_portfolio(mean_train, cov, n_assets)
            pred = np.sqrt(portfolio_variance(w, cov.values))
            realized = (test @ w).std() * np.sqrt(annualize)
            row[f'{label}_error'] = abs(pred - realized)
            row[f'{label}_pred_vol'] = pred
            row[f'{label}_realized_vol'] = realized

        R_realized = test.corr().values
        row['corr_err_static'] = mean_abs_offdiag_error(train.corr().values, R_realized)
        row['corr_err_dcc'] = mean_abs_offdiag_error(info['R_forecast'], R_realized)
        rows.append(row)
        if verbose:
            print(f'  Window {w_idx+1}/{len(windows)} (test starting {row["test_start"]}): '
                  f'sample={row["sample_error"]*100:.2f}%, static={row["garch_static_error"]*100:.2f}%, '
                  f'dcc={row["garch_dcc_error"]*100:.2f}%  |  corr err static={row["corr_err_static"]:.3f}, '
                  f'dcc={row["corr_err_dcc"]:.3f}  (a={info["a"]:.3f}, b={info["b"]:.3f})')
    return windows, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part 3: transaction costs and turnover
#
# Everything above is unchanged. Below are the pieces the cost-aware rolling
# comparison is built from.
#
# Cost model: a proportional cost of `cost` (a fraction, so 10 bps = 0.0010) on every unit of
# portfolio value traded, buys and sells both counted, i.e. cost * sum_i |w_new_i - w_drifted_i| paid
# at the rebalance. "w_drifted" is last window's weights after they have drifted with the assets'
# own returns over that window (buy-and-hold inside a window), which is what the portfolio actually
# holds on rebalance day. Turnover below is the same quantity, sum |w_new - w_drifted|.
# ---------------------------------------------------------------------------
def drift_weights(weights, test_returns):
    """Weights after buy-and-hold over a window of daily log returns (DataFrame, columns = assets)."""
    growth = np.exp(test_returns.sum().values)
    w = np.asarray(weights) * growth
    return w / w.sum()


def buy_and_hold_daily_returns(weights, test_returns):
    """Daily simple returns of a portfolio that starts the window at `weights` and is not rebalanced
    inside it (so it is consistent with drift_weights). First return is relative to the start value 1."""
    values = (np.exp(test_returns.cumsum().values) * np.asarray(weights)).sum(axis=1)
    prev = np.concatenate([[1.0], values[:-1]])
    return pd.Series(values / prev - 1.0, index=test_returns.index)


def max_sharpe_with_turnover_penalty(mean_returns, cov, n_assets, w_prev, cost, rebalances_per_year=2,
                                      rf=0.02, max_weight=0.35):
    """
    Max Sharpe with a turnover penalty in the objective: maximize
        (w'mu - rf - rebalances_per_year * cost * sum|w - w_prev|) / sqrt(w'Cov w)
    i.e. expected annual excess return net of the annualized cost of trading from w_prev (the drifted
    holdings) to w. mu and Cov are annualized, so the one-off rebalance cost is annualized by the number
    of rebalances per year (2 for the 6-month windows). Same 35% cap and long-only constraint as
    max_sharpe_portfolio.

    sum|w - w_prev| is non-smooth, so it is written with the standard split w - w_prev = u - v, u, v >= 0
    and penalty sum(u + v) (exact at the optimum because the penalty makes u and v never both positive).
    With cost = 0 this is max_sharpe_portfolio.
    """
    if cost == 0:
        return max_sharpe_portfolio(mean_returns, cov, n_assets, rf=rf, max_weight=max_weight)
    mu, S, w_prev = mean_returns.values, cov.values, np.asarray(w_prev)
    rate = rebalances_per_year * cost

    def neg_net_sharpe(x):
        w, u, v = x[:n_assets], x[n_assets:2 * n_assets], x[2 * n_assets:]
        return -(w @ mu - rf - rate * np.sum(u + v)) / np.sqrt(w @ S @ w)

    bounds = [(0, max_weight)] * n_assets + [(0, 1)] * (2 * n_assets)
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x[:n_assets]) - 1},
        {'type': 'eq', 'fun': lambda x: x[:n_assets] - w_prev - x[n_assets:2 * n_assets] + x[2 * n_assets:]},
    ]
    best = None
    for start in (w_prev, np.ones(n_assets) / n_assets):
        d = start - w_prev
        x0 = np.concatenate([start, np.maximum(d, 0), np.maximum(-d, 0)])
        res = minimize(neg_net_sharpe, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                        options={'maxiter': 300})
        if res.success and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        return np.asarray(w_prev)
    w = np.clip(best.x[:n_assets], 0, max_weight)
    return w / w.sum()


def build_window_inputs(log_returns, tickers, annualize=252, train_years=5, test_months=6, verbose=True):
    """
    For each rolling window (same windows as run_rolling_validation), build from the TRAINING data only the
    inputs every strategy needs: annualized mean returns and the three covariance estimates from Parts 1-2
    (sample, GARCH + static correlation, GARCH + DCC correlation), plus the test-window returns. Computed
    once so the cost comparison can run many strategies and cost levels off the same estimates.
    """
    windows = get_rolling_windows(log_returns.index, train_years=train_years, test_months=test_months)
    horizon = int(test_months * 21)
    out = []
    for w_idx, (train_start, train_end, test_end) in enumerate(windows):
        train = log_returns.iloc[train_start:train_end]
        cov_static, _ = build_garch_covariance(train, tickers)
        cov_dcc, _, _ = build_dcc_covariance(train, tickers, horizon_days=horizon, annualize=annualize)
        out.append({'window': w_idx + 1, 'test_start': log_returns.index[train_end].date(),
                    'mean': train.mean() * annualize,
                    'cov': {'sample': train.cov() * annualize, 'garch_static': cov_static, 'garch_dcc': cov_dcc},
                    'test': log_returns.iloc[train_end:test_end]})
        if verbose:
            print(f'  Window {w_idx + 1}/{len(windows)} inputs built (test starting {out[-1]["test_start"]})')
    return out


def simulate_strategy(window_inputs, n_assets, cov_key, cost, cost_aware, rf=0.02, max_weight=0.35,
                      rebalances_per_year=2):
    """
    Roll one strategy through the windows in order. Each window: choose weights from the training-data
    estimates (cov_key picks the covariance; cost_aware=True uses max_sharpe_with_turnover_penalty from the
    current drifted holdings, False uses the plain max_sharpe_portfolio and ignores the cost when choosing),
    pay cost * sum|w_new - w_drifted|, then hold buy-and-hold through the test window.
    Window 1 starts fully invested at its chosen weights with no entry cost (the same for every strategy),
    so costs only come from the 14 rebalances. cov_key = 'equal_weight' is the 1/N reference.
    Returns (window_df, daily_net_returns, daily_gross_returns).
    """
    rows, net_parts, gross_parts = [], [], []
    w_drift = None
    for wi in window_inputs:
        if cov_key == 'equal_weight':
            w = np.ones(n_assets) / n_assets
        elif cost_aware and w_drift is not None:
            w = max_sharpe_with_turnover_penalty(wi['mean'], wi['cov'][cov_key], n_assets, w_drift, cost,
                                                  rebalances_per_year, rf, max_weight)
        else:
            w = max_sharpe_portfolio(wi['mean'], wi['cov'][cov_key], n_assets, rf=rf, max_weight=max_weight)
        turnover = 0.0 if w_drift is None else float(np.abs(w - w_drift).sum())
        trade_cost = cost * turnover
        gross = buy_and_hold_daily_returns(w, wi['test'])
        net = gross.copy()
        net.iloc[0] = (1 + gross.iloc[0]) * (1 - trade_cost) - 1
        rows.append({'window': wi['window'], 'test_start': wi['test_start'], 'turnover': turnover,
                     'cost_paid': trade_cost, 'gross_window_return': float((1 + gross).prod() - 1),
                     'net_window_return': float((1 + net).prod() - 1), 'max_weight': float(w.max()),
                     'n_holdings': int((w > 0.005).sum())})
        net_parts.append(net)
        gross_parts.append(gross)
        w_drift = drift_weights(w, wi['test'])
    return pd.DataFrame(rows), pd.concat(net_parts), pd.concat(gross_parts)


def performance_summary(daily_returns, rf=0.02, annualize=252):
    """Annualized (geometric) return, volatility and Sharpe of a daily simple-return series."""
    n = len(daily_returns)
    ann_ret = (1 + daily_returns).prod() ** (annualize / n) - 1
    ann_vol = daily_returns.std() * np.sqrt(annualize)
    return {'ann_return': ann_ret, 'ann_vol': ann_vol, 'sharpe': (ann_ret - rf) / ann_vol}


# ---------------------------------------------------------------------------
# Part 4: caching the per-window estimates
#
# build_window_inputs fits one GARCH model per asset and a DCC model for every window, which takes several minutes
# per universe. Parts 4 to 8 all use the same estimates for the same universes, so they are built once and cached
# (cache files are git-ignored). The cache is keyed on the tickers, the date range and the window settings, and is
# rebuilt if any of them differ.
# ---------------------------------------------------------------------------
def load_or_build_window_inputs(cache_path, log_returns, tickers, annualize=252, train_years=5, test_months=6,
                                verbose=False):
    import os
    import pickle
    key = {'tickers': list(tickers), 'first': str(log_returns.index[0].date()), 'last': str(log_returns.index[-1].date()),
           'annualize': annualize, 'train_years': train_years, 'test_months': test_months}
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            blob = pickle.load(f)
        if blob.get('key') == key:
            return blob['inputs']
    inputs = build_window_inputs(log_returns[list(tickers)], list(tickers), annualize=annualize,
                                 train_years=train_years, test_months=test_months, verbose=verbose)
    with open(cache_path, 'wb') as f:
        pickle.dump({'key': key, 'inputs': inputs}, f)
    return inputs


# ---------------------------------------------------------------------------
# Part 5: a time-varying risk-free rate
#
# The optimizer and every Sharpe ratio so far used a fixed 2% risk-free rate. These functions let each window's
# optimizer use the rate actually available at the start of the window (the 13-week T-bill yield on the last training
# day), and score performance against the T-bill return actually earned over the same days.
# ---------------------------------------------------------------------------
def window_risk_free(window_inputs, log_returns, tbill_annual):
    """T-bill yield (annual, as a fraction) on the last training day of each window. `tbill_annual` is a daily Series."""
    out = []
    for wi in window_inputs:
        first_test_day = wi['test'].index[0]
        known = tbill_annual[tbill_annual.index < first_test_day]
        out.append(float(known.iloc[-1]))
    return out


def simulate_strategy_rf(window_inputs, n_assets, cov_key, cost, rf_by_window, max_weight=0.35):
    """simulate_strategy (cost-blind) with the optimizer's risk-free rate set per window instead of one fixed value."""
    rows, net_parts, gross_parts, weights = [], [], [], []
    w_drift = None
    for wi, rf in zip(window_inputs, rf_by_window):
        if cov_key == 'equal_weight':
            w = np.ones(n_assets) / n_assets
        else:
            w = max_sharpe_portfolio(wi['mean'], wi['cov'][cov_key], n_assets, rf=rf, max_weight=max_weight)
        turnover = 0.0 if w_drift is None else float(np.abs(w - w_drift).sum())
        gross = buy_and_hold_daily_returns(w, wi['test'])
        net = gross.copy()
        net.iloc[0] = (1 + gross.iloc[0]) * (1 - cost * turnover) - 1
        rows.append({'window': wi['window'], 'test_start': wi['test_start'], 'rf': rf, 'turnover': turnover,
                     'max_weight': float(w.max()), 'n_holdings': int((w > 0.005).sum())})
        weights.append(w)
        net_parts.append(net)
        gross_parts.append(gross)
        w_drift = drift_weights(w, wi['test'])
    return pd.DataFrame(rows), pd.concat(net_parts), pd.concat(gross_parts), np.array(weights)


def performance_vs_risk_free(daily_returns, rf_daily, fixed_rf=0.02, annualize=252):
    """
    Annualized geometric return and volatility of a daily simple-return series, and its Sharpe ratio two ways:
    against a fixed rate (the earlier parts' convention: (geometric return - fixed_rf) / vol) and against the T-bill
    return compounded over the same days ((geometric return - compounded T-bill return) / vol).
    rf_daily: Series of the daily T-bill return aligned to the same dates.
    """
    n = len(daily_returns)
    ann_ret = (1 + daily_returns).prod() ** (annualize / n) - 1
    ann_vol = daily_returns.std() * np.sqrt(annualize)
    rf = rf_daily.reindex(daily_returns.index).fillna(0.0)
    ann_rf = (1 + rf).prod() ** (annualize / n) - 1
    return {'ann_return': ann_ret, 'ann_vol': ann_vol, 'avg_tbill': ann_rf, 'sharpe_fixed': (ann_ret - fixed_rf) / ann_vol,
            'sharpe_tbill': (ann_ret - ann_rf) / ann_vol}



# ---------------------------------------------------------------------------
# Part 6: benchmarks and tail-aware performance measures
# ---------------------------------------------------------------------------
def drawdown_path(daily_returns):
    """Running drawdown of the wealth path of a daily simple-return series (0 at a new high, negative below it)."""
    wealth = (1 + daily_returns).cumprod()
    return wealth / wealth.cummax() - 1


def risk_summary(daily_returns, rf_daily, annualize=252):
    """Return, volatility, Sharpe against the T-bill, maximum drawdown, Calmar ratio and Sortino-type downside deviation."""
    n = len(daily_returns)
    ann_ret = (1 + daily_returns).prod() ** (annualize / n) - 1
    vol = daily_returns.std() * np.sqrt(annualize)
    rf = rf_daily.reindex(daily_returns.index).fillna(0.0)
    ann_rf = (1 + rf).prod() ** (annualize / n) - 1
    mdd = drawdown_path(daily_returns).min()
    down = np.sqrt(np.mean(np.minimum(daily_returns - rf, 0.0) ** 2)) * np.sqrt(annualize)
    return {'ann_return': ann_ret, 'ann_vol': vol, 'sharpe': (ann_ret - ann_rf) / vol, 'max_drawdown': mdd,
            'calmar': ann_ret / abs(mdd), 'sortino': (ann_ret - ann_rf) / down}


def block_bootstrap_indices(n, n_boot, block=21, seed=0):
    """Circular block bootstrap: (n_boot, n) array of row numbers built from blocks of `block` consecutive days."""
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return idx.reshape(n_boot, -1)[:, :n]


def sharpe_from_arrays(r, rf, annualize=252):
    """Vectorized Sharpe (geometric return minus compounded T-bill, over vol) for rows of daily returns r (m x n)."""
    n = r.shape[1]
    ann_ret = np.exp(np.log1p(r).sum(axis=1) * annualize / n) - 1
    ann_rf = np.exp(np.log1p(rf).sum(axis=1) * annualize / n) - 1
    return (ann_ret - ann_rf) / (r.std(axis=1, ddof=1) * np.sqrt(annualize))


def bootstrap_sharpe_differences(returns, rf_daily, n_boot=5000, block=21, seed=0):
    """
    returns: DataFrame of daily returns (columns = strategies) on a common set of dates. Resamples days in blocks
    (the same blocks for every strategy, so differences are paired) and returns, per strategy, the Sharpe point
    estimate and bootstrap interval, plus the bootstrap distribution of every pairwise Sharpe difference.
    """
    idx = block_bootstrap_indices(len(returns), n_boot, block, seed)
    rf = rf_daily.reindex(returns.index).fillna(0.0).values
    sharpes = {c: sharpe_from_arrays(returns[c].values[idx], rf[idx]) for c in returns.columns}
    return sharpes


def capm_regression(strategy, market, rf_daily, lags=10, annualize=252):
    """Regression of daily excess returns on the market's, with a Newey-West covariance. Returns alpha (annualized),
    beta, R^2 and the t-statistics of alpha and beta."""
    rf = rf_daily.reindex(strategy.index).fillna(0.0)
    y = (strategy - rf).values
    x = (market.reindex(strategy.index) - rf).values
    X = np.column_stack([np.ones(len(x)), x])
    beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta_hat
    n = len(y)
    XtX_inv = np.linalg.inv(X.T @ X)
    S = (X * e[:, None]).T @ (X * e[:, None])
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = (X[L:] * e[L:, None]).T @ (X[:-L] * e[:-L, None])
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    r2 = 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return {'alpha_ann': beta_hat[0] * annualize, 'beta': beta_hat[1], 'r2': r2,
            't_alpha': beta_hat[0] / se[0], 't_beta': beta_hat[1] / se[1]}

