import contextlib
import itertools
import logging
import multiprocessing
import os
from datetime import datetime
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as optimize
from cvxopt import matrix, solvers
from pandas_datareader.data import DataReader
from scipy.special import betaln
from statsmodels import api as sm
from statsmodels.api import OLS

solvers.options["show_progress"] = False


@contextlib.contextmanager
def mp_pool(n_jobs):
    n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
    if n_jobs == 1:
        # For single-threaded execution, use a dummy context that provides map function
        class DummyPool:
            def map(self, func, iterable):
                return [func(item) for item in iterable]

        yield DummyPool()
    else:
        pool = multiprocessing.Pool(n_jobs)
        try:
            yield pool
        finally:
            pool.close()


def dataset(name):
    """Return sample dataset from /data directory."""
    filename = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "data", name + ".csv"
    )
    return pd.read_csv(filename)


def profile(algo, data=None, to_profile=[]):
    """Profile algorithm using line_profiler.
    :param algo: Algorithm instance.
    :param data: Stock prices, default is random portfolio.
    :param to_profile: List of methods to profile, default is `step` method.

    Example of use:
        tools.profile(Anticor(window=30, c_version=False), to_profile=[Anticor.weights])
    """
    from line_profiler import LineProfiler

    if data is None:
        data = random_portfolio(n=1000, k=10, mu=0.0)

    to_profile = to_profile or [algo.step]
    profile = LineProfiler(*to_profile)
    profile.runcall(algo.run, data)
    profile.print_stats()


def load_ticker(ticker, start=datetime(2000, 1, 1), end=None):
    return DataReader(ticker, "yahoo", start=start, end=None)


def quickrun(algo, data=None, n=1000, **kwargs):
    """Run algorithm and print its running time and some statistics."""
    if data is None:
        data = random_portfolio(n=n, k=3, mu=0.0001)
    t = time()
    result = algo.run(data)
    logging.debug("Time: {:.2f}s".format(time() - t))

    print(result.summary())
    result.plot(**kwargs)
    plt.show()

    return result


def random_portfolio(n, k, mu=0.0, sd=0.01, corr=None, dt=1.0, nan_pct=0.0):
    """Generate asset prices assuming multivariate geometric Brownian motion.

    :param n: Number of time steps.
    :param k: Number of assets.
    :param mu: Drift parameter. Can be scalar or vector. Default is 0.
    :param sd: Volatility of single assets. Default is 0.01.
    :param corr: Correlation matrix of assets. Default is identity.
    :param dt: Time step.
    :param nan_pct: Add given percentage of NaN values. Useful for testing
    """
    # default values
    corr = corr if corr is not None else np.eye(k)
    sd = sd * np.ones(k)
    if isinstance(mu, (int, float)):
        mu = mu * np.ones(k)
    elif mu.ndim == 2:
        mu = mu[1:, :]
    else:
        mu = np.array(mu)[1:, np.newaxis] @ np.ones((1, k))
        mu = mu[1:, :]

    # drift
    nu = mu - sd**2 / 2.0

    # do a Cholesky factorization on the correlation matrix
    R = np.linalg.cholesky(corr).T

    # generate uncorrelated random sequence
    x = np.random.normal(size=(n - 1, k))

    # correlate the sequences
    ep = x @ R

    # multivariate brownian
    W = nu * dt + ep @ np.diag(sd) * np.sqrt(dt)

    # generate potential path
    S = np.vstack([np.ones((1, k)), np.cumprod(np.exp(W), 0)])

    # add nan values
    if nan_pct > 0:
        r = S * 0 + np.random.random(S.shape)
        S[r < nan_pct] = np.nan

    return pd.DataFrame(S, columns=["S{}".format(i) for i in range(S.shape[1])])


def opt_weights(
    X,
    metric="return",
    max_leverage=1,
    rf_rate=0.0,
    alpha=0.0,
    freq: float = 252,
    no_cash=False,
    sd_factor=1.0,
    **kwargs,
):
    """Find best constant rebalanced portfolio with regards to some metric.
    :param X: Prices in ratios.
    :param metric: what performance metric to optimize, can be either `return` or `sharpe`
    :max_leverage: maximum leverage
    :rf_rate: risk-free rate for `sharpe`, can be used to make it more aggressive
    :alpha: regularization parameter for volatility in sharpe
    :freq: frequency for sharpe (default 252 for daily data)
    :no_cash: if True, we can't keep cash (that is sum of weights == max_leverage)
    """
    assert metric in ("return", "sharpe", "drawdown", "ulcer")
    assert X.notnull().all().all()

    x_0 = max_leverage * np.ones(X.shape[1]) / float(X.shape[1])
    if metric == "return":
        objective = lambda b: -np.sum(np.log(np.maximum(np.dot(X - 1, b) + 1, 0.0001)))
    elif metric == "ulcer":
        objective = lambda b: -ulcer(
            np.log(np.maximum(np.dot(X - 1, b) + 1, 0.0001)), rf_rate=rf_rate, freq=freq
        )
    elif metric == "sharpe":
        objective = lambda b: -sharpe(
            np.log(np.maximum(np.dot(X - 1, b) + 1, 0.0001)),
            rf_rate=rf_rate,
            alpha=alpha,
            freq=freq,
            sd_factor=sd_factor,
        )
    elif metric == "drawdown":

        def objective(b):
            R = np.dot(X - 1, b) + 1
            L = np.cumprod(R)
            dd = max(1 - L / np.maximum.accumulate(L))
            annual_ret = np.mean(R) ** freq - 1
            return -annual_ret / (dd + alpha)

    if no_cash:
        cons = ({"type": "eq", "fun": lambda b: max_leverage - sum(b)},)
    else:
        cons = ({"type": "ineq", "fun": lambda b: max_leverage - sum(b)},)

    while True:
        # problem optimization
        res = optimize.minimize(
            objective,
            x_0,
            bounds=[(0.0, max_leverage)] * len(x_0),
            constraints=cons,
            method="slsqp",
            **kwargs,
        )

        # result can be out-of-bounds -> try it again
        EPS = 1e-7
        if (res.x < 0.0 - EPS).any() or (res.x > max_leverage + EPS).any():
            X = X + np.random.randn(1)[0] * 1e-5
            logging.debug("Optimal weights not found, trying again...")
            continue
        elif res.success:
            break
        else:
            if np.isnan(res.x).any():
                logging.warning("Solution does not exist, use zero weights.")
                res.x = np.zeros(X.shape[1])
            else:
                logging.warning("Converged, but not successfully.")
            break

    return res.x


def opt_markowitz(
    mu, sigma, long_only=True, reg=0.0, rf_rate=0.0, q=1.0, max_leverage=1.0
):
    """Get optimal weights from Markowitz framework."""
    # delete assets with NaN values or no volatility
    keep = ~(mu.isnull() | (np.diag(sigma) < 0.00000001))

    mu = mu[keep]
    sigma = sigma.loc[keep, keep]

    m = len(mu)

    # replace NaN values with 0
    sigma = sigma.fillna(0.0)

    # convert to matrices
    sigma = np.matrix(sigma)
    mu = np.matrix(mu).T

    # regularization for sigma matrix
    sigma += np.eye(m) * reg

    # pure approach - problems with singular matrix
    if not long_only:
        sigma_inv = np.linalg.inv(sigma)
        b = q / 2 * (1 + rf_rate) * sigma_inv @ (mu - rf_rate)
        b = np.ravel(b)
    else:

        def maximize(mu, sigma, r, q):
            n = len(mu)

            P = 2 * matrix((sigma - r * mu * mu.T + (n * r) ** 2) / (1 + r))
            q = matrix(-mu) * q
            G = matrix(-np.eye(n))
            h = matrix(np.zeros(n))

            if max_leverage is None or max_leverage == float("inf"):
                sol = solvers.qp(P, q, G, h)
            else:
                A = matrix(np.ones(n)).T
                b = matrix(np.array([float(max_leverage)]))
                sol = solvers.qp(P, q, G, h, A, b)

            return np.squeeze(sol["x"])

        while True:
            try:
                b = maximize(mu, sigma, rf_rate, q)
                break
            except ValueError:
                raise
                # deal with singularity
                logging.warning("Singularity problems")
                sigma = sigma + 0.0001 * np.eye(len(sigma))

    # add back values for NaN assets
    b = pd.Series(b, index=keep.index[keep])
    b = b.reindex(keep.index).fillna(0.0)

    return b


def bcrp_weights(X):
    """Find best constant rebalanced portfolio.
    :param X: Prices in ratios.
    """
    return opt_weights(X)


def rolling_cov_pairwise(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return df.rolling(**kwargs).cov(other=df, pairwise=True)


def rolling_corr(x, y, **kwargs):
    """Rolling correlation between columns from x and y."""

    def rolling(dataframe, *args, **kwargs):
        ret = dataframe.copy()
        for col in ret:
            ret[col] = ret[col].rolling(*args, **kwargs).mean()
        return ret

    n, k = x.shape

    EX = rolling(x, **kwargs)
    EY = rolling(y, **kwargs)
    EX2 = rolling(x**2, **kwargs)
    EY2 = rolling(y**2, **kwargs)

    RXY = np.zeros((n, k, k))

    for i, col_x in enumerate(x):
        for j, col_y in enumerate(y):
            DX = EX2[col_x] - EX[col_x] ** 2
            DY = EY2[col_y] - EY[col_y] ** 2
            product = x[col_x] * y[col_y]
            RXY[:, i, j] = product.rolling(**kwargs).mean() - EX[col_x] * EY[col_y]
            RXY[:, i, j] = RXY[:, i, j] / np.sqrt(DX * DY)

    return RXY, EX.values


def simplex_proj(y):
    """Projection of y onto simplex."""
    m = len(y)
    bget = False

    s = sorted(y, reverse=True)
    tmpsum = 0.0

    for ii in range(m - 1):
        tmpsum = tmpsum + s[ii]
        tmax = (tmpsum - 1) / (ii + 1)
        if tmax >= s[ii + 1]:
            bget = True
            break

    if not bget:
        tmax = (tmpsum + s[m - 1] - 1) / m

    return np.maximum(y - tmax, 0.0)


def __mesh(d, k):
    """Return integer non-negative solutions to equation x_1 + ... x_d = k."""
    if d == 1:
        yield [k]
    else:
        for i in range(k + 1):
            for s in __mesh(d - 1, k - i):
                yield [i] + s


def simplex_mesh(d, points):
    """Create uniform grid on simplex. In 2-dim case the algorithm just selects
    equally spaced points on interval [0,1]. In 3-dim, it selects vertices of 3-simplex
    triangulation.
    :param d: Number of dimensions.
    :param points: Total number of points (approximately).
    """
    # find k for __mesh such that points is number of points
    # total number of points is combination(d + k - 1, k)
    fun = lambda k: -np.log(d + k) - betaln(k + 1, d) - np.log(points)
    k = int(optimize.newton(fun, x0=1))
    k = max(k, 1)
    return np.array(sorted(__mesh(d, k))) / float(k)


def mc_simplex(d, points):
    """Sample random points from a simplex with dimension d.
    :param d: Number of dimensions.
    :param points: Total number of points.
    """
    a = np.sort(np.random.random((points, d)))
    a = np.hstack([np.zeros((points, 1)), a, np.ones((points, 1))])
    return np.diff(a)


def combinations(S, r):
    """Generator of all r-element combinations of stocks from portfolio S."""
    for ncols in itertools.combinations(S.columns, r):
        # yield S.iloc[:,ncols]
        yield S[list(ncols)]


def log_progress(i, total, by=1):
    """Log progress by pcts."""
    progress = ((100 * i / total) // by) * by
    last_progress = ((100 * (i - 1) / total) // by) * by

    if progress != last_progress:
        logging.debug("Progress: {}%...".format(progress))


def mu_std(R, rf_rate=None, freq=None):
    """Return mu and std."""
    freq = freq or _freq(R.index)

    if rf_rate is None:
        rf_rate = R["RFR"]

    # adjust rf rate by frequency
    rf = rf_rate / freq

    # subtract risk-free rate
    mu, sd = (R.sub(rf, 0)).mean(), (R.sub(rf, 0)).std()

    # annualize return and sd
    mu = mu * freq
    sd = sd * np.sqrt(freq)

    return pd.DataFrame(
        {
            "mu": mu,
            "sd": sd,
        }
    )


def _sub_rf(r, rf):
    if isinstance(rf, float):
        r -= rf
    elif len(r.shape) == 1:
        r -= rf.values
    else:
        r = r.sub(rf, 0)
    return r


def ulcer(r, rf_rate=0.0, freq=None):
    """Compute Ulcer ratio."""
    freq = freq or _freq(r.index)
    rf = rf_rate / freq

    # subtract risk-free rate
    r = _sub_rf(r, rf)

    # annualized excess return
    mu = r.mean() * freq

    # ulcer index
    x = (1 + r).cumprod()

    if isinstance(x, pd.Series):
        drawdown = 1 - x / x.cummax()
    else:
        drawdown = 1 - x / np.maximum.accumulate(x)

    if (drawdown == 0).all():
        return 0

    return mu / np.sqrt((drawdown**2).mean())


def sharpe(r, rf_rate=0.0, alpha=0.0, freq=None, sd_factor=1.0, w=None):
    """Compute annualized sharpe ratio from returns. If data does
    not contain datetime index, assume daily frequency with 252 trading days a year

    See https://treasury.govt.nz/sites/default/files/2007-09/twp03-28.pdf for more info.
    """
    freq = freq or _freq(r.index)

    # adjust rf rate by frequency
    rf = rf_rate / freq

    # subtract risk-free rate
    r = _sub_rf(r, rf)

    # annualize return and sd
    if w is None:
        mu = r.mean()
        sd = r.std()
    else:
        mu = w_avg(r, w)
        sd = w_std(r, w)

    mu = mu * freq
    sd = sd * np.sqrt(freq)

    if sd + alpha == 0:
        return 0

    sh = mu / (sd + alpha) ** sd_factor

    if isinstance(sh, float):
        if sh == np.inf:
            return np.inf * np.sign(mu - rf ** (1.0 / freq))
    else:
        pass
        # sh[sh == np.inf] *= np.sign(mu - rf**(1./freq))
    return sh


def w_avg(y, w):
    return (y * w).sum() / w.sum()


def w_std(y, w):
    return np.sqrt(np.maximum(0, w_avg(y**2, w) - (w_avg(y, w)) ** 2))


def sharpe_std(r, rf_rate=None, freq=None):
    """Calculate sharpe ratio std. Confidence interval is taken from
    https://quantdare.com/probabilistic-sharpe-ratio
    :param X: log returns
    """
    sh = sharpe(r, rf_rate=rf_rate, freq=freq)
    n = r.notnull().sum()
    freq = freq or _freq(r.index)

    # Normalize sh to daily
    sh /= np.sqrt(freq)

    # Simpler version
    # ret = np.sqrt((1.0 + sh**2 / 2.0) / n)

    # More complex version
    mean_return = np.mean(r)
    std_dev = np.std(r, ddof=1)
    skewness = np.mean(((r - mean_return) / std_dev) ** 3)
    kurtosis = np.mean(((r - mean_return) / std_dev) ** 4)
    ret = np.sqrt(
        (1 + sh**2 / 2.0 - skewness * sh + (kurtosis - 3) / 4 * sh**2) / (n - 1)
    )

    # Back to yearly sharpe
    return ret * np.sqrt(freq)


def freq(ix: pd.Index) -> float:
    """Number of data items per year. If data does not contain
    datetime index, assume daily frequency with 252 trading days a year."""
    assert isinstance(ix, pd.Index), "freq method only accepts pd.Index object"

    assert len(ix) > 1, "Index must contain more than one item"

    # sort if data is not monotonic
    if not ix.is_monotonic_increasing:
        ix = ix.sort_values()

    if isinstance(ix, pd.DatetimeIndex):
        days = (ix[-1] - ix[0]).days
        return len(ix) / float(days) * 365.0
    else:
        return 252.0


# add alias to allow use of freq keyword in functions
_freq = freq


def fill_synthetic_data(S, corr_threshold=0.95, backfill=False, beta_type="regression"):
    """Fill synthetic history of ETFs based on history of other stocks (e.g. UBT is 2x TLT).
    If there's an asset with corr_threshold higher than corr_threshold, we use its returns
    to calculate returns for missing values. Otherwise we will use the same price.
    """
    # revert prices (count backwards)
    S = S.iloc[::-1]

    # convert data into returns
    X = S / S.shift(1) - 1

    # compute correlation
    corr = X.corr()

    # go over stocks from those with longest history
    ordered_cols = (S.isnull()).sum().sort_values().index
    for i, col in enumerate(ordered_cols):
        if i > 0 and S[col].isnull().any():
            # find maximum correlation
            synth = corr.loc[col, ordered_cols[:i]].idxmax()

            if pd.isnull(synth):
                logging.info("NaN proxy for {} found, backfill prices".format(col))
                continue

            cr = corr.loc[col, synth]
            if abs(cr) >= corr_threshold:
                nn = X[col].notnull()

                if beta_type == "regression":
                    # calculate b in y = b*x
                    b = (X.loc[nn, col] * X.loc[nn, synth]).sum() / (
                        X.loc[nn, synth] ** 2
                    ).sum()
                elif beta_type == "std":
                    # make sure standard deviation is identical
                    b = X.loc[nn, col].std() / X.loc[nn, synth].std()
                else:
                    raise NotImplementedError()

                # fill missing data
                X.loc[~nn, col] = b * X.loc[~nn, synth]

                logging.info(
                    "Filling missing values of {} by {:.2f}*{} (correlation {:.2f})".format(
                        col, b, synth, cr
                    )
                )
            else:
                if backfill:
                    logging.info("No proxy for {} found, backfill prices.".format(col))
                else:
                    logging.info("No proxy for {} found.".format(col))

    # reconstruct prices by going from end
    X = X + 1
    X.iloc[0] = S.iloc[0]

    # revert back
    S = X.cumprod().iloc[::-1]

    # fill missing values backward
    if backfill:
        S = S.fillna(method="bfill")

    return S


def fill_regressed_data(S):
    """Fill missing returns by linear combinations of assets without missing returns."""
    S = S.copy()
    R = np.log(S).diff()
    R.iloc[0] = 0

    X = R.dropna(1)

    for col in set(S.columns) - set(X.columns):
        R[col].iloc[0] = np.nan
        y = R[col]

        # fit regression
        res = sm.OLS(y=y, x=X, intercept=True).fit()
        pred = res.predict(x=X[y.isnull()])

        # get absolute prices
        pred = pred.cumsum()
        pred += np.log(S[col].dropna().iloc[0]) - pred.iloc[-1]

        # fill missing data
        S[col] = S[col].fillna(np.exp(pred))

    return S


def short_assets(S):
    """Create synthetic short assets. If return of an asset is more than 100%, short asset
    will go to zero."""
    X = S / S.shift(1)

    # shorting
    X = 2 - X
    X.iloc[0] = S.iloc[0]

    # negative return means we got bankrupt
    X = X.clip(lower=0)

    # reconstruct original
    return X.cumprod()


def bootstrap_history(S, drop_fraction=0.1, size=None, random_state=None):
    """Remove fraction of days and reconstruct time series from remaining days. Useful for stress-testing
    strategies."""
    # work with returns
    R = S / S.shift(1)

    # drop days randomly
    if random_state is not None:
        np.random.seed(random_state)

    if size is None:
        size = int(len(R) * (1 - drop_fraction))

    ix = np.random.choice(R.index, size=size, replace=False)
    R = R.loc[sorted(ix)]

    # reconstruct series
    R.iloc[0] = S.loc[R.index[0]]
    return R.cumprod()


def _bootstrap_mp(algo_bS):
    algo, bS = algo_bS
    return algo.run(bS)


def bootstrap_algo(S, algo, n, drop_fraction=0.1, random_state=None, n_jobs=-1):
    """Use bootstrap_history to create several simulated results of our strategy
    and evaluate algo on those samples paralelly."""
    if random_state:
        np.random.seed(random_state)

    def generator():
        for _ in range(n):
            bS = bootstrap_history(S, drop_fraction=drop_fraction)
            yield algo, bS

    with mp_pool(n_jobs) as pool:
        results = pool.map(_bootstrap_mp, generator())
    return results


def cov_to_corr(sigma):
    """Convert covariance matrix to correlation matrix."""
    return sigma / np.sqrt(np.outer(np.diag(sigma), np.diag(sigma)))


def get_cash(rfr, ib_fee=0.015):
    rf_rate = 1 + (rfr + ib_fee) / freq(rfr.index)
    cash = pd.Series(rf_rate, index=rfr.index)
    cash = cash.cumprod()
    cash = cash / cash.iloc[-1]
    return cash


def tradable_etfs():
    return [
        "TLT",
        "SPY",
        "RSP",
        "GLD",
        "EDV",
        "MDY",
        "QQQ",
        "IWM",
        "EFA",
        "IYR",
        "ASHR",
        "SSO",
        "TMF",
        "UPRO",
        "EDC",
        "TQQQ",
        "XIV",
        "ZIV",
        "EEM",
        "UGLD",
        "FAS",
        "UDOW",
        "UMDD",
        "URTY",
        "TNA",
        "ERX",
        "BIB",
        "UYG",
        "RING",
        "LABU",
        "XLE",
        "XLF",
        "IBB",
        "FXI",
        "XBI",
        "XSD",
        "GOOGL",
        "AAPL",
        "VNQ",
        "DRN",
        "O",
        "IEF",
        "GBTC",
        "KBWY",
        "KBWR",
        "DPST",
        "YINN",
        "FHK",
        "XOP",
        "GREK",
        "SIL",
        "JPNL",
        "KRE",
        "IAT",
        "SOXL",
        "RETL",
        "VIXM",
        "QABA",
        "KBE",
        "USDU",
        "UUP",
        "TYD",
    ]


def same_vol(S, target=None, target_vol=None):
    R = S.pct_change().drop(columns=["RFR"])
    rfr = S["RFR"]
    vol = R.std()
    if not target_vol:
        target_vol = vol[target] if target else vol.mean()
    leverage = target_vol / vol
    R = (leverage * (R.sub(rfr / 252, axis=0))).add(rfr / 252, axis=0)
    S = (1 + R.fillna(0)).cumprod()
    S["RFR"] = rfr
    return S


def capm(y: pd.Series, bases: pd.DataFrame, rf=0.0, fee=0.0):
    freq = _freq(y.index)
    rf = rf / freq
    fee = fee / freq
    R = y.pct_change() - rf
    R.name = y.name
    R_base = bases.pct_change().sub(rf, axis=0)

    # CAPM:
    # R = alpha + rf + beta * (Rm - rf)
    model = OLS(R, R_base.assign(Intercept=1), missing="drop").fit()

    alpha = model.params["Intercept"] * freq
    betas = model.params[bases.columns]

    # reconstruct artificial portfolio
    proxy = R_base @ betas + (1 - betas.sum()) * (rf + fee)
    cumproxy = (1 + proxy).cumprod()

    # residual portfolio
    r = y.pct_change() - cumproxy.pct_change()
    residual = (1 + r).cumprod()

    return {
        "alpha": alpha,
        "betas": betas,
        "cumproxy": cumproxy,
        "model": model,
        "residual": residual,
    }


def to_rebalance(B, X):
    """
    :param X: price relatives (1 + r)
    """
    # equity increase
    E = (B * (X - 1)).sum(axis=1) + 1

    X = X.copy()

    # calculate new value of assets and normalize by new equity to get
    # weights for tomorrow
    hold_B = (B * X).div(E, axis=0)

    return B - hold_B.shift(1)
