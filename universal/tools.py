import pandas as pd
import numpy as np
import scipy.optimize as optimize
from scipy.special import betaln
from pandas.stats.moments import rolling_mean as rolling_m
from pandas.stats.moments import rolling_corr
import matplotlib.pyplot as plt
from time import time
from datetime import datetime
from pandas.io.data import DataReader
import sys
import os
import logging
import itertools
import multiprocessing
from pandas.stats.api import ols
import contextlib
from cvxopt import solvers, matrix
solvers.options['show_progress'] = False


@contextlib.contextmanager
def mp_pool(n_jobs):
    n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
    pool = multiprocessing.Pool(n_jobs)
    try:
        yield pool
    finally:
        pool.close()


def dataset(name):
    """ Return sample dataset from /data directory. """
    mod = sys.modules[__name__]
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', name + '.pkl')
    return pd.read_pickle(filename)


def profile(algo, data=None, to_profile=[]):
    """ Profile algorithm using line_profiler.
    :param algo: Algorithm instance.
    :param data: Stock prices, default is random portfolio.
    :param to_profile: List of methods to profile, default is `step` method.

    Example of use:
        tools.profile(Anticor(window=30, c_version=False), to_profile=[Anticor.weights])
    """
    from line_profiler import LineProfiler

    if data is None:
        data = random_portfolio(n=1000, k=10, mu=0.)

    to_profile = to_profile or [algo.step]
    profile = LineProfiler(*to_profile)
    profile.runcall(algo.run, data)
    profile.print_stats()


def load_ticker(ticker, start=datetime(2000, 1, 1), end=None):
    return DataReader(ticker,  "yahoo", start=start, end=None)


def quickrun(algo, data=None, n=1000, **kwargs):
    """ Run algorithm and print its running time and some statistics. """
    if data is None:
        data = random_portfolio(n=n, k=3, mu=0.0001)
    t = time()
    result = algo.run(data)
    logging.debug('Time: {:.2f}s'.format(time() - t))

    print(result.summary())
    result.plot(**kwargs)
    plt.show()

    return result


def random_portfolio(n, k, mu=0., sd=0.01, corr=None, dt=1., nan_pct=0.):
    """ Generate asset prices assuming multivariate geometric Brownian motion.

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
    mu = mu * np.ones(k)

    # drift
    nu = mu - sd**2 / 2.

    # do a Cholesky factorization on the correlation matrix
    R = np.linalg.cholesky(corr).T

    # generate uncorrelated random sequence
    x = np.matrix(np.random.normal(size=(n - 1,k)))

    # correlate the sequences
    ep = x * R

    # multivariate brownian
    W = nu * dt + ep * np.diag(sd) * np.sqrt(dt)

    # generate potential path
    S = np.vstack([np.ones((1, k)), np.cumprod(np.exp(W), 0)])

    # add nan values
    if nan_pct > 0:
        r = S * 0 + np.random.random(S.shape)
        S[r < nan_pct] = np.nan

    return pd.DataFrame(S, columns=['S{}'.format(i) for i in range(S.shape[1])])


def opt_weights(X, metric='return', max_leverage=1, rf_rate=0., alpha=0., freq=252, no_cash=False, sd_factor=1., **kwargs):
    """ Find best constant rebalanced portfolio with regards to some metric.
    :param X: Prices in ratios.
    :param metric: what performance metric to optimize, can be either `return` or `sharpe`
    :max_leverage: maximum leverage
    :rf_rate: risk-free rate for `sharpe`, can be used to make it more aggressive
    :alpha: regularization parameter for volatility in sharpe
    :freq: frequency for sharpe (default 252 for daily data)
    :no_cash: if True, we can't keep cash (that is sum of weights == max_leverage)
    """
    assert metric in ('return', 'sharpe', 'drawdown')

    x_0 = max_leverage * np.ones(X.shape[1]) / float(X.shape[1])
    if metric == 'return':
        objective = lambda b: -np.sum(np.log(np.maximum(np.dot(X - 1, b) + 1, 0.0001)))
    elif metric == 'sharpe':
        objective = lambda b: -sharpe(np.log(np.maximum(np.dot(X - 1, b) + 1, 0.0001)),
                                      rf_rate=rf_rate, alpha=alpha, freq=freq, sd_factor=sd_factor)
    elif metric == 'drawdown':
        def objective(b):
            R = np.dot(X - 1, b) + 1
            L = np.cumprod(R)
            dd = max(1 - L / np.maximum.accumulate(L))
            annual_ret = np.mean(R) ** freq - 1
            return -annual_ret / (dd + alpha)

    if no_cash:
        cons = ({'type': 'eq', 'fun': lambda b: max_leverage - sum(b)},)
    else:
        cons = ({'type': 'ineq', 'fun': lambda b: max_leverage - sum(b)},)

    while True:
        # problem optimization
        res = optimize.minimize(objective, x_0, bounds=[(0., max_leverage)]*len(x_0), constraints=cons, method='slsqp', **kwargs)

        # result can be out-of-bounds -> try it again
        EPS = 1E-7
        if (res.x < 0. - EPS).any() or (res.x > max_leverage + EPS).any():
            X = X + np.random.randn(1)[0] * 1E-5
            logging.debug('Optimal weights not found, trying again...')
            continue
        elif res.success:
            break
        else:
            if np.isnan(res.x).any():
                logging.warning('Solution does not exist, use zero weights.')
                res.x = np.zeros(X.shape[1])
            else:
                logging.warning('Converged, but not successfully.')
            break

    return res.x


def opt_markowitz(mu, sigma, long_only=True, reg=0., rf_rate=0., q=1., max_leverage=1.):
    """ Get optimal weights from Markowitz framework. """
    # delete assets with NaN values or no volatility
    keep = ~(mu.isnull() | (np.diag(sigma) < 0.00000001))

    mu = mu[keep]
    sigma = sigma.ix[keep, keep]

    m = len(mu)

    # replace NaN values with 0
    sigma = sigma.fillna(0.)

    # convert to matrices
    sigma = np.matrix(sigma)
    mu = np.matrix(mu).T

    # regularization for sigma matrix
    sigma += np.eye(m) * reg

    # pure approach - problems with singular matrix
    if not long_only:
        sigma_inv = np.linalg.inv(sigma)
        b = (1 + rf_rate) * sigma_inv * (mu - rf_rate)
        b = np.ravel(b)
    else:
        def maximize(mu, sigma, r, q):
            n = len(mu)

            P = 2 * matrix((sigma - r*mu*mu.T + (n*r)**2) / (1+r))
            q = matrix(-mu) * q
            G = matrix(-np.eye(n))
            h = matrix(np.zeros(n))

            if max_leverage is None or max_leverage == float('inf'):
                sol = solvers.qp(P, q, G, h)
            else:
                A = matrix(np.ones(n)).T
                b = matrix(np.array([max_leverage]))
                sol = solvers.qp(P, q, G, h, A, b)

            return np.squeeze(sol['x'])

        while True:
            try:
                b = maximize(mu, sigma, rf_rate, q)
                break
            except ValueError:
                raise
                # deal with singularity
                logging.warning('Singularity problems')
                sigma = sigma + 0.0001 * np.eye(len(sigma))

    # add back values for NaN assets
    b = pd.Series(b, index=keep.index[keep])
    b = b.reindex(keep.index).fillna(0.)

    return b


def bcrp_weights(X):
    """ Find best constant rebalanced portfolio.
    :param X: Prices in ratios.
    """
    return opt_weights(X)


def rolling_cov_pairwise(df, *args, **kwargs):
    d = {}
    for c in df.columns:
        d[c] = pd.rolling_cov(df[c], df, *args, **kwargs)
    p = pd.Panel(d)
    return p.transpose(1, 0, 2)


def rolling_corr(x, y, **kwargs):
    """ Rolling correlation between columns from x and y. """
    def rolling(dataframe, *args, **kwargs):
        ret = dataframe.copy()
        for col in ret:
            ret[col] = rolling_m(ret[col], *args, **kwargs)
        return ret

    n, k = x.shape

    EX = rolling(x, **kwargs)
    EY = rolling(y, **kwargs)
    EX2 = rolling(x ** 2, **kwargs)
    EY2 = rolling(y ** 2, **kwargs)

    RXY = np.zeros((n, k, k))

    for i, col_x in enumerate(x):
        for j, col_y in enumerate(y):
            DX = EX2[col_x] - EX[col_x] ** 2
            DY = EY2[col_y] - EY[col_y] ** 2
            RXY[:, i, j] = rolling_m(x[col_x] * y[col_y], **kwargs) - EX[col_x] * EY[col_y]
            RXY[:, i, j] = RXY[:, i, j] / np.sqrt(DX * DY)

    return RXY, EX.values


def simplex_proj(y):
    """ Projection of y onto simplex. """
    m = len(y)
    bget = False

    s = sorted(y, reverse=True)
    tmpsum = 0.

    for ii in range(m-1):
        tmpsum = tmpsum + s[ii]
        tmax = (tmpsum - 1) / (ii + 1);
        if tmax >= s[ii+1]:
            bget = True
            break

    if not bget:
        tmax = (tmpsum + s[m-1] -1)/m

    return np.maximum(y-tmax,0.)


def __mesh(d, k):
    """ Return integer non-negative solutions to equation x_1 + ... x_d = k."""
    if d == 1:
        yield [k]
    else:
        for i in range(k+1):
            for s in __mesh(d-1, k-i):
                yield [i] + s


def simplex_mesh(d, points):
    """ Create uniform grid on simplex. In 2-dim case the algorithm just selects
    equally spaced points on interval [0,1]. In 3-dim, it selects vertices of 3-simplex
    triangulation.
    :param d: Number of dimensions.
    :param points: Total number of points (approximately).
    """
    # find k for __mesh such that points is number of points
    # total number of points is combination(d + k - 1, k)
    fun = lambda k: - np.log(d+k) - betaln(k+1,d) - np.log(points)
    k = int(optimize.newton(fun, x0=1))
    k = max(k,1)
    return np.array(sorted(__mesh(d,k))) / float(k)


def mc_simplex(d, points):
    """ Sample random points from a simplex with dimension d.
    :param d: Number of dimensions.
    :param points: Total number of points.
    """
    a = np.sort(np.random.random((points, d)))
    a = np.hstack([np.zeros((points,1)), a, np.ones((points,1))])
    return np.diff(a)


def combinations(S, r):
    """ Generator of all r-element combinations of stocks from portfolio S. """
    for ncols in itertools.combinations(S.columns, r):
        #yield S.iloc[:,ncols]
        yield S[list(ncols)]


def log_progress(i, total, by=1):
    """ Log progress by pcts. """
    progress = ((100 * i / total) // by) * by
    last_progress = ((100 * (i-1) / total) // by) * by

    if progress != last_progress:
        logging.debug('Progress: {}%...'.format(progress))


def sharpe(r_log, rf_rate=0., alpha=0., freq=None, sd_factor=1.):
    """ Compute annualized sharpe ratio from log returns. If data does
        not contain datetime index, assume daily frequency with 252 trading days a year

        TODO: calculate real sharpe ratio (using price relatives), see
            http://www.treasury.govt.nz/publications/research-policy/wp/2003/03-28/twp03-28.pdf
    """
    freq = freq or _freq(r_log.index)

    mu, sd = r_log.mean(), r_log.std()

    # annualize return and sd
    mu = mu * freq
    sd = sd * np.sqrt(freq)

    # risk-free rate
    rf = np.log(1 + rf_rate)

    sh = (mu - rf) / (sd + alpha)**sd_factor

    if isinstance(sh, float):
        if sh == np.inf:
            return np.inf * np.sign(mu - rf**(1./freq))
    else:
        sh[sh == np.inf] *= np.sign(mu - rf**(1./freq))
    return sh


def sharpe_std(X):
    """ Calculate sharpe ratio std. Confidence interval is taken from
    https://cran.r-project.org/web/packages/SharpeR/vignettes/SharpeRatio.pdf
    :param X: log returns
    """
    sh = sharpe(X)
    n = X.notnull().sum()
    f = freq(X.index)
    return np.sqrt((1. + sh**2/2.) * f / n)


def freq(ix):
    """ Number of data items per year. If data does not contain
    datetime index, assume daily frequency with 252 trading days a year."""
    assert isinstance(ix, pd.Index), 'freq method only accepts pd.Index object'

    # sort if data is not monotonic
    if not ix.is_monotonic:
        ix = ix.order()

    if isinstance(ix, pd.DatetimeIndex):
        days = (ix[-1] - ix[0]).days
        return len(ix) / float(days) * 365.
    else:
        return 252.

# add alias to allow use of freq keyword in functions
_freq = freq


def fill_synthetic_data(S, corr_threshold=0.95, backfill=False):
    """ Fill synthetic history of ETFs based on history of other stocks (e.g. UBT is 2x TLT).
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
    ordered_cols = (S.isnull()).sum().order().index
    for i, col in enumerate(ordered_cols):
        if i > 0 and S[col].isnull().any():
            # find maximum correlation
            synth = corr.ix[col, ordered_cols[:i]].idxmax()

            cr = corr.ix[col, synth]
            if abs(cr) >= corr_threshold:
                # calculate b in y = b*x
                nn = X[col].notnull()
                b = (X.ix[nn, col] * X.ix[nn, synth]).sum() / (X.ix[nn, synth]**2).sum()

                # fill missing data
                X.ix[~nn, col] = b * X.ix[~nn, synth]

                logging.info('Filling missing values of {} by {:.2f}*{} (correlation {:.2f})'.format(
                        col, b, synth, cr))
            else:
                if backfill:
                    logging.info('No proxy for {} found, backfill prices.'.format(col))
                else:
                    logging.info('No proxy for {} found.'.format(col))

    # reconstruct prices by going from end
    X = X + 1
    X.iloc[0] = S.iloc[0]

    # revert back
    S = X.cumprod().iloc[::-1]

    # fill missing values backward
    if backfill:
        S = S.fillna(method='bfill')

    return S


def fill_regressed_data(S):
    """ Fill missing returns by linear combinations of assets without missing returns. """
    S = S.copy()
    R = np.log(S).diff()
    R.iloc[0] = 0

    X = R.dropna(1)

    for col in set(S.columns) - set(X.columns):
        R[col].iloc[0] = np.nan
        y = R[col]

        # fit regression
        res = ols(y=y, x=X, intercept=True)
        pred = res.predict(x=X[y.isnull()])

        # get absolute prices
        pred = pred.cumsum()
        pred += np.log(S[col].dropna().iloc[0]) - pred.iloc[-1]

        # fill missing data
        S[col] = S[col].fillna(np.exp(pred))

    return S


def short_assets(S):
    """ Create synthetic short assets. """
    X = S / S.shift(1)

    # shorting
    X = 2 - X
    X.iloc[0] = S.iloc[0]

    # reconstruct original
    return X.cumprod()


def bootstrap_history(S, drop_fraction=0.1, random_state=None):
    """ Remove fraction of days and reconstruct time series from remaining days. Useful for stress-testing
    strategies. """
    # work with returns
    R = S / S.shift(1)

    # drop days randomly
    if random_state is not None:
        np.random.seed(random_state)
    ix = np.random.choice(R.index, size=int(len(R) * (1-drop_fraction)), replace=False)
    R = R.ix[sorted(ix)]

    # reconstruct series
    R.iloc[0] = S.ix[R.index[0]]
    return R.cumprod()


def _bootstrap_mp(algo_bS):
    algo, bS = algo_bS
    return algo.run(bS)


def bootstrap_algo(S, algo, n, drop_fraction=0.1, random_state=None, n_jobs=-1):
    """ Use bootstrap_history to create several simulated results of our strategy
    and evaluate algo on those samples paralelly. """
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
    """ Convert covariance matrix to correlation matrix. """
    return sigma / np.sqrt(np.matrix(np.diag(sigma)).T.dot(np.matrix(np.diag(sigma))))

