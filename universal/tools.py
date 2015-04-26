# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scipy.optimize as optimize
from scipy.special import betaln
from pandas.stats.moments import rolling_mean as rolling_m
from pandas.stats.moments import rolling_corr
from warnings import warn
import matplotlib.pyplot as plt
from time import time
from datetime import datetime
from pandas.io.data import DataReader
import sys
import os
import logging
import itertools


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


def load_ticker(ticker, start=datetime(2000,1,1), end=None):
    return DataReader(ticker,  "yahoo", start=start, end=None)


def quickrun(algo, data=None, **kwargs):
    """ Run algorithm and print its running time and some statistics. """
    if data is None:
        data = random_portfolio(n=1000, k=3, mu=0.0001)
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


def bcrp_weights(X):
    """ Find best constant rebalanced portfolio.
    :param X: Prices in ratios.
    """
    x_0 = np.ones(X.shape[1]) / float(X.shape[1])
    fun = lambda b: -np.prod(np.dot(X, b))
    cons = ({'type': 'eq', 'fun': lambda b:  sum(b) - 1.},)
    res = optimize.minimize(fun, x_0, bounds=[(0.,1.)]*len(x_0), constraints=cons, method='slsqp')
    if not res.success:
        warn('BCRP not found', RuntimeWarning)

    return res.x


def rolling_cov_pairwise(df, *args, **kwargs):
    d = {}
    for c in df.columns:
        d[c] = pd.rolling_cov(df[c], df, *args, **kwargs)
    p = pd.Panel(d)
    return p.transpose(1,0,2)


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
