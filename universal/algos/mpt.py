from ..algo import Algo
import numpy as np
import pandas as pd
from sklearn import covariance
from sklearn.base import BaseEstimator
from scipy import optimize
from cvxopt import solvers, matrix
import logging
from .. import tools
solvers.options['show_progress'] = False


class MPT(Algo):
    """ Modern portfolio theory approach. See https://en.wikipedia.org/wiki/Modern_portfolio_theory.
    """

    PRICE_TYPE = 'log'

    def __init__(self, window=float('inf'), mu_estimator=None, cov_estimator=None,
                 min_history=None, max_leverage=1., method='mpt', q=0.01, gamma=0., allow_cash=False, **kwargs):
        """
        :param window: Window for calculating mean and variance. Use float('inf') for entire history.
        :param mu_estimator: TODO
        :param cov_estimator: TODO
        :param min_history: Use zero weights for first min_periods.
        :param max_leverage: Max leverage to use.
        :param method: optimization objective - can be "mpt", "sharpe" and "variance"
        :param q: depends on method, e.g. for "mpt" it is risk aversion parameter (higher means lower aversion to risk)
        :param gamma: Penalize changing weights (can be number or Series with individual weights such as fees)
        :param allow_cash: Allow holding cash (weights doesn't have to sum to 1)
        """
        if np.isinf(window):
            window = int(1e+8)
            min_history = min_history or 50
        else:
            min_history = min_history or window

        super(MPT, self).__init__(min_history=min_history, **kwargs)
        self.window = window
        self.max_leverage = max_leverage
        self.method = method
        self.q = q
        self.gamma = gamma
        self.allow_cash = allow_cash

        if cov_estimator is None:
            cov_estimator = 'empirical'

        if isinstance(cov_estimator, basestring):
            if cov_estimator == 'empirical':
                # use pandas covariance in init_step
                cov_estimator = covariance.EmpiricalCovariance()
            elif cov_estimator == 'ledoit-wolf':
                cov_estimator = covariance.LedoitWolf()
            elif cov_estimator == 'graph-lasso':
                cov_estimator = covariance.GraphLasso()
            elif cov_estimator == 'oas':
                cov_estimator = covariance.OAS()
            else:
                raise NotImplemented('Unknown covariance estimator {}'.format(cov_estimator))

        # handle sklearn models
        if isinstance(cov_estimator, BaseEstimator):
            cov_estimator = CovarianceEstimator(cov_estimator)

        if mu_estimator is None:
            mu_estimator = MuEstimator()

        if isinstance(mu_estimator, basestring):
            if mu_estimator == 'historical':
                mu_estimator = HistoricalEstimator(window)
            elif mu_estimator == 'sharpe':
                mu_estimator = MuEstimator()
            else:
                raise NotImplemented('Unknown mu estimator {}'.format(mu_estimator))

        self.cov_estimator = cov_estimator
        self.mu_estimator = mu_estimator

    def init_step(self, X):
        # replace covariance estimator with empirical covariance and precompute it
        if isinstance(self.cov_estimator, covariance.EmpiricalCovariance):
            class EmpiricalCov(object):
                """ Behave like sklearn covariance estimator. """

                allow_nan = True

                def __init__(self, X, window, min_history):
                    self.C = tools.rolling_cov_pairwise(X, window=window, min_periods=min_history)

                def fit(self, X):
                    # get sigma matrix
                    x = X.iloc[-1]
                    sigma = self.C[x.name]

                    # make sure sigma is properly indexed
                    sigma = sigma.reindex(index=x.index).reindex(columns=x.index)

                    self.covariance_ = sigma.values
                    return self

            self.cov_estimator = CovarianceEstimator(EmpiricalCov(X, self.window, self.min_history))

    def estimate_mu_sigma(self, S):
        history = self._convert_prices(S, self.PRICE_TYPE, self.REPLACE_MISSING)
        X = history.iloc[-self.window:]

        sigma = self.cov_estimator.fit(X)
        mu = self.mu_estimator.fit(X, sigma)

        return mu, sigma

    def step(self, x, last_b, history):
        # get sigma and mu estimates
        X = history.iloc[-self.window:]

        # remove assets with NaN values
        # cov_est = self.cov_estimator.cov_est
        # if hasattr(cov_est, 'allow_nan') and cov_est.allow_nan:
        #     na_assets = (X.notnull().sum() < self.min_history).values
        # else:
        #     na_assets = X.isnull().any().values

        na_assets = (X.notnull().sum() < self.min_history).values

        X = X.iloc[:, ~na_assets]
        x = x[~na_assets]
        last_b = last_b[~na_assets]

        # get sigma and mu estimations
        sigma = self.cov_estimator.fit(X)
        mu = self.mu_estimator.fit(X, sigma)

        # make Series from gamma
        gamma = self.gamma
        if isinstance(gamma, float):
            gamma = x * 0 + gamma
        elif callable(gamma):
            # use gamma as a function
            pass
        else:
            gamma = gamma.reindex(x.index)
            gamma_null = gamma[gamma.isnull()]
            assert len(gamma_null) == 0, 'gamma is missing values for {}'.format(gamma_null.index)

        # find optimal portfolio
        last_b = pd.Series(last_b, index=x.index, name=x.name)
        b = self.optimize(mu, sigma, q=self.q, gamma=gamma, max_leverage=self.max_leverage, last_b=last_b)

        return pd.Series(b, index=X.columns).reindex(history.columns, fill_value=0.)

    def optimize(self, mu, sigma, q, gamma, max_leverage, last_b):
        if self.method == 'mpt':
            return self._optimize_mpt(mu, sigma, q, gamma, max_leverage, last_b)
        elif self.method == 'sharpe':
            return self._optimize_sharpe(mu, sigma, q, gamma, max_leverage, last_b)
        elif self.method == 'variance':
            return self._optimize_variance(mu, sigma, q, gamma, max_leverage, last_b)
        else:
            raise Exception('Unknown method {}'.format(self.method))

    def _optimize_sharpe(self, mu, sigma, q, gamma, max_leverage, last_b):
        """ Maximize sharpe ratio b.T * mu / sqrt(b.T * sigma * b + q) """
        mu = np.matrix(mu)
        sigma = np.matrix(sigma)

        def maximize(bb):
            if callable(gamma):
                fee_penalization = gamma(bb, last_b)
            else:
                fee_penalization = sum(gamma * abs(bb - last_b))
            bb = np.matrix(bb)
            return -mu * bb.T / np.sqrt(bb * sigma * bb.T + q) + fee_penalization

        if self.allow_cash:
            cons = ({'type': 'ineq', 'fun': lambda b: max_leverage - sum(b)},)
        else:
            cons = ({'type': 'eq', 'fun': lambda b: max_leverage - sum(b)},)

        x0 = last_b
        res = optimize.minimize(maximize, x0, bounds=[(0., max_leverage)]*len(x0),
                                constraints=cons, method='slsqp')

        return res.x

    def _optimize_mpt(self, mu, sigma, q, gamma, max_leverage, last_b):
        """ Minimize b.T * sigma * b - q * b.T * mu """
        sigma = np.matrix(sigma)
        mu = np.matrix(mu).T

        # regularization parameter for singular cases
        ALPHA = 0.001

        def maximize(mu, sigma, q):
            n = len(last_b)

            P = matrix(2 * (sigma + ALPHA * np.eye(n)))
            q = matrix(-q * mu + 2 * ALPHA * np.matrix(last_b).T)
            G = matrix(-np.eye(n))
            h = matrix(np.zeros(n))

            if max_leverage is None or max_leverage == float('inf'):
                sol = solvers.qp(P, q, G, h)
            else:
                if self.allow_cash:
                    G = matrix(np.r_[G, matrix(np.ones(n)).T])
                    h = matrix(np.r_[h, matrix([self.max_leverage])])
                    sol = solvers.qp(P, q, G, h, initvals=last_b)
                else:
                    A = matrix(np.ones(n)).T
                    b = matrix(np.array([max_leverage]))
                    sol = solvers.qp(P, q, G, h, A, b, initvals=last_b)

            if sol['status'] != 'optimal':
                logging.warning("Solution not found for {}, using last weights".format(last_b.name))
                return last_b

            return np.squeeze(sol['x'])

        def maximize_with_penalization(b, last_b, mu, sigma, q, gamma):
            n = len(mu)
            c = np.sign(b - last_b)
            sigma = matrix(sigma)
            mu = matrix(mu)

            P = 2 * (sigma + ALPHA * matrix(np.eye(n)))
            qq = 2 * sigma * matrix(last_b) - q * mu + matrix(gamma * c)

            G = matrix(np.r_[-np.diag(c), np.eye(n), -np.eye(n)])
            h = matrix(np.r_[np.zeros(n), self.max_leverage - last_b, last_b])

            A = matrix(np.ones(n)).T
            b = matrix([self.max_leverage - sum(last_b)])

            sol = solvers.qp(P, qq, G, h, A, b, initvals=np.zeros(n))

            return np.squeeze(sol['x']) + np.array(last_b)

        try:
            b = maximize(mu, sigma, q)
        except ValueError:
            b = last_b

        # second optimization for fees
        if (gamma != 0).any() and (b != last_b).any():
            b = maximize_with_penalization(b, last_b, mu, sigma, q, gamma)
        return b

    def _optimize_variance(self, mu, sigma, q, gamma, max_leverage, last_b):
        """ Minimize b.T * sigma * b subject to b.T * mu >= q. If you find no such solution,
        just maximize return. """
        sigma = np.matrix(sigma)
        mu = np.matrix(mu)

        def maximize(mu, sigma, q):
            n = len(last_b)

            P = matrix(2 * sigma)
            qq = matrix(np.zeros(n))
            G = matrix(np.r_[-np.eye(n), -mu])
            h = matrix(np.r_[np.zeros(n), -q])

            try:
                if max_leverage is None or max_leverage == float('inf'):
                    sol = solvers.qp(P, qq, G, h)
                else:
                    if self.allow_cash:
                        G = matrix(np.r_[G, matrix(np.ones(n)).T])
                        h = matrix(np.r_[h, matrix([self.max_leverage])])
                        sol = solvers.qp(P, qq, G, h, initvals=last_b)
                    else:
                        A = matrix(np.ones(n)).T
                        b = matrix(np.array([max_leverage]))
                        sol = solvers.qp(P, qq, G, h, A, b, initvals=last_b)

                if sol['status'] == 'unknown':
                    raise ValueError()

            except ValueError:
                # no feasible solution - maximize return instead
                P = P * 0
                qq = matrix(-mu.T)
                G = matrix(np.r_[-np.eye(n), matrix(np.ones(n)).T])
                h = matrix(np.r_[np.zeros(n), self.max_leverage])

                sol = solvers.qp(P, qq, G, h)

            return np.squeeze(sol['x'])

        b = maximize(mu, sigma, q)
        return b


class CovarianceEstimator(object):
    """ Estimator which accepts sklearn objects. """

    def __init__(self, cov_est):
        self.cov_est = cov_est

    def fit(self, X):
        # remove zero-variance elements
        zero_variance = X.std() == 0
        Y = X.iloc[:, ~zero_variance.values]

        # can estimator handle NaN values?
        if getattr(self.cov_est, 'allow_nan', False):
            self.cov_est.fit(Y)
            cov = pd.DataFrame(self.cov_est.covariance_, index=Y.columns, columns=Y.columns)
        else:
            # estimation for matrix without NaN values - should be larger than min_history
            cov = self.cov_est.fit(Y.dropna()).covariance_
            cov = pd.DataFrame(cov, index=Y.columns, columns=Y.columns)

            # improve estimation for those with full history
            Y = Y.dropna(1, how='any')
            full_cov = self.cov_est.fit(Y).covariance_
            full_cov = pd.DataFrame(full_cov, index=Y.columns, columns=Y.columns)
            cov.update(full_cov)

        # put back zero covariance
        cov = cov.reindex(X.columns).reindex(columns=X.columns).fillna(0.)

        # annualize covariance
        cov *= tools.freq(X.index)

        return cov


class MuEstimator(object):

    def fit(self, X, sigma):
        # assume that all assets have yearly sharpe ratio 0.5 and deduce return from volatility
        mu = 0.5 * pd.Series(np.sqrt(np.diag(sigma)), index=sigma.index)
        return mu


class MuVarianceEstimator(object):

    def fit(self, X, sigma):
        # assume that all assets have yearly sharpe ratio 1 and deduce return from volatility
        mu = np.matrix(sigma).dot(np.ones(sigma.shape[0]))
        return mu


class HistoricalEstimator(object):

    def __init__(self, window):
        self.window = window

    def fit(self, X, sigma):
        # assume that all assets have yearly sharpe ratio 1 and deduce return from volatility
        mu = X.iloc[-self.window:].mean() * tools.freq(X.index)
        return mu


class MixedMuEstimator(object):

    def __init__(self, window, alpha=0.5):
        self.mu_estimator = MuEstimator()
        self.historical_estimator = HistoricalEstimator(window)
        self.alpha = alpha

    def fit(self, X, sigma):
        prior_mu = self.mu_estimator.fit(X, sigma)
        historical_mu = self.historical_estimator.fit(X, sigma)
        return self.alpha * prior_mu + (1-self.alpha) * historical_mu

# use case
if __name__ == '__main__':
    data = tools.random_portfolio(n=100, k=3, mu=0.0001)
    data.iloc[:20, 0] = np.nan
    tools.quickrun(MPT(min_history=10, window=100, method='variance', cov_estimator='oas'), data=data)
