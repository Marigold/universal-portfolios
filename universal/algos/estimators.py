from sklearn.covariance import EmpiricalCovariance
from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd
from .. import tools
from sklearn.decomposition import PCA
from numpy.linalg import inv
from scipy.linalg import sqrtm


class SharpeEstimator(object):

    GLOBAL_SHARPE = 0.5

    def fit(self, X, sigma):
        # assume that all assets have yearly sharpe ratio 0.5 and deduce return from volatility
        mu = self.GLOBAL_SHARPE * pd.Series(np.sqrt(np.diag(sigma)), index=sigma.index)
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
        if self.window:
            X = X.iloc[-self.window:]

        mu = X.mean()
        mu = (1 + mu)**tools.freq(X.index) - 1
        return mu


class MixedEstimator(object):
    """ Combines historical estimation with sharpe estimation from volatility.
    Has two parameters alpha and beta that works like this:
    alpha in (0, 1) controls regularization of covariance matrix
        alpha = 0 -> assume covariance is zero
        alpha = 1 -> don't regularize
    beta in (0, inf) controls weight we give on historical mean
        beta = 0 -> return is proportional to volatility if alpha = 0 or row sums
            of covariance matrix if alpha = 1
        beta = inf -> use historical return
    """

    def __init__(self, window=None, alpha=0., beta=0.):
        self.GLOBAL_SHARPE = SharpeEstimator.GLOBAL_SHARPE
        self.historical_estimator = HistoricalEstimator(window=window)
        self.alpha = alpha
        self.beta = beta

    def fit(self, X, sigma):
        alpha = self.alpha
        beta = self.beta
        m = X.shape[1]

        # calculate historical return
        historical_mu = self.historical_estimator.fit(X, sigma)

        # regularize sigma
        reg_sigma = alpha * sigma + (1 - alpha) * np.diag(np.diag(sigma))

        # avoid computing inversions
        if beta == 0:
            mu = self.GLOBAL_SHARPE * np.real(sqrtm(reg_sigma)).dot(np.ones(m))
        else:
            # estimate mean
            mu_tmp = beta * historical_mu + self.GLOBAL_SHARPE * inv(np.real(sqrtm(reg_sigma))).dot(np.ones(m))
            mu = inv(inv(reg_sigma) + beta * np.eye(m)).dot(mu_tmp)

        return pd.Series(mu, index=X.columns)


class PCAEstimator(object):

    def __init__(self, window, n_components='mle'):
        self.window = window
        self.n_components = n_components

    def fit(self, X, sigma):
        # take recent period (PCA could be estimated from sigma too)
        R = X.iloc[-self.window:].fillna(0.)

        pca = PCA(n_components=self.n_components).fit(R)
        pca_mu = np.sqrt(pca.explained_variance_) * 0.5 * np.sqrt(tools.freq(X.index))
        comp = pca.components_.T

        # principal components have arbitraty orientation -> choose orientation to maximize final mean return
        comp = comp * np.sign(comp.sum(0))

        pca_mu = comp.dot(pca_mu)
        pca_mu = pd.Series(pca_mu, index=X.columns)
        return pca_mu


class MLEstimator(object):
    """ Predict mean using sklearn model. """

    def __init__(self, model, freq='M'):
        self.model = model
        self.freq = freq

    def featurize(self, H):
        X = pd.DataFrame({
            'last_sh': H.shift(1).stack(),
            'history_sh': pd.expanding_mean(H).shift(1).stack(),
            'history_sh_vol': pd.expanding_std(H).shift(1).stack(),
            'nr_days': H.notnull().cumsum().stack()
        })
        return X

    def fit(self, X, sigma):
        # work with sharpe ratio of log returns (assume raw returns)
        R = np.log(X + 1)
        H = R.resample(self.freq, how=lambda s: s.mean() / s.std() * np.sqrt(tools.freq(X.index)))

        # calculate features
        XX = self.featurize(H)
        yy = H.stack()

        # align training data and drop missing values
        XX = XX.dropna()
        yy = yy.dropna()
        XX = XX.ix[yy.index].dropna()
        yy = yy.ix[XX.index]

        # fit model on historical data
        self.model.fit(XX, yy)
        # print(self.model.intercept_, pd.Series(self.model.coef_, index=XX.columns))

        # make predictions for all assets with features
        XX_pred = XX.ix[XX.index[-1][0]]
        pred_sh = self.model.predict(XX_pred)
        pred_sh = pd.Series(pred_sh, index=XX_pred.index)

        # assume 0.5 sharpe for assets with missing features
        pred_sh = pred_sh.reindex(X.columns).fillna(0.5)

        # convert predictions from sharpe ratio to means
        mu = pred_sh * np.diag(sigma)
        return mu


class SingleIndexCovariance(BaseEstimator):
    """ Estimation of covariance matrix by Ledoit and Wolf (http://www.ledoit.net/ole2.pdf).
    It combines sample covariance matrix with covariance matrix from single-index model and
    automatically estimates shrinking parameter alpha.
    Assumes that first column represents index.
    """

    def __init__(self, alpha=None):
        self.alpha = alpha

    def _sample_covariance(self, X):
        return EmpiricalCovariance().fit(X).covariance_

    def _single_index_covariance(self, X, S):
        # estimate beta from CAPM (use precomputed sample covariance to calculate beta)
        # https://en.wikipedia.org/wiki/Simple_linear_regression#Fitting_the_regression_line
        var_market = S[0,0]
        y = X[:,0]
        beta = S[0,:] / var_market
        alpha = np.mean(X, 0) - beta * np.mean(y)

        # get residuals and their variance
        eps = X - alpha - np.matrix(y).T * np.matrix(beta)
        D = np.diag(np.var(eps, 0))

        return var_market * np.matrix(beta).T * np.matrix(beta) + D

    def _P(self, X, S):
        Xc = X - np.mean(X, 0)
        T, N = X.shape
        P = np.zeros((N, N))
        for i in range(N):
            for j in range(i, N):
                P[i,j] = P[j,i] = sum((Xc[:,i] * Xc[:,j] - S[i,j])**2)
        return P / T

    def _rho(self, X, S, F, P):
        Xc = X - np.mean(X, 0)
        T, N = X.shape
        R = np.zeros((N, N))
        for i in range(N):
            for j in range(i, N):
                g = (S[j,0] * S[0,0] * Xc[:,i] + S[i,0] * S[0,0] * Xc[:,j] - S[i,0]*S[j,0] * Xc[:,0]) / S[0,0]**2
                R[i,j] = R[j,i] = 1./T * sum(g * Xc[:,0] * Xc[:,i] * Xc[:,j] - F[i,j] * S[i,j])
        return np.sum(R)

    def _gamma(self, S, F):
        return np.sum((F - S)**2)

    def _optimal_alpha(self, X, S, F):
        T = X.shape[0]
        P = self._P(X, S)
        phi = np.sum(P)
        gamma = self._gamma(S, F)
        rho = self._rho(X, S, F, P)
        return 1./T * (phi - rho) / gamma

    def fit(self, X):
        # use implicitely with arrays
        X = np.array(X)

        # sample and single-index covariance
        S = self._sample_covariance(X)
        F = self._single_index_covariance(X, S)

        alpha = self.alpha or self._optimal_alpha(X, S, F)
        S_hat = alpha * F + (1 - alpha) * S
        self.covariance_ = S_hat
        self.optimal_alpha_ = alpha
        return self


class HistoricalSharpeEstimator(object):

    PRIOR_SHARPE = 0.3

    def __init__(self, window=None, alpha=1e10, override_sharpe=None):
        self.window = window
        self.alpha = alpha
        self.override_sharpe = override_sharpe or {}

    def fit(self, X, sigma):
        if self.window:
            X = X.iloc[-self.window:]

        # get mean and variance of sharpe ratios
        mu_sh = tools.sharpe(X)
        var_sh = tools.sharpe_std(X) ** 2

        # combine prior sharpe ratio with observations
        alpha = self.alpha
        est_sh = (mu_sh / var_sh + self.PRIOR_SHARPE * alpha) / (1. / var_sh + alpha)

        # override sharpe ratios
        for k, v in self.override_sharpe.items():
            if k in est_sh:
                est_sh[k] = v

        mu = est_sh * pd.Series(np.sqrt(np.diag(sigma)), index=sigma.index)
        # print(est_sh[{'XIV', 'ZIV', 'UGAZ'} & set(est_sh.index)].to_dict())
        return mu
