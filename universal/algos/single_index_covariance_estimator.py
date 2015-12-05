from sklearn.covariance import EmpiricalCovariance
from sklearn.base import BaseEstimator
import numpy as np


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
