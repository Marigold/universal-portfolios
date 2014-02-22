# -*- coding: utf-8 -*-
from universal.algo import Algo
import numpy as np
import pandas as pd
import universal.tools as tools
from cvxopt import solvers, matrix
solvers.options['show_progress'] = False
 

class Kelly(Algo):
    """ Kelly fractioned betting. See
    http://en.wikipedia.org/wiki/Kelly_criterion#Application_to_the_stock_market
    for quick introduction.
    """

    PRICE_TYPE = 'log'
    REPLACE_MISSING = False
    
    def __init__(self, window=float('inf'), r=0., fraction=1., long_only=False, min_history=50):
        """ 
        :param window: Window for calculating mean and variance. Use float('inf') for entire history.
        :param min_history: Use zero weights for first min_periods.
        :param r: Risk-free rate.
        :param long_only: Restrict to positive portfolio weights.
        :param fraction: Use fraction of Kelly weights. 1. is full Kelly, 0.5 is half Kelly.
        """
        super(Kelly, self).__init__(min_history)
        self.window = window if not np.isinf(window) else int(1e+8)
        self.r = r
        self.fraction = fraction
        self.long_only = long_only
    
    
    def init_step(self, X):
        # precalculate correlations
        self.S = tools.rolling_cov_pairwise(X, window=self.window, min_periods=self.min_history)
        self.M = pd.rolling_mean(X, window=self.window, min_periods=self.min_history) 
    
    def step(self, x, last_b):
        # get sigma and mu matrix
        mu = self.M.ix[x.name]
        sigma = self.S[x.name]
        
        # delete assets with NaN values
        m = mu.copy()
        s = sigma.copy()
        keep = mu.notnull()
        mu = mu[keep]
        
        sigma = sigma.ix[keep, keep]
        
        # replace NaN values with 0
        sigma = sigma.fillna(0.)
    
        # convert to matrices    
        sigma = np.matrix(sigma)
        mu = np.matrix(mu).T
        
        # pure approach - problems with singular matrix
        if not self.long_only:
            sigma_inv = np.linalg.inv(sigma)
            b = (1 + self.r) * sigma_inv * (mu - self.r)
            b = np.ravel(b)
        else:
            while True:
                try:
                    b = self.maximize(mu, sigma, self.r)
                    break
                except ValueError:
                    raise
                    # deal with singularity
                    logging.warning('Singularity problems')
                    sigma = sigma + 0.0001 * np.eye(len(sigma))
        
        # add back values for NaN assets
        b = pd.Series(b, index=keep.index[keep])
        b = b.reindex(keep.index).fillna(0.)
        
        # use Kelly fraction
        b *= self.fraction
        
        return b
    
    
    def maximize(self, mu, sigma, r):
        n = len(mu)
        
        P = matrix((sigma - r*mu*mu.T + (n*r)**2) / (1+r))
        q = matrix(-mu)
        G = matrix(-np.eye(n))
        h = matrix(np.zeros(n))
        
        sol = solvers.qp(P, q, G, h)
        return np.squeeze(sol['x'])
    
    
    def plot_fraction(self, S, fractions=np.linspace(0.,2.,10), **kwargs):
        """ Plot graph with Kelly fraction on x-axis and total wealth on y-axis. 
        :param S: Stock prices.
        :param fractions: List (ndarray) of fractions used.
        """
        wealths = [] 
        for fraction in fractions:
            self.fraction = fraction
            wealths.append(self.run(S).total_wealth)
            
        ax = pd.Series(wealths, index=fractions, **kwargs).plot(**kwargs)
        ax.set_xlabel('Kelly Fraction')
        ax.set_ylabel('Total Wealth')
        return ax


# use case
if __name__ == '__main__':
    tools.quickrun(Kelly())

