from ..algos import CRP
from .. import tools
import numpy as np
import pandas as pd


class BestMarkowitz(CRP):
    """ Optimal Markowitz portfolio constructed in hindsight.

    Reference:
        https://en.wikipedia.org/wiki/Modern_portfolio_theory
    """

    PRICE_TYPE = 'ratio'
    REPLACE_MISSING = False

    def __init__(self, global_sharpe=None, **kwargs):
        self.global_sharpe = global_sharpe
        self.opt_markowitz_kwargs = kwargs

    def weights(self, X):
        """ Find optimal markowitz weights. """
        # update frequency
        freq = tools.freq(X.index)

        R = X - 1

        # calculate mean and covariance matrix and annualize them
        sigma = R.cov() * freq

        if self.global_sharpe:
            mu = pd.Series(np.sqrt(np.diag(sigma)) * self.global_sharpe, X.columns)
        else:
            mu = R.mean() * freq

        self.b = tools.opt_markowitz(mu, sigma, **self.opt_markowitz_kwargs)

        return super(BestMarkowitz, self).weights(R)


if __name__ == '__main__':
    tools.quickrun(BestMarkowitz())
