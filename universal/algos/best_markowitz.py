from ..algos import CRP
from .. import tools
import numpy as np


class BestMarkowitz(CRP):
    """ Optimal Markowitz portfolio constructed in hindsight.

    Reference:
        https://en.wikipedia.org/wiki/Modern_portfolio_theory
    """

    PRICE_TYPE = 'log'
    REPLACE_MISSING = False

    def __init__(self, **kwargs):
        self.opt_markowitz_kwargs = kwargs

    def weights(self, X):
        """ Find optimal markowitz weights. """
        # update frequency
        freq = tools.freq(X.index)

        # calculate mean and covariance matrix and annualize them
        mu = X.mean() * freq
        sigma = X.cov() * freq

        self.b = tools.opt_markowitz(mu, sigma, **self.opt_markowitz_kwargs)

        return super(BestMarkowitz, self).weights(X)


if __name__ == '__main__':
    tools.quickrun(BestMarkowitz())
